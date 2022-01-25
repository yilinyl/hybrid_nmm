#!/usr/bin/env python3
import os
import glob
import time
import pickle
import scipy.io
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torchdiffeq import odeint
from models import NeuralMass, RunningAverageMeter
from models import Bold
from utils import *

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


# Manual integration using Stochastic Euler method
# TODO:
#  1) test with single individual case
#  2) load all SC data in batch


def pipeline(sc_mat, fc_true, period, n_regions=None, t0=0, sim_length=1e3, lr=0.1,
             dt=0.1, voi=('E', 'I'), t_setup=0, epoch=1000, noise_sig=1e-3, w_setup=0.01, fig_dir='./figures',
             print_freq=10, plot=True, subject_id='', checkpt_path='./checkpoint.pt', **kwargs):
    if not n_regions:
        n_regions = sc_mat.shape[0]
    # n_obs = round(sim_length / obs_interim_t)
    t_max = sim_length * dt
    t_obs_span = torch.linspace(t0, t_max, int(sim_length))
    # n_istep = round(obs_interim_t / dt)
    e_init = torch.randn(size=(n_regions,))  # TODO: also defined in NeuralMass module
    i_init = torch.randn(size=(n_regions,))
    state = (e_init, i_init)
    # Initialize neural mass model
    neural_mass_ode = NeuralMass(sc_mat=sc_mat, n_regions=n_regions,
                                 init_e=e_init, init_i=i_init, **kwargs)

    # Initialize hemodynamic response function
    bold_monitor = Bold(period=period)
    # Initialize optimizer
    # optimizer = torch.optim.SGD(neural_mass_ode.parameters(), lr=lr)
    optimizer = torch.optim.Adam(neural_mass_ode.parameters(), lr=lr)
    loss_meter = RunningAverageMeter()
    loss_func = nn.MSELoss()
    loss_all = list()
    for e in range(epoch):
        start_time = time.time()
        print("----- Epoch [{e}/{epoch}] -----".format(e=e + 1, epoch=epoch))
        bold_monitor.config_for_sim(n_voi=len(voi), n_nodes=n_regions, dt=dt)
        optimizer.zero_grad()
        t_list = []
        x_list = []
        interim_t = t0 + np.arange(1, int(sim_length)+1) * dt
        for ti in range(1, int(sim_length)+1):
            # tt_span = torch.linspace(t0, float(t_obs_span[ti]), n_istep)  # expand each interim_t interval
            # tt_span = torch.tensor([t_obs_span[ti-1], t_obs_span[ti]])
            # sim_e, sim_i = odeint(neural_mass_ode, y0=(e_init, i_init), t=tt_span)
            t_tmp = t0 + ti * dt
            d_e, d_i = neural_mass_ode(t=t_tmp, state_vars=state)

            # Add noise to integrated results
            # sim_e = state[0] + d_e * dt
            # sim_i = state[1] + d_i * dt
            sim_ei = torch.stack([state[0] + d_e * dt, state[1] + d_i * dt])
            sim_noise = torch.randn(size=sim_ei.shape) * np.sqrt(2*noise_sig)
            sim_ei = sim_ei + sim_noise
            state = (sim_ei.detach()[0, :], sim_ei.detach()[1, :])
            # if t_tmp <= t_setup:  # calculate BOLD only after t_setup
            #     continue

            output = bold_monitor.sample(step=ti, state=sim_ei)  # current step
            # print(ti)
            if output is not None:
                tt, sim_bold = output
                t_list.append(tt)
                x_list.append(sim_bold)

            # t0 = float(t_obs_span[ti])  # update t0

        bold_t = torch.tensor(t_list)
        bold_x = torch.stack(x_list)  # shape: (tt, n_var, n_regions)

        # fc_setup = torch.corrcoef(bold_x.transpose(dim0=0, dim1=2)[:, 0, :])

        # bold_x_clean = torch.cat([w_setup * bold_x[(bold_t <= t_setup), 0, :],
        #                           bold_x[(bold_t > t_setup), 0, :]], dim=0)  # reduce impact of setup period
        # x_setup = torch.zeros_like(bold_x)
        # x_setup[(bold_t <= t_setup), :, :] = bold_x[(bold_t <= t_setup), :, :]
        # bold_x_clean = bold_x - x_setup
        fc_pred = torch.corrcoef(bold_x.transpose(dim0=0, dim1=2)[:, 0, (bold_t >= t_setup)])  # corr: rows are variables, cols are obs
        # fc_combine = fc_pred + 0.5 * fc_setup
        up_tri_idx = torch.triu_indices(n_regions, n_regions)
        exclude_diag = up_tri_idx[0] != up_tri_idx[1]
        fc_pred_uptri = fc_pred[up_tri_idx[0][exclude_diag], up_tri_idx[1][exclude_diag]]
        fc_true_uptri = fc_true[up_tri_idx[0][exclude_diag], up_tri_idx[1][exclude_diag]]
        # loss = loss_func(fc_true, (1-w_setup) * fc_pred + w_setup * fc_setup)  # modified loss
        loss = loss_func(fc_true_uptri, fc_pred_uptri)  # compare upper triangle only
        # torch.cuda.synchronize()
        epoch_time = time.time()
        print("Loss: {}, Elapsed time: {}".format(loss, epoch_time-start_time))
        if plot:
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)
            if (e + 1) % print_freq == 0:
                # Visualize intermediate FC matrix
                plt.figure()
                sns.heatmap(fc_pred.detach().numpy(), vmin=-1, vmax=1, cmap='PiYG_r')
                plt.title("Predicted FC (epoch {})".format(e + 1))
                plt.savefig(os.path.join(fig_dir, "{}fc_pred_e{}.png".format(subject_id, e + 1)))

                # Intermediate Bold time series
                # plt.figure()
                # plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 0, :], color='k', alpha=0.2);
                # plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 1, :], color='b', alpha=0.2);
                # plt.title("Bold time series (epoch {})".format(e + 1))
                # plt.savefig(os.path.join(fig_dir, "boldts_e{}.png".format(e + 1)))

        # if e > 0 and abs(loss - loss_all[-1]) <= eps_stop:  # early termination
        #     loss_all.append(float(loss))
        #     break

        loss_all.append(float(loss))
        loss.backward()

        plt.figure()
        plot_grad_flow(neural_mass_ode.named_parameters())
        plt.savefig(os.path.join(fig_dir, "grad_flow_e{}.png".format(e + 1)))

        optimizer.step()

    plt.figure()
    plt.plot(np.arange(epoch), np.array(loss_all))
    plt.title("Loss")
    plt.savefig(os.path.join(fig_dir, "{}_loss_all.png".format(subject_id)))

    # plt.figure()
    # sns.heatmap(fc_pred.detach().numpy(), vmin=-1, vmax=1, cmap='PiYG_r')
    # plt.title("Predicted FC (epoch {} id={})".format(e + 1, subject_id))
    # plt.savefig(os.path.join(fig_dir, "{}_fc_pred_e{}.png".format(subject_id, e + 1)))

    # plt.figure()
    # plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 0, :], color='k', alpha=0.2);
    # plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 1, :], color='b', alpha=0.2);
    # plt.title("Bold time series (final)")
    # plt.savefig(os.path.join(fig_dir, "bold_final.png"))

    # Save checkpoint (final)
    torch.save({
        'epoch': e,
        'model_state_dict': neural_mass_ode.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bold_t': bold_t.detach(),
        'bold_x': bold_x.detach(),
        'fc_pred': fc_pred.detach(),
        'fc_true': fc_true.detach(),
        'loss': loss,
    }, checkpt_path)

    return fc_pred
    # for name, param in neural_mass_ode.named_parameters():
    #     print("{name}: {data}".format(name=name, data=param.data))


def get_args():
    # system args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/home/ubuntu/coco-lab/')
    parser.add_argument('--sc_path', default='data/sc_ifod2act_fs86_997subj.mat')
    parser.add_argument('--fc_dir', default='data/FCmat_hpf_fs86_gsr_FCcov_concat/')
    parser.add_argument('--fc_path', default='data/FCmat_hpf_fs86_gsr_FCcov_concat/'
                                             '100206_concat_fmriclean_hpf_fs86_gsr_FCcov.mat')
    parser.add_argument('--fig_dir', default='./figures')
    parser.add_argument('--checkpt_path', default='./checkpoint.pt')
    parser.add_argument('--result_dir', default='./')
    parser.add_argument('--subject_start', type=int, default=0)
    parser.add_argument('--n_subject', type=int, default=5)

    # hyper-parameters for training
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--w_setup', type=float, default=0.01)
    parser.add_argument('--print_freq', type=int, default=10)


    # Simulator parameters
    parser.add_argument('--voi', default='E,I', help='Variables of interest')
    parser.add_argument('--dt', type=float, default=1)
    parser.add_argument('--sim_length', type=float, default=2e3, help='Simulation length')
    parser.add_argument('--period', type=float, default=50, help='Period of Bold monitor')
    parser.add_argument('--t_setup', type=float, default=0,
                        help='Only time period after it will be accounted for FC')
    parser.add_argument('--noise_sig', type=float, default=1e-3)

    # Initial parameter values for neural mass model
    # Excitatory
    parser.add_argument('--tau_e', type=float, default=10.0)
    parser.add_argument('--a_e', type=float, default=1.)
    parser.add_argument('--b_e', type=float, default=1.)
    parser.add_argument('--c_e', type=float, default=1.)
    parser.add_argument('--theta_e', type=float, default=0.0)
    parser.add_argument('--r_e', type=float, default=1.)
    parser.add_argument('--k_e', type=float, default=1.)
    parser.add_argument('--alpha_e', type=float, default=1.)

    # Inhibitory
    parser.add_argument('--tau_i', type=float, default=10.0)
    parser.add_argument('--a_i', type=float, default=1.)
    parser.add_argument('--b_i', type=float, default=1.)
    parser.add_argument('--c_i', type=float, default=1.)
    parser.add_argument('--theta_i', type=float, default=0.0)
    parser.add_argument('--r_i', type=float, default=1.)
    parser.add_argument('--k_i', type=float, default=1.)
    parser.add_argument('--alpha_i', type=float, default=1.)

    # External inputs
    parser.add_argument('-P', type=float, default=0.0)
    parser.add_argument('-Q', type=float, default=0.0)

    # Transaction weights
    parser.add_argument('--c_ee', type=float, default=12.)
    parser.add_argument('--c_ei', type=float, default=4.)
    parser.add_argument('--c_ie', type=float, default=13.)
    parser.add_argument('--c_ii', type=float, default=11.)

    parser.add_argument('--eta', type=float, default=20.)  # global coupling strength

    args = parser.parse_args()

    return args


def init_ode_pars(args, n_regions, regional_pars, global_pars):
    arg_dict = vars(args)
    for par in regional_pars:
        arg_dict.update({par: torch.full(size=(n_regions,), fill_value=arg_dict[par]) +
                              torch.randn(size=(n_regions,))*0.1})

    for par in global_pars:
        arg_dict.update({par: torch.tensor([arg_dict[par]])})

    return arg_dict


def plot_fc(fc_mat, title, cmap='PiYG_r'):
    plt.figure()
    sns.heatmap(fc_mat, vmin=-1, vmax=1, cmap=cmap)
    plt.title(title)
    plt.show()


def run(regional_pars, global_pars, verbose=True):
    args = get_args()
    sc_data_all = scipy.io.loadmat(os.path.join(args.root_dir, args.sc_path))
    fc_data = scipy.io.loadmat(os.path.join(args.root_dir, args.fc_path))

    sc_mat0 = torch.tensor(sc_data_all['sift2volnorm'][0][0])  # test for subset

    fc_mat0 = torch.tensor(fc_data['C'])
    n_regions = sc_mat0.shape[0]

    voi_tup = tuple(s.strip() for s in args.voi.split(','))

    arg_dict = vars(args)
    arg_dict.update({'n_regions': n_regions, 'voi': voi_tup})
    if verbose:
        print("============= Initial Settings ===============")
        print("Learning rate: {}".format(arg_dict['lr']))
        print("Simulation length: {}".format(arg_dict['sim_length']))

        for key in regional_pars:
            print("* [Regional] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        for key in global_pars:
            print("* [Global] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        print("==============================================")

    arg_dict.update(init_ode_pars(args, n_regions, regional_pars, global_pars))
    # arg_dict.update({'eta': torch.tensor([arg_dict['eta']])})

    fc_pred = pipeline(sc_mat0, fc_mat0, **arg_dict)

    # save final prediction to file
    # with open(args.fc_pred_path, 'wb') as f_pkl:
    #     pickle.dump(fc_pred, f_pkl)


def run_avg(regional_pars, global_pars, verbose=True):
    args = get_args()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    sc_data_all = scipy.io.loadmat(os.path.join(args.root_dir, args.sc_path))
    is_missing = sc_data_all['ismissing']
    select_idx = np.where(is_missing == 0)[1]
    print("# Optimization using average connectivity of {} individuals".format(select_idx.shape[0]))
    sc_mat_select = sc_data_all['sift2volnorm'].squeeze()[select_idx]  # exclude missing item
    sc_mat_avg = np.mean(sc_mat_select)
    print(sc_mat_avg.shape)
    subject_ids = sc_data_all['subject'].squeeze()

    n_regions = sc_mat_select[0].shape[0]
    voi_tup = tuple(s.strip() for s in args.voi.split(','))
    arg_dict = vars(args)

    arg_dict.update({'n_regions': n_regions, 'voi': voi_tup})
    if verbose:
        print("============= Initial Settings ===============")
        print("Learning rate: {}".format(arg_dict['lr']))
        print("Simulation length: {}".format(arg_dict['sim_length']))

        for key in regional_pars:
            print("* [Regional] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        for key in global_pars:
            print("* [Global] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        print("==============================================")

    arg_dict.update(init_ode_pars(args, n_regions, regional_pars, global_pars))
    fc_avg_path = os.path.join(arg_dict['root_dir'], 'data/fc_avg.pkl')
    if os.path.exists(fc_avg_path):
        with open(fc_avg_path, 'rb') as f_pkl:
            fc_avg_mat = pickle.load(f_pkl)
    else:
        fc_true_all = list()
        for i in select_idx:
            sub_id = subject_ids[i][0]
            fc_file_l = glob.glob(os.path.join(arg_dict['root_dir'], arg_dict['fc_dir'], sub_id + '_Retest*mat'))
            if not fc_file_l:
                fc_file_l = glob.glob(os.path.join(arg_dict['root_dir'], arg_dict['fc_dir'], sub_id + '_concat*mat'))
            fc_file = fc_file_l[0]
            fc_data = scipy.io.loadmat(fc_file)
            fc_true_all.append(fc_data['C'])
        fc_mat_all = np.array(fc_true_all)
        print(fc_mat_all.shape)
        fc_avg_mat = np.mean(fc_mat_all, axis=0)
        print(fc_avg_mat.shape)
        with open(os.path.join(arg_dict['root_dir'], 'data/fc_avg.pkl'), 'wb') as f_pkl:
            pickle.dump(fc_avg_mat, f_pkl)

    sc_mat = torch.tensor(sc_mat_avg)
    fc_mat = torch.tensor(fc_avg_mat)

    fc_pred = pipeline(sc_mat, fc_mat, **arg_dict)


def run_all(regional_pars, global_pars, verbose=True):
    args = get_args()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    sc_data_all = scipy.io.loadmat(os.path.join(args.root_dir, args.sc_path))
    missing_id = np.where(sc_data_all['ismissing'])[1]
    sc_mat_all = sc_data_all['sift2volnorm'].squeeze()
    subject_ids = sc_data_all['subject'].squeeze()
    # n_subject = subject_ids.shape[0]
    n_subject = args.n_subject

    voi_tup = tuple(s.strip() for s in args.voi.split(','))
    n_regions = sc_mat_all[0].shape[0]
    arg_dict = vars(args)
    arg_dict.update({'n_regions': n_regions, 'voi': voi_tup})

    if verbose:
        print("============= Initial Settings ===============")
        print("Learning rate: {}".format(arg_dict['lr']))
        for key in regional_pars:
            print("* [Regional] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        for key in global_pars:
            print("* [Global] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        print("==============================================")
    arg_dict.update(init_ode_pars(args, n_regions, regional_pars, global_pars))
    fc_all = list()
    subject_all = list()

    for i in range(args.subject_start, args.subject_start+n_subject):
        subject = subject_ids[i]
        sub_id = subject[0]
        print("======= Subject {sid} [{i} / {total}] ========".format(sid=sub_id, i=i, total=n_subject))
        subject_all.append(sub_id)
        arg_dict.update({'checkpt_path': os.path.join(arg_dict['result_dir'], sub_id + '_checkpt.pt'),
                         'subject_id': sub_id})
        fc_file_l = glob.glob(os.path.join(arg_dict['root_dir'], arg_dict['fc_dir'], sub_id + '_Retest*mat'))
        if not fc_file_l:
            fc_file_l = glob.glob(os.path.join(arg_dict['root_dir'], arg_dict['fc_dir'], sub_id + '_concat*mat'))
        fc_file = fc_file_l[0]
        fc_data = scipy.io.loadmat(fc_file)
        fc_mat = torch.tensor(fc_data['C'])
        sc_mat = torch.tensor(sc_mat_all[i])

        fc_pred = pipeline(sc_mat, fc_mat, **arg_dict)
        with open(os.path.join(arg_dict['result_dir'], sub_id+'_fc_pred.pkl'), 'wb') as f_pkl:
            pickle.dump({'fc_pred': fc_pred, 'subject': sub_id}, f_pkl)
        fc_all.append(fc_pred)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    # reg_pars = 'c_ee, c_ei, c_ie, c_ii, ' \
    #            'alpha_e, theta_e, alpha_i, theta_i, ' \
    #            'tau_e, a_e, b_e, c_e, k_e, r_e,' \
    #            'tau_i, a_i, b_i, c_i, k_i, r_i'.split(',')
    # reg_pars_tup = tuple(s.strip() for s in reg_pars)
    reg_pars_tup = tuple()

    fix_pars = 'c_ee, c_ei, c_ie, c_ii, ' \
               'alpha_e, theta_e, alpha_i, theta_i, ' \
               'tau_e, a_e, b_e, c_e, k_e, r_e,' \
               'tau_i, a_i, b_i, c_i, k_i, r_i, P, Q, eta'.split(',')
    # fix_pars = 'P, Q, eta'.split(',')
    fix_pars_tup = tuple(s.strip() for s in fix_pars)

    run(reg_pars_tup, fix_pars_tup, verbose=True)
    print('Done!')

