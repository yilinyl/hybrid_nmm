#!/usr/bin/env python3
import os
import pickle
import scipy.io
import argparse
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from torchdiffeq import odeint
from models import NeuralMass, RunningAverageMeter
from models import Bold

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

# writer = SummaryWriter()

# Pipeline for predicting FC from SC (start with single individual)
# TODO:
#  1) test with single individual case
#  2) load all SC data in batch


def pipeline(sc_mat, fc_true, period, n_regions=None, max_t=1e3, dt=0.1, voi=('E', 'I'),
             lr=0.1, epoch=1000, noise_sig=1e-3, fig_dir='./figures', print_freq=10, plot=True, **kwargs):
    if not n_regions:
        n_regions = sc_mat.shape[0]
    n_step = round(max_t / dt)
    # print("")
    t_span = torch.linspace(0, max_t, n_step)
    e_init = torch.autograd.Variable(torch.randn(size=(n_regions,)))  # TODO: also defined in NeuralMass module
    i_init = torch.autograd.Variable(torch.randn(size=(n_regions,)))

    # Initialize neural mass model
    neural_mass_ode = NeuralMass(sc_mat=sc_mat, n_regions=n_regions,
                                 init_e=e_init, init_i=i_init, **kwargs)

    # Initialize hemodynamic response function
    bold_monitor = Bold(period=period)

    # Initialize optimizer
    optimizer = torch.optim.SGD(neural_mass_ode.parameters(), lr=lr)
    loss_meter = RunningAverageMeter()
    loss_func = nn.MSELoss()

    for e in range(epoch):
        optimizer.zero_grad()
        bold_monitor.config_for_sim(n_voi=len(voi), n_nodes=n_regions, dt=dt)

        sim_e, sim_i = odeint(neural_mass_ode, y0=(e_init, i_init), t=t_span)
        sim_ei = torch.stack([sim_e, sim_i]).transpose(dim0=0, dim1=1)
        sim_noise = torch.randn(size=sim_ei.shape) * np.sqrt(2 * noise_sig)
        sim_ei = sim_ei + sim_noise

        n_bold_step = round(period / dt)
        t_list = []
        x_list = []
        for ti in range(n_step):
            # TODO: check shape of odeint outputs
            output = bold_monitor.sample(step=ti+1, state=sim_ei[ti, :, :])
            if output is not None:
                tt, sim_bold = output
                t_list.append(tt)
                x_list.append(sim_bold)
        bold_t = torch.tensor(t_list)
        bold_x = torch.stack(x_list)  # shape: (tt, n_var, n_regions)

        # Note: excitatory only
        fc_pred = torch.corrcoef(bold_x.transpose(dim0=0, dim1=2)[:, 0, :])  # corr: rows are variables, cols are obs
        loss = loss_func(fc_true, fc_pred)
        # writer.add_scalar("Loss", loss, epoch)
        print("----- Epoch [{e}/{epoch}] -----".format(e=e+1, epoch=epoch))
        print("Loss: {}".format(loss))
        if plot:
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)
            if (e+1) % print_freq == 0:
                # Visualize intermediate FC matrix
                plt.figure()
                sns.heatmap(fc_pred.detach().numpy())
                plt.title("Predicted FC (epoch {})".format(e+1))
                plt.savefig(os.path.join(fig_dir, "fc_pred_e{}.png".format(e+1)))

                # Intermediate Bold time series
                plt.figure()
                plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 0, :], color='k', alpha=0.2);
                plt.plot(bold_t.detach().numpy(), bold_x.detach().numpy()[:, 1, :], color='b', alpha=0.2);
                plt.title("Bold time series (epoch {})".format(e + 1))
                plt.savefig(os.path.join(fig_dir, "boldts_e{}.png".format(e + 1)))
            # print(neural_mass_ode.eta)

        loss.backward()
        optimizer.step()
    # for name, param in neural_mass_ode.named_parameters():
    #     print("{name}: {data}".format(name=name, data=param.data))

    return fc_pred


def get_args():
    # system args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='/home/ubuntu/coco-lab/')
    parser.add_argument('--sc_path', default='data/sc_ifod2act_fs86_997subj.mat')
    parser.add_argument('--fc_path', default='data/FCmat_hpf_fs86_gsr_FCcov_concat/'
                                             '100206_concat_fmriclean_hpf_fs86_gsr_FCcov.mat')
    parser.add_argument('--fig_dir', default='./figures')


    # hyper-parameters for training
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)

    # Simulator parameters
    parser.add_argument('--voi', default='E,I', help='Variables of interest')
    # parser.add_argument('--reg_par_str', default='c_ee, c_ei, c_ie, c_ii', help='Specify regional parameters')
    # parser.add_argument('--fix_par_str', default='', help='Specify global parameters')

    parser.add_argument('--dt', type=float, default=1)
    parser.add_argument('--max_t', type=float, default=5e3, help='Maximum monitoring time')
    parser.add_argument('--period', type=float, default=50, help='Period of Bold monitor')

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

    parser.add_argument('--eta', type=float, default=15.)  # global coupling strength

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
        for key in regional_pars:
            print("* [Regional] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        for key in global_pars:
            print("* [Global] {par_name}: {init_value}".format(par_name=key, init_value=arg_dict[key]))
        print("==============================================")

    arg_dict.update(init_ode_pars(args, n_regions, regional_pars, global_pars))
    # arg_dict.update({'eta': torch.tensor([arg_dict['eta']])})

    fc_pred = pipeline(sc_mat0, fc_mat0, **arg_dict)

    return fc_pred


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    reg_pars = 'c_ee, c_ei, c_ie, c_ii, ' \
               'alpha_e, theta_e, alpha_i, theta_i, ' \
               'tau_e, a_e, b_e, c_e, k_e, r_e, ' \
               'tau_i, a_i, b_i, c_i, k_i, r_i'.split(',')
    reg_pars_tup = tuple(s.strip() for s in reg_pars)
    # reg_pars_tup = tuple()

    fix_pars = 'P, Q, eta'.split(',')
    fix_pars_tup = tuple(s.strip() for s in fix_pars)

    fc_est = run(reg_pars_tup, fix_pars_tup, verbose=True)

    print('Done!')

