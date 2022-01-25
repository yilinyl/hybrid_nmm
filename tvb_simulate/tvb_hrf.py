#!/usr/bin/env python3
import numpy as np
import pickle
import scipy.io
from tvb.simulator.lab import *
from tvb.simulator.simulator import *
from tvb.simulator.models.wilson_cowan import WilsonCowan
from tvb.basic import readers


# conn = connectivity.Connectivity.from_file(source_file='./data/connectivity_76.zip')


def sim_bold_signal(conn):
    sim_bold = simulator.Simulator(
        model=models.WilsonCowan(variables_of_interest=('E', 'I',)),
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([0.5 / 50.0])),
        integrator=integrators.EulerStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([1e-5]))),
        #     monitors=(monitors.TemporalAverage(period=1.),),
        monitors=(monitors.Bold(period=50, ),),  # an integral multiple of 500 (ms)
        simulation_length=2e3).configure()  # t_max = simulation_length * dt

    (sim_time, sim_data), = sim_bold.run()
    with open("./bold_sim_history.pkl", 'wb') as f_pkl:
        pickle.dump({
            'history': sim_bold.history.buffer,  # (536, 2, 86, 1)
            'current_step': sim_bold.current_step,  # 20000
            'current_state': sim_bold.current_state,  # (2, 86, 1)
            'bold_inner': sim_bold.monitors[0]._interim_stock,  # (40, 2, 86, 1)
            'bold': sim_bold.monitors[0]._stock,  # (5000, 2, 86, 1)
            'rng': sim_bold.integrator.noise.random_stream.get_state()
        }, f_pkl)
    return sim_time, sim_data  # n_observations: simulation_length / monitor.period


def load_sc(sc_mat: np.array, dist_data: dict):
    sc_conn = connectivity.Connectivity()
    n_region = sc_mat.shape[0]
    tract_len_tmp = np.random.rand(n_region, n_region)
    np.fill_diagonal(tract_len_tmp, 0)

    sc_conn.region_labels = dist_data['labels'][0]
    sc_conn.weights = sc_mat
    sc_conn.number_of_regions = n_region
    sc_conn.centres = dist_data['centroids']
    sc_conn.tract_lengths = dist_data['D']  # Euclidean distance used here

    return sc_conn


if __name__ == "__main__":
    dist_data_raw = scipy.io.loadmat("../data/fs86_pairwise_distances.mat")
    sc_data_all = scipy.io.loadmat('../data/sc_ifod2act_fs86_997subj.mat')
    sc_mat0 = sc_data_all['sift2volnorm'][0][0]

    sc_obj = load_sc(sc_mat0, dist_data_raw)  # structural connectivity object
    time, data = sim_bold_signal(sc_obj)
    print(data.shape)

