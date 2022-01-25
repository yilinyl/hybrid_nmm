#!/usr/bin/env python3
import numpy as np
# from .tvb_library.tvb.datatypes import equations


class FirstOrderVolterra:

    def __init__(self, tau_s=0.8, tau_f=0.4, k_1=5.6, v_0=0.02):
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.k_1 = k_1
        self.v_0 = v_0

    def eval(self, var):
        return 1/3. * np.exp(-0.5 * (var / self.tau_s)) * \
               (np.sin(np.sqrt(1. / self.tau_f - 1./(4. * self.tau_s**2)) * var)) / \
               (np.sqrt(1./self.tau_f - 1./(4. * self.tau_s**2)))


class Bold:
    """
    Base class for the Bold monitor.

    **Attributes**

        hrf_kernel: the haemodynamic response function (HRF) used to compute
                    the BOLD (Blood Oxygenation Level Dependent) signal.

        length    : duration of the hrf in seconds.

        period    : the monitor's period
    """
    def __init__(self, period=2000, hrf_length=20000):

        self.period = period  # sampling period [ms]
        self.hrf_kernel = FirstOrderVolterra()
        self.hrf_length = hrf_length  # duration of hrf kernel [ms]
        self.dt = None
        self.istep = None

        self.hrf = None
        self.voi_idx = None  # indices of variables of interest
        self._interim_period = None
        self._interim_istep = None
        self._interim_stock = None
        self._stock_steps = None
        self._stock_time = None
        self._stock_sample_rate = 2 ** -2

    def compute_hrf(self):
        """
        Compute the hemodynamic response function.

        """
        magic_number = self.hrf_length  # * 0.8  # truncates G, volterra kernel, once ~zero
        # Length of history needed for convolution in steps @ _stock_sample_rate
        required_history_length = self._stock_sample_rate * magic_number  # 5000
        self._stock_steps = np.ceil(required_history_length).astype(int)
        stock_time_max = magic_number / 1000.0  # 20 [s]
        stock_time_step = stock_time_max / self._stock_steps  # [s]
        self._stock_time = np.arange(0.0, stock_time_max, stock_time_step)  # [s]
        # Compute the HRF kernel
        G = self.hrf_kernel.eval(self._stock_time)
        # Reverse it, need it into the past for matrix-multiply of stock
        G = G[::-1]
        self.hrf = G[np.newaxis, :]
        # Interim stock configuration
        self._interim_period = 1.0 / self._stock_sample_rate  # (4) period in ms
        self._interim_istep = int(round(self._interim_period / self.dt))  # interim period in integration time steps

    def _config_vois(self, n_voi):
        # variable of interest
        self.voi_idx = np.r_[:n_voi]  # [0, 1] in the case of 2 vois

    def _config_time(self, n_nodes, dt=0.1):
        self.dt = dt
        self.istep = round(self.period / self.dt)
        self.compute_hrf()
        sample_shape = self.voi_idx.shape[0], n_nodes
        self._interim_stock = np.zeros((self._interim_istep,) + sample_shape)
        self._stock = np.zeros((self._stock_steps,) + sample_shape)

    def config_for_sim(self, n_voi, n_nodes, dt=0.1):
        """Configure monitor for given simulator.

        Grab the Simulator's integration step size. Set the monitor's variables
        of interest based on the Monitor's 'variables_of_interest' attribute, if
        it was specified, otherwise use the 'variables_of_interest' specified
        for the Model. Calculate the number of integration steps (isteps)
        between returns by the record method. This method is called from within
        the the Simulator's configure() method.

        """
        self._config_vois(n_voi)
        self._config_time(n_nodes, dt)

    def sample(self, step, state):  # state at single time spot
        # Update the interim-stock at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi_idx, :]
        # At stock's period update it with the temporal average of interim-stock
        if step % self._interim_istep == 0:
            avg_interim_stock = np.mean(self._interim_stock, axis=0)
            self._stock[((step // self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock
        # At the monitor's period, apply the heamodynamic response function to
        # the stock and return the resulting BOLD signal.
        if step % self.istep == 0:
            time = step * self.dt
            hrf = np.roll(self.hrf,
                             ((step // self._interim_istep % self._stock_steps) - 1),
                             axis=1)
            # if isinstance(self.hrf_kernel, FirstOrderVolterra):
            k1_V0 = self.hrf_kernel.k_1 * self.hrf_kernel.v_0
            bold = (np.dot(hrf, self._stock.transpose((1, 0, 2))) - 1.0) * k1_V0

            bold = bold.reshape(self._stock.shape[1:])
            return [time, bold]


if __name__ == "__main__":
    # t_seq = np.arange(0.0, 20, 20/5000)
    # hrf_kernel = FirstOrderVolterra()
    # hrf = hrf_kernel.eval(t_seq)
    import pickle
    import matplotlib.pyplot as plt

    with open('~/Documents/Cornell/Research/CoCoLab/sim_ei_data_5e3_86r.pkl', 'rb') as f_pkl:
        time, sim_ei = pickle.load(f_pkl)  # sim_ei: (T, n_var, n_region, 1)
    state0 = sim_ei[0, :, :, 0]  # shape: (n_var, n_region)
    n_step, n_var, n_node, _ = sim_ei.shape
    # n_var = state0.shape[0]
    # n_node = state0.shape[1]

    bold_monitor = Bold(period=50)
    bold_monitor.config_for_sim(n_voi=n_var, n_nodes=n_node, dt=1)
    t_list = []
    x_list = []
    for s in range(n_step):
        output = bold_monitor.sample(step=s+1, state=sim_ei[s, :, :, 0])
        if output is not None:
            tt, bold = output
            t_list.append(tt)
            x_list.append(bold)

    t_arr = np.array(t_list)
    x_arr = np.array(x_list)

    plt.figure()
    plt.plot(t_arr, x_arr[:, 0, :], color='k', alpha=0.2);
    plt.plot(t_arr, x_arr[:, 1, :], color='b', alpha=0.2);
    plt.show()
