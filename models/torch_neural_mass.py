#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt

import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchdiffeq import odeint, odeint_adjoint

# torch.set_default_dtype(torch.float64)


class NeuralMass(nn.Module):

    def __init__(self, tau_e, alpha_e, theta_e, a_e, b_e, c_e, k_e, r_e,
           tau_i, alpha_i, theta_i, a_i, b_i, c_i, k_i, r_i,
           c_ee, c_ei, c_ie, c_ii, n_regions, eta, sc_mat, P, Q, init_e=None, init_i=None, adjoint=False, **kwargs):
        super().__init__()
        self.t0 = torch.tensor([0.0])
        self.regions = n_regions
        self.init_e = init_e
        self.init_i = init_i
        if init_e == None:
            self.init_e = Variable(torch.rand((self.regions,)))
        if init_i == None:
            self.init_i = Variable(torch.rand((self.regions,)))

        self.sigmoid = nn.Sigmoid()

        self.tau_e = nn.Parameter(tau_e)
        self.alpha_e = nn.Parameter(alpha_e)
        self.theta_e = nn.Parameter(theta_e)
        self.a_e = nn.Parameter(a_e)
        self.b_e = nn.Parameter(b_e)
        self.c_e = nn.Parameter(c_e)
        self.k_e = nn.Parameter(k_e)
        self.r_e = nn.Parameter(r_e)
        self.tau_i = nn.Parameter(tau_i)
        self.alpha_i = nn.Parameter(alpha_i)
        self.theta_i = nn.Parameter(theta_i)
        self.a_i = nn.Parameter(a_i)
        self.b_i = nn.Parameter(b_i)
        self.c_i = nn.Parameter(c_i)
        self.k_i = nn.Parameter(k_i)
        self.r_i = nn.Parameter(r_i)

        self.c_ee = nn.Parameter(c_ee)
        self.c_ei = nn.Parameter(c_ei)
        self.c_ie = nn.Parameter(c_ie)
        self.c_ii = nn.Parameter(c_ii)

        # External inputs
        self.P = nn.Parameter(P)
        self.Q = nn.Parameter(Q)

        self.eta = nn.Parameter(eta)
        self.sc_mat = sc_mat

        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state_vars):
        # Wilson Cowan model
        state_e, state_i = state_vars  # tuple (E, I)
        x_e = self.alpha_e * (self.c_ee * state_e - self.c_ei * state_i + self.P - self.theta_e +
                              self.eta * torch.matmul(self.sc_mat, state_e).mean())
        x_i = self.alpha_i * (self.c_ie * state_e - self.c_ii * state_i + self.Q - self.theta_i)

        s_e = self.c_e * self.sigmoid(-self.a_e * (x_e - self.b_e))
        s_i = self.c_i * self.sigmoid(-self.a_i * (x_i - self.b_i))

        d_e = (-state_e + (self.k_e - self.r_e * state_e) * s_e) / self.tau_e
        d_i = (-state_i + (self.k_i - self.r_i * state_i) * s_i) / self.tau_i

        return d_e, d_i

    def get_initial_state(self):
        state_vars = (self.init_e, self.init_i)
        return self.t0, state_vars

    def get_time_range(self, t_max, n_step):
        return torch.range(float(self.t0[0]), t_max, n_step)

    def simulate(self, state_vars, t_max, n_step):
        t_obs = self.get_time_range(t_max, n_step)

        t0, state = self.get_initial_state()
        state_e = [state[0][None]]
        state_i = [state[1][None]]
        times = [t0.reshape(-1)]

        for t_state in t_obs:
            tt = torch.linspace(float(t0), float(t_state), int((float(t_state) - float(t0)) * 50))[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, t_state.reshape(-1)])
            solution = odeint(self, state, tt)

            state_e.append(solution[0])
            state_i.append(solution[1])
            times.append(tt)

            state = solution
            t0 = t_state

        return torch.cat(times), torch.cat(state_e, dim=0).reshape(-1), \
               torch.cat(state_i, dim=0).reshape(-1), t_obs


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
