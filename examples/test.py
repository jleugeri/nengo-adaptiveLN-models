"""
Copyright 2015 by Johannes Leugering (jleugeri@uos.de)

This file is part of the adaptive LN Neurons package.

    The adaptive LN Neurons package is free software: 
    you can redistribute it and/or modify it under the terms of the 
    GNU Lesser General Public License as published by the 
    Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the 
    GNU Lesser General Public License along with this package
    in files named COPYING and COPYING.LESSER.
    If not, see <http://www.gnu.org/licenses/>.
"""
import nengo
from nengo_adaptiveLN_models import *
import numpy as np
import matplotlib.pyplot as pp

model = nengo.Network()
num_neurons = 1

# mean and standard deviation for 3 different normal input distributions
mu  = 1.5,-1.0, 0.5
sig = 1.0, 0.5, 2.0

def stimulus(t):
    """ create a stimulus that cycles through the 3 different input distributions all 300s """
    i = int((t-0.1)/100) % 3
    return np.random.randn()*sig[i] + mu[i]

# set position and scale of the neurons (these can be used instead of bias and
# gain to get diverse output distributions for different neurons, if desired)
position = np.zeros((num_neurons,)) #randn(num_neurons)*0.5
scale    = np.ones((num_neurons,))  #0.5+rand(num_neurons)

#neuron_model = AdaptiveLNuniform
neuron_model = AdaptiveLNlogNormal

# Set time-constant of the adaptation variables in seconds
tau_adapt= 10.0

with model:
    stim    = nengo.Node(stimulus)
    # intialize the ensemble; manually set bias and gain to zeros and ones,
    # respectively, to side-step them alltogether
    ens = nengo.Ensemble(n_neurons=num_neurons, dimensions=1,
                            neuron_type=neuron_model(tau_adapt=tau_adapt, position=position, scale=scale),
                         bias=np.zeros_like(position), gain=np.ones_like(scale))

    # make a connection with very fast synapses to make comparisons with the
    # target values of the adaptation variables easy (slower time-constants reduce
    # the variance of the neurons' input distributions)
    nengo.Connection(stim,ens, transform=1, synapse=1e-5)

    # probe the adaptation variables and the output rate
    probe_mu = nengo.Probe(ens.neurons, 'mu_in')
    probe_sq = nengo.Probe(ens.neurons, 'sq_in')
    probe_r  = nengo.Probe(ens.neurons, 'rates')


# Plot the result of 300s of simulation
tmax = 300
# each input distribution is shown for tseg=100s
tseg = int(tmax/3.0)

# Simulate!
sim = nengo.Simulator(model)
sim.run(tmax)

# retrieve data
dat_mu = sim.data[probe_mu]
dat_sq = sim.data[probe_sq]
dat_r  = sim.data[probe_r]

# plot traces of estimated mean
fig = pp.figure()
t   = np.linspace(0,tmax,dat_mu.shape[0])
ax1 = fig.add_subplot(411)
ax1.plot(t, dat_mu, label="est. mean")
ax1.set_title("mean")
ax1.set_xlabel(r"[mV]")
ax1.set_xlabel("time [s]")
for seg in range(3):
    ax1.plot(seg*tseg+ np.vstack([np.zeros((1,num_neurons)),tseg+np.zeros((1,num_neurons))]), mu[seg] + np.zeros((2,num_neurons)),lw=2, label="real mean #%d" % seg)
ax1.legend()

# plot traces of estimated standard deviation
ax2 = fig.add_subplot(412)
ax2.plot(t, np.sqrt(dat_sq-dat_mu**2), label="est. std")
ax2.set_title("standard deviation")
ax2.set_ylabel(r"$[mV^2]$")
ax2.set_xlabel("time [s]")
for seg in range(3):
    ax2.plot(seg*tseg+ np.vstack([np.zeros((1,num_neurons)),tseg+np.zeros((1,num_neurons))]), sig[seg] + np.zeros((2,num_neurons)),lw=2, label="real std. #%d" % seg)
ax2.legend()

# plot traces of the neurons' output rates
ax3 = fig.add_subplot(413)
ax3.plot(t, dat_r)
ax3.set_title("firing rate")
ax3.set_ylabel("rate [Hz]")
ax3.set_xlabel("time [s]")

# plot histograms of the output in the second half of each of the three periods
for i in range(3):
    ax = fig.add_subplot(4,3,10+i)
    ax.hist(dat_r[tseg/sim.dt*(i+0.5):tseg/sim.dt*(i+1)],50, normed=True)
    ax.set_title("distribution of outputs")
    ax.set_ylabel("rel. freq.")
    ax.set_xlabel("outputs in seg. #%d" % i)

fig.subplots_adjust(hspace=0.4)
pp.show()
