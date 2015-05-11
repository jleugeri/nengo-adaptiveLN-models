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
from pylab import *

model = nengo.Network()
num_neurons = 2

mu  = 1.5,-1.0, 0.5
sig = 1.0, 0.5, 2.0

def stimulus(t):
    i = int((t-0.1)/10)
    return randn()*sig[i] + mu[i]

with model:
    stim = nengo.Node(stimulus)
    position = randn(num_neurons)*0.5
    scale = 0.5+rand(num_neurons)
    ens = nengo.Ensemble(n_neurons=num_neurons, dimensions=1,
                            neuron_type=AdaptiveLNlogNormal(tau_adapt=1.0, position=position, scale=scale), 
                         bias=np.zeros_like(position), gain=np.ones_like(scale))

    nengo.Connection(stim,ens, transform=1, synapse=1e-5)
    probe_mu = nengo.Probe(ens.neurons, 'mu_in')
    probe_sq = nengo.Probe(ens.neurons, 'sq_in')
    probe_r  = nengo.Probe(ens.neurons, 'rates')

tmax = 30
tseg = int(tmax/3.0)

sim = nengo.Simulator(model)
sim.run(tmax)
dat_mu = sim.data[probe_mu]
dat_sq = sim.data[probe_sq]
dat_r  = sim.data[probe_r]

t = linspace(0,tmax,dat_mu.shape[0])
ion()
subplot(311)
plot(t, dat_mu)
for seg in range(3):
    plot(seg*tseg+ vstack([zeros((1,num_neurons)),tseg+zeros((1,num_neurons))]), mu[seg] + zeros((2,num_neurons)),lw=2)

subplot(312)
plot(t, sqrt(dat_sq-dat_mu**2))
for seg in range(3):
    plot(seg*tseg+ vstack([zeros((1,num_neurons)),tseg+zeros((1,num_neurons))]), sig[seg] + zeros((2,num_neurons)),lw=2)

subplot(313)
plot(t, dat_r)
