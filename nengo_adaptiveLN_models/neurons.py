# -*- coding: utf-8 -*-
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

==============================================================================

.. module:: neurons
    :platform: Unix, Windows
    :synopsis: Defines adaptive neuron models of the linear-non-linear variety. 
        They map from normally distributed random inputs to various, fixed output distributions.

.. moduleauthor:: Johannes Leugering <jleugeri@uos.de>
"""

_docstring_template = \
""" %s that maps to %s

.. note::
    
    properties of adaptive LN neurons which can be probed include:
        
        :rates: the rate output of the neuron
        :mu_in: the estimated mean (1st sufficient statistic) of the normal input
        :sq_in: the estimated mean of squares (2nd sufficient statistic) of the normal input

:param tau_adapt:   time-constant of the adaptation variables (in ms)
:type tau_adapt:    scalar
:param scale: a scale parameter (typically multiplicative, depending on the output distribution)
:type scale:  ndarry(dtype=float)
:param position: a position parameter (typically additive, depending on the output distribution)
:type scale:  ndarry(dtype=float)
"""

import numpy as np
import scipy as sp
import scipy.special

from nengo.neurons import NeuronType
from nengo.params import NumberParam

class AdaptiveLN(NeuronType):
    __doc__ = _docstring_template % (   "Base class for adaptive linear-non-linear (LN) neuron models", 
                                        "various output distributions defined by subclasses" )
    tau_adapt = NumberParam(low=0)
    probeable = ['rates', 'mu_in', 'sq_in']

    def __init__(self, tau_adapt=1000.0, position=np.zeros((1,)), scale=np.zeros((1,))):
        """Set up the adaptive linear-non-linear neuron model.
        
        :param tau_adapt:   time-constant of the adaptation variables
        :type tau_adapt:    scalar
        :param scale: a scale parameter (typically multiplicative, depending on the output distribution)
        :type scale:  ndarry(dtype=float)
        :param position: a position parameter (typically additive, depending on the output distribution)
        :type scale:  ndarry(dtype=float)
        """
        self.tau_adapt  = tau_adapt
        self.position   = position
        self.scale      = scale

    def gain_bias(self, *args, **kwargs):
        """ Not implemented.

        .. warning::
            
            This function is only called if the network is set up incorrectly.
            The gain and bias variables should be set to ones and zeros, respectively,
            when creating an ensemble.

            Use the `bias` and `gain` keyword arguments of :func:`nengo.Ensemble` to set them manually!
        """
        raise NotImplementedError("Setting gain and bias has no effect for this model."+
                                    "Set them manually to ones and zeros, respectively!")

    def rates(self, x, scale, position):
        """ Calculate firing rate of differently tuned neurons in response to inputs.
        
        :param x: input(s) to the neurons
        :type x: ndarry(dtype=float)
        :param scale: a scale parameter (typically multiplicative, depending on the output distribution); **currenty ignored**
        :type scale:  ndarry(dtype=float)
        :param position: a position parameter (typically additive, depending on the output distribution); **currenty ignored**

        :type scale:  ndarry(dtype=float)

        .. warning ::

            The `scale` and `position` parameters given as arguments corresponding to `gain` and `bias` in :mod:`nengo`
            don't yet behave as desired and should thus be set to ones and zeros, respectively.

        """
        out     = np.zeros_like(x)
        mu_in   = np.zeros_like(x)
        sq_in   = np.ones_like(x)
        #self.position   = np.zeros_like(position)
        #self.scale      = np.ones_like(scale)
        self.step_math(dt=1, J=np.ones_like(scale)*x, output=out, mu_in=mu_in, sq_in=sq_in)
        return out

    def step_math(self, dt, J, output, mu_in, sq_in):
        """Compute firing rates (in Hz).
        
        :param dt:      step-size of numerical simulation (in ms)
        :type dt:       scalar
        :param J:       inputs
        :type J:        ndarray(dtype=float)
        :param output:  array in which to story the resulting outputs
        :type output:   ndarry(dtype=float)
        :param mu_in:   the estimated mean (1st sufficient statistic) of the normal input
        :type mu_in:    ndarray(dtype=float)
        :param sq_in:   the estimated mean of squares (2nd sufficient statistic) of the normal input
        :type mu_in:    ndarray(dtype=float)
        """
        # Work-around for the bug that sets all internal variables to 0 in the first step:
        sq_in[sq_in<1e-10] = 1 # sq_in should never be exactly zero 

        # Update the moving average of the sufficient statistics of the input distribution:
        alpha = dt/self.tau_adapt
        mu_in[...] += alpha*(J      - mu_in)
        sq_in[...] += alpha*(J**2   - sq_in)

        # Whiten the input using the running ML estimates of the parameters of the input distribution
        whitened = (J - mu_in) / np.sqrt(sq_in - mu_in**2)

        # Calculate the output by applying the non-linearity to the whitened normal input
        output[...] = self.nonlinearity(whitened)

    def nonlinearity(self, whitened_J):
        """ Not implemented. Must be defined by subclasses."""
        raise NotImplementedError("This method must be defined by the inherited sub-classes.")

class AdaptiveLNuniform(AdaptiveLN):
    __doc__ = _docstring_template % (   "Adaptive LN neuron model", "uniformly distributed outputs" )

    def nonlinearity(self, whitened):
        """Calculates the nonlinearity for a mapping from standard normal inputs to uniform outputs
        
        :param whitened:    whitened input to the neurons' nonlinearity
        :rtype:             ndarray(dtype=float)
        """
        return self.position + self.scale*0.5*(sp.special.erf(whitened/np.sqrt(2))+1)








class AdaptiveLNlogNormal(AdaptiveLN):
    __doc__ = (_docstring_template % (   "Adaptive LN neuron model", "log-normally distributed outputs" )) + \
        "\n:param cutoff:  maximum attainable firing rate at which the exponential output is cut off " + \
        "\n:type cutoff:   scalar \n\n"

    def __init__(self, cutoff=100.0, **kwargs):
        """ Initializes the neuron model.

        :param cutoff:  maximum attainable firing rate at which the exponential output is cut off
        :type cutoff:   scalar
        :param kwargs:  additional keyword arguments passed to :func:`AdaptiveLN.__init__`
        """
        super(AdaptiveLNlogNormal, self).__init__(**kwargs)
        self.log_cutoff = np.log(cutoff)

    def nonlinearity(self, whitened):
        """Calculates the nonlinearity for a mapping from standard normal inputs to log-normal outputs.
        
        :param whitened:    whitened input to the neurons' nonlinearity
        :rtype:             ndarray(dtype=float)
        """
        val = np.minimum(self.scale*whitened + self.position, self.log_cutoff)
        return np.exp(val)


