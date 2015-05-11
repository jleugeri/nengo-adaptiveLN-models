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

=============================================================================

The nengo adaptive LN models package contains the module :mod:`neurons` that defines several
adaptive linear-non-linear neuron models mapping normally distributed inputs to various output
distributions.

.. moduleauthor:: Johannes Leugering
"""
from neurons import AdaptiveLN, AdaptiveLNuniform, AdaptiveLNlogNormal

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.neurons import SimNeurons

@Builder.register(AdaptiveLN)
def build_adln(model, adLN, neurons):
    model.sig[neurons]['mu_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.mu_in" % neurons)
    model.sig[neurons]['sq_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.sq_in" % neurons)
    model.add_op(SimNeurons(neurons=adLN,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['mu_in'],
                                    model.sig[neurons]['sq_in']]))


@Builder.register(AdaptiveLNuniform)
def build_adlnuniform(model, adLNuniform, neurons):
    model.sig[neurons]['mu_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.mu_in" % neurons)
    model.sig[neurons]['sq_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.sq_in" % neurons)
    model.add_op(SimNeurons(neurons=adLNuniform,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['mu_in'],
                                    model.sig[neurons]['sq_in']]))


@Builder.register(AdaptiveLNlogNormal)
def build_adlnlognormal(model, adLNlognormal, neurons):
    model.sig[neurons]['mu_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.mu_in" % neurons)
    model.sig[neurons]['sq_in'] = Signal(
        np.zeros(neurons.size_in), name="%s.sq_in" % neurons)
    model.add_op(SimNeurons(neurons=adLNlognormal,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['mu_in'],
                                    model.sig[neurons]['sq_in']]))

