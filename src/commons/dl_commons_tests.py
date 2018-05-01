#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Copyright 2017 - 2018 Sumeet S Singh

    This file is part of im2latex solution by Sumeet S Singh.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the Affero GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Affero GNU General Public License for more details.

    You should have received a copy of the Affero GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

@author: Sumeet S Singh

Works on python 2.7
"""

import unittest
import dl_commons as dlc
from dl_commons import PD, LambdaVal, integer, integerOrNone, instanceof, equalto
#import tf_commons as tfc

class Props(dlc.Params):
    proto = (
            PD('m', '',
               integer(),
               64),
            PD('D', '',
               integer(),
               512)
            )
    def __init__(self, initVals={}):
        dlc.Params.__init__(self, self.proto, initVals)

class Props2(dlc.Params):
    def makeProto(self, GLOBAL):
        return Props.proto + (
            PD('i', '',
               integer(),
               LambdaVal(lambda _, __: GLOBAL.m + GLOBAL.D)
               ),
            PD('m2', '',
               integer(),
               equalto('m', GLOBAL)),
            PD('D2', '',
               integer(),
               equalto('D', GLOBAL)),
            PD('j', '',
               integerOrNone(),
               None
               ),
            PD('k', '',
               integerOrNone(),
               1
               ),
            )
    def __init__(self, initVals={}):
        dlc.Params.__init__(self, self.makeProto(initVals), initVals)

class Props3(dlc.Params):
    def makeProto(self, GLOBAL):
        return Props.proto + (
            PD('i', '',
               integer(),
               equalto('i', GLOBAL)
               ),
            PD('m3', '',
               integer(),
               equalto('m2', GLOBAL)),
            PD('D3', '',
               integer(),
               equalto('D2', GLOBAL)),
            PD('j', '',
               integerOrNone(),
               2
               ),
            PD('k', '',
               integerOrNone(),
               2
               ),
            PD('l', '',
               integerOrNone(),
               2
               ),
            )
    def __init__(self, initVals={}):
        dlc.Params.__init__(self, self.makeProto(initVals), initVals)



class TestCaseBase(unittest.TestCase):
    @staticmethod
    def dictSet(d, name, val):
        d[name] = val

    @staticmethod
    def dictGet(d, name):
        return d[name]

    @staticmethod
    def instantiate(cls, *args):
        cls(*args)

class PropertiesTest(TestCaseBase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)

    def test_good_props(self):
        props = {
                'model_name':'im2latex',
                'num_layers':None,
                'unset':None
                }
        open = dlc.Properties(props)
        sealed = dlc.Properties(open).seal()
        props['num_layers'] = 10
        frozen = dlc.Properties(props).freeze()

        open.layer_type = 'MLP' # create new property
        self.assertEqual(open.layer_type, 'MLP')
        self.assertEqual(open['layer_type'], 'MLP')
        open['layer_type'] = 'CNN'
        self.assertEqual(open.layer_type, 'CNN')
        self.assertEqual(open['layer_type'], 'CNN')

        self.assertEqual(frozen.model_name, 'im2latex')
        self.assertEqual(frozen.unset, None)
        self.assertEqual(frozen['unset'], None)
        self.assertEqual(frozen['num_layers'], 10)
        self.assertEqual(frozen.num_layers, 10)


    def test_bad_props(self):
        props = {
                'model_name':'im2latex',
                'num_layers':None,
                'unset':None
                }
        open = dlc.Properties(props)
        sealed = dlc.Properties(open).seal()
        props['num_layers'] = 10
        frozen = dlc.Properties(props).freeze()

        self.assertRaises(dlc.AccessDeniedError, setattr, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(dlc.AccessDeniedError, self.dictSet, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(dlc.AccessDeniedError, setattr, frozen, "name", "MyNeuralNetwork")
        self.assertRaises(dlc.AccessDeniedError, self.dictSet, frozen, "name", "MyNeuralNetwork")

        self.assertRaises(KeyError, getattr, sealed, "x")
        self.assertRaises(KeyError, self.dictGet, sealed, "x")

    def test_good_params(self):
        sealed = dlc.Params((
                dlc.ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN'], 'LSTM'),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11))
                )
            ).seal()
        frozen = dlc.Params(sealed, {'num_layers':10}).freeze()
        sealed.layer_type = 'MLP'
        self.assertEqual(sealed.layer_type, 'MLP')
        self.assertEqual(sealed['layer_type'], 'MLP')
        sealed['layer_type'] = 'CNN'
        self.assertEqual(sealed.layer_type, 'CNN')
        self.assertEqual(sealed['layer_type'], 'CNN')

        self.assertEqual(frozen.model_name, 'im2latex')
        self.assertEqual(frozen.layer_type, 'LSTM')
        self.assertEqual(frozen['num_layers'], 10)
        self.assertEqual(frozen.num_layers, 10)


    def test_bad_params(self):
        proto = (
                dlc.ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN']),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11))
                )
        sealed = dlc.Params(proto).seal()
        frozen = dlc.Params(proto, {'num_layers':10}).freeze()
        self.assertRaises(KeyError, setattr, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(KeyError, self.dictSet, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(KeyError, setattr, frozen, "name", "MyNeuralNetwork")
        self.assertRaises(KeyError, self.dictSet, frozen, "name", "MyNeuralNetwork")

        self.assertRaises(ValueError, setattr, sealed, "layer_type", "SVM")
        self.assertRaises(ValueError, self.dictSet, sealed, "layer_type", "SVM")

        self.assertRaises(KeyError, getattr, sealed, "x")
        self.assertRaises(KeyError, self.dictGet, sealed, "x")

    def test_good_hyperparams(self):
        sealed = dlc.HyperParams((
                dlc.ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN'], 'MLP'),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11)),
                dlc.ParamDesc('none', 'None property', (None,), None)
                )
            ).seal()
        frozen = dlc.HyperParams(sealed, {'num_layers':10}).freeze()
        self.assertRaises(dlc.OneValError, setattr, sealed, "model_name", "xyz")
        self.assertRaises(dlc.OneValError, setattr, sealed, "layer_type", "xyz")
        self.assertEqual(sealed.layer_type, 'MLP')
        self.assertEqual(sealed['layer_type'], 'MLP')

        self.assertEqual(frozen.model_name, 'im2latex')
        self.assertEqual(frozen['num_layers'], 10)
        self.assertEqual(frozen.num_layers, 10)
        self.assertEqual(frozen.none, None)
        self.assertEqual(frozen['none'], None)
        self.assertEqual(sealed.none, None)
        self.assertEqual(sealed['none'], None)

    def test_bad_hyperparams(self):
        sealed = dlc.HyperParams((
            dlc.ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
            dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN']),
            dlc.ParamDesc('num_layers', 'Number of layers to create', range(1, 11)),
            dlc.ParamDesc('unset', 'Unset property', range(1, 11)),
            dlc.ParamDesc('none', 'None property', (None,), None)
        )).seal()
        frozen = dlc.HyperParams(sealed, {'num_layers': 10}).freeze()
        self.assertRaises(KeyError, setattr, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(KeyError, self.dictSet, sealed, "x", "MyNeuralNetwork")
        self.assertRaises(KeyError, setattr, frozen, "name", "MyNeuralNetwork")
        self.assertRaises(KeyError, self.dictSet, frozen, "name", "MyNeuralNetwork")

        self.assertRaises(ValueError, setattr, sealed, "layer_type", "SVM")
        self.assertRaises(ValueError, self.dictSet, sealed, "layer_type", "SVM")

        self.assertRaises(KeyError, getattr, sealed, "x")
        self.assertRaises(KeyError, self.dictGet, sealed, "x")
        self.assertRaises(KeyError, getattr, frozen, 'layer_type')
        self.assertRaises(KeyError, getattr, sealed, 'layer_type')


    def test_lambda_vals(self):
        p = Props()
        p2 = Props2(p)
        p3 = Props3(p2)
        self.assertEqual(p.m, 64)
        self.assertEqual(p.D, 512)
        self.assertEqual(p2.m, 64)
        self.assertEqual(p2.D, 512)
        self.assertEqual(p2.i, 512+64)
        self.assertEqual(p2.m2, 64)
        self.assertEqual(p2.D2, 512)
        self.assertEqual(p3.m, 64)
        self.assertEqual(p3.D, 512)
        self.assertEqual(p3.i, 512+64)
        self.assertEqual(p3.m3, 64)
        self.assertEqual(p3.D3, 512)

        p.m = 128
        self.assertEqual(p.m, 128)
        self.assertEqual(p.D, 512)
        self.assertEqual(p2.m, 64)
        self.assertEqual(p2.D, 512)
        self.assertEqual(p2.i, 512+128)
        self.assertEqual(p2.m2, 128)
        self.assertEqual(p2.D2, 512)
        self.assertEqual(p3.m, 64)
        self.assertEqual(p3.D, 512)
        self.assertEqual(p3.i, 512+128)
        self.assertEqual(p3.m3, 128)
        self.assertEqual(p3.D3, 512)
        self.assertEqual(p3.j, None)
        self.assertEqual(p3.k, 1)
        self.assertEqual(p3.l, 2)


unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(PropertiesTest))
