#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Sumeet S Singh
"""

import unittest
import dl_commons as dlc
#import tf_commons as tfc

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
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN']),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11), None)
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
        self.assertEqual(frozen.layer_type, None)
        self.assertEqual(frozen['num_layers'], 10)
        self.assertEqual(frozen.num_layers, 10)
        

    def test_bad_params(self):
        sealed = dlc.Params((
                dlc.ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN']),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11), None)
                )
            ).seal()
        frozen = dlc.Params(sealed, {'num_layers':10}).freeze()
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
                dlc.ParamDesc('layer_type', 'Type of layers to be created', ['CNN', 'MLP', 'LSTM', 'RNN']),
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11), None),
                dlc.ParamDesc('none', 'None property', (None,), None)
                )
            ).seal()
        frozen = dlc.HyperParams(sealed, {'num_layers':10}).freeze()
        sealed.layer_type = 'MLP'
        self.assertEqual(sealed.layer_type, 'MLP')
        self.assertEqual(sealed['layer_type'], 'MLP')
        sealed['layer_type'] = 'CNN'
        self.assertEqual(sealed.layer_type, 'CNN')
        self.assertEqual(sealed['layer_type'], 'CNN')
        
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
                dlc.ParamDesc('num_layers', 'Number of layers to create', range(1,11)),
                dlc.ParamDesc('unset', 'Unset property', range(1,11), None),
                dlc.ParamDesc('none', 'None property', (None,), None)
                )
            ).seal()
        frozen = dlc.HyperParams(sealed, {'num_layers':10}).freeze()
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
        
unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(PropertiesTest))
