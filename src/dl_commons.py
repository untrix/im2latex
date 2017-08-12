# -*- coding: utf-8 -*-
"""
    Copyright 2017 Sumeet S Singh

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

Tested on python 2.7

@author: Sumeet S Singh
"""
import collections

class AccessDeniedError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

class Properties(dict):
    """
    A shorthand way to create a class with
    several properties without having to hand-code the getter and setter functions.
    getter/setter access is automatically provided saving you from having to write
    two decorated functions (getter and setter) per property.
    The properties can either be provided at init time or added on the fly simply
    by setting/assigning them.
    You can also view this as a javascript Object style dictionary because it allows
    both attribute getter/setter as well as dictionary accessor (square brackets) syntax.
    Additionally, you my call freeze() or seal() to freeze or seal
    the dictionary - just as in Javascript.
    The class inherits from dict therefore all standard python dictionary interfaces
    are available as well (such as iteritems etc.)

    x = Properties({'x':1, 2:'y'})
    assert d.x == d['x']
    d.x = 5
    assert d['x'] == 5
    d.seal()
    d[3] = 'z' # okay, new property created
    assert d.3 == 'z'
    d[2] = 'a' # raises AccessDeniedError
    d.freeze()
    d[4] = 'a' # raises AccessDeniedError
    d[2] = 'a' # raises AccessDeniedError
    """

    def __init__(self, d={}):
        dict.__init__(self, d)
        object.__setattr__(self, '_isFrozen', False)
        object.__setattr__(self, '_isSealed', False)

    def _get_val_(self, key):
        return dict.__getitem__(self, key)

    def _rvn(self, key):
        """
        For internal use only - needed by Params.__init__.
        returns the dictionary value or None
        """
        try:
            return self._get_val_(key)
        except KeyError:
            return None

    def _set_val_(self, key, val):
        if self.isFrozen():
            raise AccessDeniedError('Object is frozen, therefore key "%s" cannot be modified'%(key,))
        elif self.isSealed() and key not in dict.keys(self):
            raise AccessDeniedError('Object is sealed, new key "%s" cannot be added'%(key,))
        else:
            dict.__setitem__(self, key, val)

    def __copy__(self):
        ## Shallow copy
        return self.__class__(self)

    def copy(self):
        ## Shallow copy
        return self.__class__(self)

    def updated(self, other):
        """ chain-update
        Same as dict.update except that it returns self and therefore
        supports call-chaining. E.g. Properties(other).cupdate(other2).cupdate(other2)
        """
        dict.update(self, other)
        return self

    def __getattr__(self, key):
        return self._get_val_(key)

    def __setattr__(self, key, val):
        return self._set_val_(key, val)

    def __getitem__(self, key):
        return self._get_val_(key)

    def __setitem__(self, key, val):
        return self._set_val_(key, val)

    def isFrozen(self):
        return object.__getattribute__(self, '_isFrozen')

    def isSealed(self):
        return object.__getattribute__(self, '_isSealed')

    def freeze(self):
        object.__setattr__(self, '_isFrozen', True)
        return self

    def seal(self):
        object.__setattr__(self, '_isSealed', True)
        return self

class NoneProperties(Properties):
    """
    A variation of Properties which will silently return a None value for missing keys instead of throwing
    an exception.
    """
    def _get_val_(self, key):
        try:
            return Properties._get_val_(self, key)
        except KeyError:
            return None

class ParamDesc(Properties):
    """
    A property descriptor.
    """
    def __init__(self, name, text, validator=None, default=None):
        """
        @name = name of property,
        @text = textual description,
        @validator = (optional) a validator object that implements the __contains__()
            method so that membership may be inspected using the 'in' operator -
            as in 'if val in validator:'
        @default = value (optional) stands for the default value. Set to None if
            unspecified.

        The object gets immediately frozen after
        initialization so that the property descriptor can be re-used repeatedly
        without fear of modification.
        """

        Properties.__init__(self, {'name':name, 'text':text, 'validator':validator, 'default':default})
        self.freeze()

## A shorter alias of ParamDesc
PD = ParamDesc

class PDTuple(tuple):
    def __new__ (cls, pd_tuple):
        return super(PDTuple, cls).__new__(cls, pd_tuple)

    def __getitem__(self, name):
        return [pd for pd in self if pd.name == name][0]

## Shorter alias of ParamList
PDL = PDTuple

class ParamsValueError(ValueError):
    pass

class Params(Properties):
    """
    Prototyped Properties class. The prototype is passed into the constructor and
    should be a list of ParamDesc objects denoting all the allowed properties.
    No new properties can be added in the future - i.e. the object is sealed.
    You may also freeze the object at any point if you want.
    This class is used to define and store descriptors of a model. It inherits
    from class Properties.
    """

    def __init__(self, prototype, initVals=None):
        """
        Takes property descriptors and their values. After initialization, no new params
        may be created - i.e. the object is sealed (see class Properties). The
        property values can be modified however (unless you call freeze()).

        @param prototype (sequence of ParamDesc): Sequence of ParamDesc objects which serve as the list of
            valid properties. Can be a sequence of ParamDesc objects or another Params object.
            If it was a Params object, then descriptors would be derived
            from prototype.protoS.
            This object will also provide the property values if not specified in
            the vals argument (below). If this was a list of ParamDesc objects,
            then the default property values would be used (ParamDesc.default). If
            on the other hand, this was a Params object, then its property
            value would be used (i.e. the return value of Params['prop_name']).

        @param initVals (dict): provides initial values of the properties of this object.
            May specify a subset of the object's properties, or none at all. Unspecified
            property values will be initialized from the prototype object.
            Should be either a dictionary of name:value pairs or unspecified (None).

        If a value (even default value) is a callable (i.e. function-like) but is
        expected to be a non-callable (i.e. validator does not derive from _iscallable)
        then it will
        be called in order to get the actual value of the parameter. The callable function will
        be passed in name of the property and a dictionary of all the properties
        with their initVals or default vals as appropriate. Callables are called
        in a second pass - after the initVals and default-values of all the properties
        have been resolved (this happens in the first pass). However, at this point the callable
        objects are not invoked and therefore (internally) show up as values of the property.
        In the second pass, the callable objects are
        are invoked in the order in which the properties were declared in the prototype.
        The final value of a property will be set to the return value of the callable.
        Hence the callable function should expect that properties that occur
        later in the prototype sequence will still have their values set to
        callable objects (if any). Both initVal and default values maybe
        callables. Just don't get into loops. also, the dictionary passed into
        the callable is a copy of the resolved values, and therefore changing
        some other property's value will have no impact. This feature is used
        to set one property's value based on others. See the example below.

        Examples:
        o1 = Params(
                    [
                     ParamDesc('model_name', 'Name of Model', None, 'im2latex'),
                     ParamDesc('layer_type', 'Type of layers to be created'),
                     ParamDesc('num_layers', 'Number of layers. Defaults to 1', xrange(1,101), 1),
                     ParamDesc('num_units', 'Number of units per layer. Defaults to 10000 / num_layers',
                               xrange(1, 10000),
                               lambda name, props: 10000 // props["num_layers"]), ## Lambda function will be invoked
                     ParamDesc('activation_fn', 'tensorflow activation function',
                               iscallable(tf.nn.relu, tf.nn.tanh, None),
                               tf.nn.relu) ## Function will not be invoked because the param-type is callable.
                    ])
        o2 = Params(o1) # copies prototype from o1, uses default values
        o3 = Params(o1, initVals={'model_name':'im2latex'}) # uses descriptors from o1.protoS,
            initializes with val from vals if available otherwise with default from o1.protoS
        """
        Properties.__init__(self)
        descriptors = prototype
        props = Properties()
        vals1_ = Properties()
        vals2_ = Properties()
        _vals = {}

        if isinstance(prototype, Params):
            descriptors = prototype.protoS
            if initVals is None:
                vals1_ = prototype
            else:
                vals1_ = initVals if isinstance(initVals, Properties) else Properties(initVals)
                vals2_ = prototype
        else:
            if initVals is not None:
                vals1_ = initVals if isinstance(initVals, Properties) else Properties(initVals)

        object.__setattr__(self, '_desc_list', tuple(descriptors) )

        derivatives = []
        for prop in descriptors:
            name = prop.name
            if name not in props:
                try:
                    props[name] = prop
#                    _vals[name] = vals1_[name] if (name in vals1_) else vals2_[name] if (name in vals2_) else prop.default
                    val1 = vals1_._rvn(name)
                    val2 = vals2_._rvn(name)
                    _vals[name] = val1 if (val1 is not None) else val2 if (val2 is not None) else prop.default
                    if isinstance(_vals[name], LambdaVal):
                        ## This is a case where a proxy function has been provided
                        ## in place of a value.
                        derivatives.append(name)
                except:
                    print('##### Error while processing property: \n', prop, '\n')
                    raise
            else:
                raise ParamsValueError('property %s has already been initialized with value %s'%(name,
                                                                                                 _vals[name]))

        object.__setattr__(self, '_descr_dict', props.freeze())

        # Derivatives: Run the derived param callables
        # First copy _vals so that we may safely pass the dictionary of values to
        # the callables
#        _vals_copy = copy.copy(_vals)
#        for name in derivatives:
#            _vals[name] = _vals_copy[name] = _vals[name](name, _vals_copy)

        # Validation: Now insert the property values one by one. Doing so will invoke
        # self._set_val_ which will validate their values against self._descr_dict.
        for prop in descriptors:
            try:
                _name = prop.name
                self[_name] = _vals[_name]
            except ParamsValueError:
                raise
            except:
                print('##### Error while processing property: \n', prop, '\n')
                raise

        # Finally, seal the object so that no new properties may be added.
        self.seal()

    def _set_val_(self, name, val):
        """ Polymorphic override of _set_val_. Be careful of recursion. """
        protoD = self.protoD
        if not self.isValidName(name):
            raise KeyError('%s is not an allowed property name'%(name,))
        # As a special case, we allow setting None values even if that's not an
        # allowed value. This is so because the code may want to set the
        # property value in the future based on some dynamic decision.
        elif (val is not None) and (not isinstance(val, LambdaVal)) and (protoD[name].validator is not None) and (val not in protoD[name].validator):
            raise ParamsValueError('%s is not a valid value of property %s'%(val, name))
        else:
            return Properties._set_val_(self, name, val)


    def _resolve_raw_val(self, name, val):
        """Check to see if a value is dynamic and resolve it if so. """
        if isinstance(val, LambdaVal):
            val = val(name, self)
        return val

    def _resolve_raw_vals(self, name, vals_):
        """ Resolve a single (possibly lambda) val or sequence of (possibly lambda) vals. """
#        if isTupleOrList(vals_):
        if issequence(vals_):
            vals_l = [self._resolve_raw_val(name, val) for val in vals_]
            vals = vals_.__class__(vals_l)
        else:
            vals = self._resolve_raw_val(name, vals_)

        protoD = self.protoD
        if (vals is not None) and (protoD[name].validator is not None) and (vals not in protoD[name].validator):
            raise ParamsValueError('%s is not a valid value of property %s'%(vals, name))

        return vals

    def _rvn(self, name):
        """
        Return raw-value if key exists else return None.
        """
        try:
            return Properties._get_val_(self, name)
        except KeyError:
            return None

    def _get_val_(self, name):
        """ Polymorphic override of _get_val_. Be careful of recursion. """
        if not self.isValidName(name):
            raise KeyError('%s is not an allowed property name'%(name,))
        else:
            val = Properties._get_val_(self, name)
            return self._resolve_raw_vals(name, val)

    def isValidName(self, name):
        return name in self.protoD

    def append(self, other):
        assert isinstance(other, Params)
        for param in other.protoS:
            assert param.name not in self.protoD

        ## Update Descriptor first
        protoD = self.protoD.copy()
        for param in other.protoS:
            protoD[param.name] = param
            self.protoS.append(param)
        object.__setattr__(self, '_descr_dict', protoD.freeze())

        ## Update values
        for param in other.protoS:
            self[param.name] = other[param.name]

        return self

    @property
    def protoS(self):
        # self._desc_list won't recursively call _get_val_ because __getattribute__ will return successfully
        #return self._desc_list
        return object.__getattribute__(self, '_desc_list')

    @property
    def protoD(self):
        # self._descr_dict won't call __getattr__ because __getattribute__ returns successfully
        #return self._descr_dict
        return object.__getattribute__(self, '_descr_dict')


class HyperParams(Params):
    """
    Params class specialized for HyperParams. Adds the following semantic:
        If a key has value None, then it is deemed absent from the dictionary. Calls
        to __contains__ and _get_val_ will beget a KeyError - as if the property was
        absent from the dictionary. This is necessary to catch cases wherein one
        has forgotten to set a mandatory property. Mandatory properties must not have
        default values in their descriptor. A property that one is okay forgetting
        to specify should have a None default value set in its descriptor which may
        include 'None'.
        However as with Params, it is still possible to initialize or set a property value to
        None eventhough None may not appear in the valid-list. We allow this in
        order to enable lazy initialization - i.e. a case where the
        code may wish to initialize a property to None, and set it to a valid value
        later. Setting a property value to None tantamounts to unsetting / deleting
        the property.
    """

    def __contains__(self, name):
        """ Handles None values in a special way as stated above. """
        try:
            self._get_val_(name)
            return True
        except KeyError:
            return False

    def _get_val_(self, name):
        """ Polymorphic override of _get_val_. Be careful of recursion. """
        val = Params._get_val_(self, name)
        validator = self.protoD[name].validator
        if (val == None) and ((validator is None) or (None not in validator)):
            raise KeyError('property %s was not set'%(name,))
        else:
            return val

## Abstract parameter validator class.
class _ParamValidator(object):
    def __contains__(self, val):
        raise NotImplementedError('This is an abstract class.')

## Dynamic Value setter. Wrapper class
class LambdaVal(object):
    def __init__(self, func):
        self._func = func
    def __call__(self, name, props):
        return self._func(name, props)

class LinkedParam(_ParamValidator):
    """ A validator class indicating that the param's value is derived by calling a function.
    Its value cannot be directly set, instead a function should be provided that will
    set its value.
    """
    pass

class equalto(LambdaVal):
    """
    A callable object that can be used to set the value of
    one property equal to the one passed in.
    Example:
        dlc.Params((
            PD('A',
               'Some property A',
               None,
               (120,1075,3)
               ),
            PD('B',
               'Another property that should be equal in value to A',
               instanceof(int),
               equalto('A')
               )
           ))
    """
    def __init__(self, target, d=None):
        LambdaVal.__init__(self, lambda _, props: props[target] if d is None else d[target])


# A validator that returns False for non-None values
class _mandatoryValidator(_ParamValidator):
    def __contains__(self, val):
        return val is not None

class instanceof(_ParamValidator):
    def __init__(self, cls, noneokay=False):
        self._cls = cls
        self._noneokay = noneokay
    def __contains__(self, obj):
        return (self._noneokay and obj is None) or isinstance(obj, self._cls) or isinstance(obj, LambdaVal)

class instanceofOrNone(instanceof):
    def __init__(self, cls):
        super(instanceofOrNone, self).__init__(cls, True)

# A inclusive range validator class
class range_incl(_ParamValidator):
    def __init__(self, begin=None, end=None):
        self._begin = begin
        self._end = end
    def __contains__(self, v):
        return (self._end is None or v <= self._end) and (self._begin is None or v >= self._begin)

class _type_range_incl(instanceof):
    def __init__(self, cls, begin=0, end=None, noneokay=False):
        instanceof.__init__(self, cls, noneokay)
        self._range = range_incl(begin, end)
    def __contains__(self, v):
        return instanceof.__contains__(self, v) and (self._range.__contains__(v) or (self._noneokay and v is None))

class integer(_type_range_incl):
    def __init__(self, begin=None, end=None, noneokay=False):
        _type_range_incl.__init__(self, int, begin, end, noneokay)

class integerOrNone(integer):
    def __init__(self, begin=None, end=None):
        integer.__init__(self, begin, end, noneokay=True)

class decimal(_type_range_incl):
    def __init__(self, begin=None, end=None, noneokay=False):
        _type_range_incl.__init__(self, float, begin, end, noneokay)

class decimalOrNone(decimal):
    def __init__(self, begin=None, end=None,):
        decimal.__init__(self, begin, end, noneokay=True)

def issequence(v):
    if isinstance(v, basestring):
        return False
    return isinstance(v, collections.Sequence)

def isTupleOrList(v):
    return isinstance(v, tuple) or isinstance(v, list)

class iscallable(_ParamValidator):
    def __init__(self, lst=None, noneokay=False):
        assert lst is None or issequence(lst)
        self._noneokay = noneokay
        self._lst = lst

    def __contains__(self, v):
        if v is None: ## special handling for None values
            if self._noneokay:
                return True
            elif self._lst is not None:
                return None in self._lst
            else:
                return False
        elif self._lst is None:
            return callable(v)
        else:
            return callable(v) and v in self._lst

class iscallableOrNone(iscallable):
    def __init__(self, lst=None):
        iscallable.__init__(self, lst, noneokay=True)

class issequenceof(instanceof):
    def __init__(self, cls, noneokay=False):
        instanceof.__init__(self, cls, noneokay)
        self._noneokay = noneokay
    def __contains__(self, v):
        return (self._noneokay and v is None) or (issequence(v) and instanceof.__contains__(self, v[0]))

class issequenceofOrNone(issequenceof):
    def __init__(self, cls):
        issequenceof.__init__(self, cls, noneokay=True)

class _anyok(_ParamValidator):
    def __contains__(self, v):
        return True

# Helpful validator objects
mandatory = _mandatoryValidator()
boolean = instanceof(bool, False)

import numpy as np

def squashed_seq_list(np_seq_batch, seq_lens, remove_val, EOSToken=0):
    assert np_seq_batch.ndim == 2
    assert EOSToken == 0
    assert remove_val != EOSToken
    sq_batch = []

    for i, np_seq in enumerate(np_seq_batch):
        seq_len = seq_lens[i]
        trunc = np_seq[:seq_len]
        if trunc[-1] != 0:
            trunc = np.append(trunc, [0])
            print 'WARNING: No EOS tokens in sequence of length %d'%seq_len
        elif trunc[-2] == 0:
            trunc = np.append(np.trim_zeros(trunc, 'b'), [0])
            print 'WARNING: More than one EOS tokens in sequence of length %d'%seq_len

        squashed = trunc[trunc != remove_val]
        sq_batch.append(squashed.tolist())
        # sq_batch.append(np.pad(squashed, (0, T-squashed.shape[0]), mode='constant', constant_values=0))
    return sq_batch

def get_bleu_weights(max_len=200, frac=0.8):
    weights = [None]
    for i in range(1, max_len+1):
        o = int(np.ceil([i*(1.-frac)])[0])
        w = np.ones(o, dtype=float) / o
        z = np.zeros(i-o, dtype=float)
        weights.append( np.concatenate((z, w)) )
    return weights
BLEU_WEIGHTS = get_bleu_weights()

import nltk
def squashed_bleu_scores(predicted_ids, predicted_lens, target_ids, target_lens, space_token):
    """
    Removes space-tokens from predicted_ids and then computes the bleu scores of a batch of
    sequences.
    Args:
        predicted_ids: Numpy array of predicted sequence tokens - (B, T)
        predicted_lens: Numpy array of predicted sequence lengths - (B,)
        target_ids: Numpy array of reference sequence tokens - (B, T')
        target_lens: Numpy array of reference sequence lengths - (B,)
        space_token: (token-type) Space/Blank token to remove from predicted_ids
    """
    scores = []
    squashed_ids = squashed_seq_list(predicted_ids, predicted_lens, space_token)
    for i, predicted_seq in enumerate(squashed_ids):
        target_len = target_lens[i]
        scores.append(nltk.translate.bleu_score.sentence_bleu([target_ids[i][:target_len]], 
                                                              predicted_seq,
                                                              weights=BLEU_WEIGHTS[target_len]))
        print 'BLEU Score = %f'%scores[-1]
        print 'Target = %s'%target_ids[i]
        print 'Predicted = %s'%predicted_seq

    return scores
# nltk.translate.bleu_score.sentence_bleu([range(100)],range(100), weights=[1/100.]*100)