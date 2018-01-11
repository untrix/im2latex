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
import pprint
import numpy as np
import nltk
import data_commons as dtc

class AccessDeniedError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def Properties_Factory():
    return Properties({})

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

    def _get_unvalidated_val(self, name):
        """
        Return resolved but unvalidated value.
        Reroutes to _get_val_. Is useful only for subclasses that validate values.
        """
        return self._get_val_(name)

    def _rv(self, key):
        """
        For internal use only - needed by Params.__init__.
        returns the dictionary value
        """
        return self._get_val_(key)

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

    def __getstate__(self):
        """
        Returns pickle_state by invoking to_picklable_dict() - i.e. all values resolved but not validated.
        Will not throw an exception if invalid/unset values are detected in order to be useful for debugging. The output file can be passed to 'from_pickle'.
        NOTE however, that all functions (isCallable types) are reduced to their printable string representation before storing and can't be recovered
        later by a call to 'from_pickle'. Lambda functions embedded within LambdaVal objects are also invoked and resolved and therefore are not pickled.
        Therefore this method is mostly only useful for printing and storing values for debugging if since it can't recover the
        LambdaVals and function values.
        """
        # return self.to_picklable_dict()
        raise NotImplementedError('Please invoke Properties.dump instead')

    def dump(self, *paths):
        dtc.dump(self.to_picklable_dict(), *paths)

    def __setstate__(self, d):
        return self.updated(d)

    @classmethod
    def load(cls, *paths):
        d = dtc.load(*paths)
        return cls(d)

    def __reduce__(self):
        return (Properties_Factory, tuple(), self.__getstate__())

    def copy(self, other={}):
        ## Shallow copy
        return self.__class__(self).updated(other)

    def update(self, other):
        dict.update(self, other)
        self._trickledown()

    def updated(self, other):
        """ chain-update
        Same as dict.update except that it returns self and therefore
        supports call-chaining. E.g. Properties(other).update(other2).update(other2)
        """
        dict.update(self, other)
        return self

    def _trickledown(self):
        """
        Trickle changes down to depending parameters in sub-tree(s).
        (For same level dependencies use LambdaFunctions instead.). Useful in auto-updting complex parameter hierarchies
        where LambdaVals won't work or would make the logic too complex.
        Call at the end of __init__ and end of update.
        """
        pass

    def to_picklable_dict(self):
        return to_picklable_dict(self)

    def to_table(self, prefix=None):
        """
        Returns a table - a 2D numpy array - with each row containing a list of two items - the key and the value.
        Nested key names are appended consequtively using dot (.) as the separator.
        All values resolved but not validated.
        Used for debugging and pretty printing. Will not throw an exception
        if invalid/unset values are detected in order to be useful for debuggweing.
        Args:
            prefix: Name of a prefix to prepend to all key names. Its value is non-none when called recursively
            to expand a nested Property object.
        """
        rows = []
        if prefix is None:
            rows.append(['NAME','VALUE'])

        for key in sorted(self.keys()):
            row_name = key if prefix is None else prefix + '.' + key
            ## Resolve LambdaVals but do not validate them because we
            ## need this method to work for debugging purposes, therefore we need to
            ## see the state of the dictionary - especially the invalid values.
            val = self._get_unvalidated_val(key)
            if isinstance(val, Properties):
                rows.extend(val.to_table(row_name))
            elif issequence(val):
                for i, v in enumerate(val, start=1):
                    name = '%s.%d'%(row_name,i)
                    if not isinstance(v, Properties):
                        rows.append([name, unicode(v)])
                    else:
                        rows.extend(v.to_table(name))
            # elif isinstance(val, dict):
            #     for k, v in val.iteritems():
            #         name = '%s.%s'%(row_name,k)
            #         if not isinstance(v, Properties):
            #             rows.append([name, str(v)])
            #         else:
            #             rows.extend(v.to_table(name))
            else:
                rows.append([row_name, val])

        return np.asarray(rows, dtype=np.unicode_)

    def pformat(self):
        return pprint.pformat(to_picklable_dict(self))

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

class Undefined(object):
    pass
_undefined = Undefined()

class ParamDesc(Properties):
    """
    A property descriptor.
    """
    def __init__(self, name, text, validator=None, default=_undefined):
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

        if isMutable(default):
            raise AttributeError('ParamDesc.default values must be immutable! Property name: %s.'%name)

        if default == _undefined:
            Properties.__init__(self, {'name':name, 'text':text, 'validator':validator})
        else:
            # if default is None:
            #     print 'WARNING: Setting None default value for property %s'%name

            Properties.__init__(self, {'name':name, 'text':text, 'validator':validator, 'default':default})

        self.freeze()

    # def defaultIsSet(self):
    #     """
    #     Returns True if a default value has been set else returns False.
    #     Note that if the default value was None and if None was a valid value (per the validator if set)
    #     then this method will return True.
    #     """
    #     if self.default is not None:
    #         return True
    #     elif self.validator is None:
    #         return True
    #     else:
    #         return None in self.validator

    def defaultIsSet(self):
        """
        Returns True if a default value has been set else returns False.
        """
        return 'default' in self

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

class OneValError(Exception):
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

    def __init__(self, prototype, initVals=None, seal=False, assert_one_val=False):
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

        @param seal (bool): seals the object after initialization. Not really needed for
            this class since the keys are already fixed by the prototype. Will get
            depricated soon.

        @param assert_one_val: (bool) Turns on one_val behaviour (see documentation for HyperParams).
            By default the behaviour is turned off.

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
        object.__setattr__(self, '_do_assert_one_val', assert_one_val)

        descriptors = prototype
        descrs = Properties()
        vals_init_ = Properties()
        vals_params_ = Properties()
        _vals = {}

        if initVals is not None:
            vals_init_ = initVals if isinstance(initVals, Properties) else Properties(initVals)

        if isinstance(prototype, Params):
            descriptors = prototype.protoS
            vals_params_ = prototype

        object.__setattr__(self, '_descr_list', tuple(descriptors))

        for desc in descriptors:
            name = desc.name
            if name not in descrs:
                try:
                    descrs[name] = desc

                    if self._do_assert_one_val:
                        vals = self._assert_one_val(name, [vals_init_, vals_params_], desc)
                        if len(vals) > 0:
                            _vals[name] = self._assert_immutable(vals[0], name)
                        # else do not insert key into dictionary
                    else:
                        if name in vals_init_:
                            _vals[name] = self._assert_immutable(vals_init_._rv(name), name)
                        elif name in vals_params_:
                            _vals[name] = self._assert_immutable(vals_params_._rv(name), name)
                        elif desc.defaultIsSet():
                            _vals[name] = self._assert_immutable(desc.default, name)
                        # else do not insert key into dictionary

                except:
                    dtc.logger.critical('##### Error while processing property: %s\n', desc)
                    raise
            else:
                raise ParamsValueError('property %s has already been initialized with value %s'%(name, _vals[name]))

        object.__setattr__(self, '_descr_dict', descrs.freeze())

        # Validation: Now insert the property values one by one. Doing so will invoke
        # self._set_val_ which will validate the values.
        for desc in descriptors:
            try:
                _name = desc.name
                if _name in _vals:
                    self[_name] = _vals[_name]
            except ParamsValueError:
                raise
            except:
                dtc.logger.critical('##### Error while processing property: %s\n', desc)
                raise

        # Finally, seal the object so that no new properties may be added.
        if seal:
            self.seal()

    @staticmethod
    def _assert_immutable(val, name):
        ## warn if doing shallow-copy of a dictionary
        if isMutable(val):
            raise ParamsValueError('Shallow initializing mutable object is not allowed. Freeze object before initializing: %s =\n%s'%(name, pformat(val)))
        return val

    @staticmethod
    def _assert_one_val(name, prop_list, desc):
        num_set = 0
        vals = []
        for p in prop_list:
            if name in p:
                if (num_set == 0) or (p._rv(name) not in vals):
                    num_set += 1
                    vals.append(p._rv(name))

        if desc.defaultIsSet() and (desc.default not in vals):
            num_set += 1
            vals.append(desc.default)

        if num_set > 1:
            raise OneValError('%s has different value set. vals =\n%s'%(name, vals))
        else:
            return vals

    # def _set_val_(self, name, val):
    #     """
    #     This is a wrapper around super._set_val_. It is meant to warn the user if they set the same value twice.
    #     This is expected to happen only by mistake because hyper parametes are usually expected to be set only
    #     once.
    #
    #     """
    #     if name in self and (val != self[name]):
    #         raise Exception('%s._set_val_: Existing value of %s, %s is being overridden by %s' %
    #                         (self.__class__.__name__, name, self[name], val))
    #         # dtc.logger.warn('%s._set_val_: Existing value of %s, %s is being overridden by %s', self.__class__.__name__, name, self[name], val)
    #         # traceback.print_stack()
    #
    #     return Params._set_val_(self, name, val)


    def _set_val_(self, name, val):
        """
        Actual _set_val_ implementation separated out so that internal and external invocations may be distinguished.
        """
        # Polymorphic override of _set_val_. Be mindful of recursion.
        protoD = self.protoD

        if not self.isValidName(name):
            raise KeyError('%s is not an allowed property name'%(name,))
        elif self._do_assert_one_val and (name in self) and (val != self[name]):
            raise OneValError('%s._set_val_: Attempt to change the existing value of %s\nOld Value=\n%s\nNew Value=\n%s' %
                            (self.__class__.__name__, name, self[name], val))
        # elif (val is not None) and (not isinstance(val, LambdaVal)) and (protoD[name].validator is not None) and (val not in protoD[name].validator):
        elif (not isinstance(val, LambdaVal)) and (protoD[name].validator is not None) and (val not in protoD[name].validator):
            raise ParamsValueError('%s is not a valid value of property %s'%(val, name))
        else:
            return Properties._set_val_(self, name, val)


    def _resolve_raw_val(self, name, val):
        """Check to see if a value is dynamic and resolve it if so. """
        if isinstance(val, LambdaVal):
            val = val(name, self)
        return val

    def _resolve_raw_vals(self, name, vals_, doValidate=True):
        """ Resolve a single (possibly lambda) val or sequence of (possibly lambda) vals. """
#        if isTupleOrList(vals_):
        if issequence(vals_):
            vals_l = [self._resolve_raw_val(name, val) for val in vals_]
            vals = vals_.__class__(vals_l)
        else:
            vals = self._resolve_raw_val(name, vals_)

        if doValidate:
            protoD = self.protoD
            if (vals is not None) and (protoD[name].validator is not None) and (vals not in protoD[name].validator):
                raise ParamsValueError('%s is not a valid value of property %s'%(vals, name))

        return vals

    def _rv(self, name):
        """
        Return raw-value - i.e. unresolved Lambda Values.
        """
        return Properties._get_val_(self, name)

    def _rvn(self, name):
        """
        Return raw-value if key exists else return None.
        """
        try:
            return self._rv(name)
        except KeyError:
            return None

    def _get_val_helper(self, name, doValidate):
        if not self.isValidName(name):
            raise KeyError('%s is not an allowed property name'%(name,))
        else:
            val = Properties._get_val_(self, name)
            return self._resolve_raw_vals(name, val, doValidate)

    def _get_val_(self, name):
        """ Resolves and validates values returned by Properties._get_val_ """
        return self._get_val_helper(name, doValidate=True)

    def _get_unvalidated_val(self, name):
        """ Resolves but does not validate values returned by Properties._get_val_ """
        return self._get_val_helper(name, doValidate=False)

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

    @staticmethod
    def from_pickle(*paths):
        """ WARNING: Partial implementation: Delegates to  Properties.from_pickle and returns a Properties object. """
        return Properties.from_pickle(*paths)

    @property
    def protoS(self):
        # self._descr_list won't recursively call _get_val_ because __getattribute__ will return successfully
        #return self._descr_list
        return object.__getattribute__(self, '_descr_list')

    @property
    def protoD(self):
        # self._descr_dict won't call __getattr__ because __getattribute__ returns successfully
        #return self._descr_dict
        return object.__getattribute__(self, '_descr_dict')

class HyperParams(Params):
    """
    Params class specialized for Global HyperParams. Hyper Params are expected to be global and
    set only in one place (in order to avoid mistakes). However, manually hard-coding all hyper-params in
    one file is not practical because for one, parameters are in many cases dependant on one-another. Thus changing
    one such param could result in a cascading affect causing one to manually change all the dependant params. Doing
    so manually is not only cumbersome and laborious but also error-prone since one may not always remember all the
    dependant params (you could have 200 params easily). Furthermore, it may be beneficial to organize hyper-params
    in a containment hierarchy - to mimic the containment structure of a model's components. Secondly, the model
    itself may change based on parameters for e.g. sections of a model may be added or removed based on params. Each
    of these added or deleted sections may be governed by their own set of parameters which will need to be specified
    or unspecified easily. Furthermore each section may have its own constraints regarding valid and invalid values
    of the parameters for e.g. you may decide to allow dropouts only within certain sections of your model. There
    can be a variety of constraints that you may want to impose on different sections of your model. All these
    constraints can be expressed as constraints on the values of parameters. Also, different sections of a model
    may share a lot of parameters (e.g. regularizer function across the entire model or reuse most of the parameters
    across layers of a convolutional neural network.). It would be very convenient to specify the shared parameters
    only once instead of having to make copies of them (manually). Also, many places in your code may dynamically
    set parameters based on some other factors - e.g. data-set shape may be different for different data-sets causing
    slight changes to the structure of your model (e.g. you may want a different number of hidden units if the vocabulary
    of your normalized vs non-normalized text data-sets was different). Lastly, you may have various checks in your
    code that verify the shape and size of various tensors - in order to validate the correctness of your model.
    These checks/asserts will need to change if anything changes for any of the reasons mentioned above. Therefore
    an advanced hyper-parameter module is required which has aspects of hierarchy, reuse/inheritance, is
    declarative but at the same time can be constructed dynamically but that doesn't lose benefits of the simpler
    manual method (i.e. hard-code everything in one file in one place). This class along with its superclasses
    attempts to provide all these benefits. The superclasses (Params and Properties) provide a declarative yet dynamic
    API (via. parameter-descriptors, validators and Lambda-Values), aspects of object-oriented architecture -
    i.e. inheritance and containment via. parameter classes and a host of other utilities like pickling/unpickling
    and visual differencing of parameter-sets (via Jupyter Notebook) and JavaScript style property accessor (dot accessor)
    for all properties in addition to python-dict style (square brackets) accessor, and object freeze and seal similar
    to JavaScript Objects.
    On top the dynamic nature we add a few constraints in order to ensure the benefits of the simple hard-coded parameter
    model - 1) assert_one_val: It will not allow you to change the value of a property once set. While you can choose to set it
    declaratively within the parameter descriptor or dynamically within your code, you can not change the value once set.
    Attempting to do so will beget an exception from this class (though the super-classes will allow this
    behaviour so if you want that behaviour then just use the super-class though I recommend its better to stay with
    all the constraints if you are using this class for global hyper-parameters especially when the number of params
    is in the hundreds). 2) It will prevent
    you from inserting or reading properties that are not declared in the parameter descriptor (the Params superclass
    enforces this behaviour) and 3) Trying to access a property with a None value will cause an exception to be
    raised unless None was declared as a valid value of the property. Note that this behaviour may be depricated in
    favor of treating None as a first-class value.

    This class imposes the following semantics on top of its superclass (Params).

    1)
        NOTE: This feature needs revision since it applies to old behaviour of the Params class.
        In the new behaviour a 'None' value is considered a proper
        value as far as the Params class is concerned. A unset property will not have
        a key in the dictionary - different from the old behaviour wherein unset properties
        received a value of None. Since the None value is not treated in a special way
        anymore, this feature probably is not necessary.

        If a key has value None, then it is deemed absent from the dictionary. Calls
        to __contains__ and _get_val_ will beget a KeyError - as if the property was
        absent from the dictionary. This is necessary to catch cases wherein one
        has forgotten to set a mandatory property. Mandatory properties must not have
        default values in their descriptor. A property that one is okay forgetting
        to specify should have a None default value set in its descriptor which may
        include 'None'.
        However as with the Params class, it is still possible to initialize or set a property value to
        None eventhough None may not appear in the valid-list. We allow this in
        order to enable lazy initialization - i.e. a case where the
        code may wish to initialize a property to None, and set it to a valid value
        later. Setting a property value to None tantamounts to unsetting / deleting
        the property.
    2)  assert_one_val: Disallows changing the value of a set parameter. HyperParam values are
        expected to be global and hence setting two different values for a property may indicate an error/bug in the code.
        If you want to lazily set the value of a hyper-parameter, then leave it unset at first (for e.g.
        do not provide a value in the parameter's prototype).
    """

    def __init__(self, prototype, initVals=None, seal=False, assert_one_val=False):
        Params.__init__(self, prototype, initVals, seal, assert_one_val=True)

    def __contains__(self, name):
        """ Handles None values in a special way as stated above. """
        try:
            self._get_val_(name)
            return True
        except KeyError:
            return False

    def _get_val_(self, name):
        """ Polymorphic override of _get_val_ """
        # Be mindful of recursion.
        val = Params._get_val_(self, name)
        validator = self.protoD[name].validator
        if (val == None) and ((validator is None) or (None not in validator)):
            raise KeyError('property %s was not set'%(name,))
        else:
            return val

    def _get_unvalidated_val(self, name):
        """Return resolved but unvalidated value"""
        return Params._get_unvalidated_val(self, name)

## Abstract parameter validator class.
class _ParamValidator(object):
    def __contains__(self, val):
        raise NotImplementedError('%s is an abstract class.'%self.__class__.__name__)

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


# A validator that returns False for None values
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

class either(_ParamValidator):
    """Logical OR of multiple validators"""
    def __init__(self, *validators):
        self._validators = validators
    def __contains__(self, obj):
        return any((obj in v for v in self._validators))

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

# def isFunction(v):
#     return inspect.isfunction(v) or isinstance(v, LambdaVal)

def isMutable(v):
    if isinstance(v, Properties):
        return (not v.isFrozen())
    else:
        return isinstance(v, collections.MutableSequence) or isinstance(v, collections.MutableMapping) or isinstance(v, collections.MutableSet) or (issequence(v) and any([isMutable(e) for e in v]) )

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
booleanOrNone = instanceofOrNone(bool)

def squashed_seq_list(np_seq_batch, seq_lens, remove_val1=None, remove_val2=None, eos_token=0):
    assert np_seq_batch.ndim == 2
    assert seq_lens.ndim == 1
    assert np_seq_batch.shape[0] == seq_lens.shape[0]
    assert eos_token == 0
    sq_list = []
    for i, np_seq in enumerate(np_seq_batch):
        seq_len = seq_lens[i]
        trunc = np_seq[:seq_len]
        if (seq_len>1) and (trunc[-2]==0):
            trunc = np.trim_zeros(trunc, 'b')
        elif (seq_len>0) and trunc[-1]==0:
            trunc = trunc[:-1]
        if remove_val1 is not None:
            trunc = trunc[trunc != remove_val1]
        if remove_val2 is not None:
            trunc = trunc[trunc != remove_val2]

        # sq_list.append(trunc.tolist())
        sq_list.append(trunc)

    return sq_list

# def get_bleu_weights(max_len=200, frac=1.0):
#     weights = [None]
#     for i in range(1, max_len+1):
#         o = int(np.ceil([i*(1.-frac)])[0])
#         if o == 0:
#             o = 1
#         w = np.ones(o, dtype=float) / o
#         z = np.zeros(i-o, dtype=float)
#         weights.append( np.concatenate((z, w)) )
#     return weights
# BLEU_WEIGHTS = get_bleu_weights()

def sentence_bleu_scores(predicted_ids, predicted_lens, target_ids, target_lens, space_token=None, blank_token=None, eos_token=None):
    """
    Removes space-tokens from predicted_ids and then computes the bleu scores of a batch of
    sequences.
    Args:
        predicted_ids: Numpy array of predicted sequence tokens - (B, T)
        predicted_lens: Numpy array of predicted sequence lengths - (B,)
        target_ids: Numpy array of reference sequence tokens - (B, T'). All sequences are expected to have no blank or space tokens. Furthermore,
            the target ID must be terminated with one (and only one) eos_token.
        target_lens: Numpy array of reference sequence lengths - (B,)
        space_token: (token-type) Space token to remove from predicted_ids
        blank_token: (token-type) CTC Blank token to remove from predicted_ids
    """
    assert len(predicted_ids) == len(target_ids)
    dtc.logger.debug("tfc.sentence_bleu_scores: computing sentence_bleu scores for %d sentences", len(predicted_ids))
    scores = []
    squashed_ids = squashed_seq_list(predicted_ids, predicted_lens, space_token, blank_token, eos_token=eos_token)
    for i, predicted_seq in enumerate(squashed_ids):
        target_len = target_lens[i]
        ## remove the eos_token from target sequence. All target sequences are expected to be squashed with a eos_token at the end.
        target = target_ids[i][:target_len-1]
        score = nltk.translate.bleu_score.sentence_bleu([target],
                                                        predicted_seq)
                                                        # weights=BLEU_WEIGHTS[target_len])
        scores.append(score)

    return scores

def corpus_bleu_score_sq(predicted_ids, predicted_lens, target_ids, target_lens, space_token=None, blank_token=None, eos_token=None):
    """
    Removes space-tokens from predicted_ids and then computes the bleu scores of a batch of
    sequences.
    Args:
        predicted_ids: Numpy array of predicted sequence tokens - (B, T)
        predicted_lens: Numpy array of predicted sequence lengths - (B,)
        target_ids: Numpy array of reference sequence tokens - (B, T'). All sequences are expected to have no blank or space tokens. Furthermore,
            the target ID must be terminated with one (and only one) eos_token.
        target_lens: Numpy array of reference sequence lengths - (B,)
        space_token: (token-type) Space token to remove from predicted_ids
        blank_token: (token-type) CTC Blank token to remove from predicted_ids
    """
    assert len(predicted_ids) == len(target_ids)
    dtc.logger.debug("tfc.corpus_bleu_score_sq: computing corpus_bleu for %d sentences", len(predicted_ids))
    hypotheses = squashed_seq_list(predicted_ids, predicted_lens, space_token, blank_token, eos_token=eos_token)
    targets = [[target[:(target_lens[i]-1)]] for i, target in enumerate(target_ids)]
    return nltk.translate.bleu_score.corpus_bleu(targets, hypotheses) # weights=BLEU_WEIGHTS[target_len])

def corpus_bleu_score(predicted_ids, target_ids):
    """
    Computes the bleu scores of a list of predicted sequences against a corresponding list of target sequences
    Args:
        predicted_ids: list of predicted word/id sequences - list(id_seq)
        target_ids:    list of target word/id sequences -    list(id_seq)
    """
    assert len(predicted_ids) == len(target_ids)
    dtc.logger.debug("tfc.corpus_bleu_score: computing corpus_bleu for %d sentences", len(predicted_ids))
    targets = [[target] for target in target_ids]
    return nltk.translate.bleu_score.corpus_bleu(targets, predicted_ids)  # weights=BLEU_WEIGHTS[target_len])

def diff_dict(left, right):
    """ Assymetric comparison of left with right """
    f = {}
    f2 = {}
    for k,v in left.iteritems():
        if k in right:
            v2 = right[k]
            if isinstance(v, dict) and isinstance(v2, dict):
                diff, diff2 = diff_dict(v, v2)
                if diff != {}:
                    f[k] = diff
                if diff2 != {}:
                    f2[k] = diff2
            elif issequence(v) and all((isinstance(v_i, dict) for v_i in v)) and issequence(v2) and all((isinstance(v_i, dict) for v_i in v2)) and len(v) == len(v2):
                for i, v_i in enumerate(v):
                    k_i = '%s_%d'%(k,i+1)
                    diff, diff2 = diff_dict(v_i, v2[i])
                    if diff != {}:
                        f[k_i] = diff
                    if diff2 != {}:
                        f2[k_i] = diff2
            elif v != v2:
                if isinstance(v, str) and (v.startswith('<function') or v.startswith('<tensorflow')):
                    f2[k] = '%s != %s'%(v, v2)
                else:
                    f[k] = '%s != %s'%(v, v2)
    return f, f2

def to_picklable_dict(props, to_str=False):
    """
    Returns a dictionary with all values resolved but not validated.
    Used for debugging and pretty printing. Will not throw an exception
    if invalid/unset values are detected in order to be useful for debugging.
    If to_str is True all values are converted to their string representations.

    Args:
        props: A instance of dict or Properties.
        to_strs: If True, all values will be converted into their string representations.
    Returns:
        A picklable dict object.
    """
    def get_repr(v):
        if to_str and (not isinstance(v, str)):
            return repr(v)
        else:
            return v

    # Very important to return a dict, not Properties object because in that case unpickling of
    # the data file would be dependent on the dl_commons module. Returning a dict instead makes
    # the unpickling process depend on the builtin dict module only.
    resolved = {}
    for key in props.keys():
        ## Resolve LambdaVals but do not validate them because we
        ## need this method to work for debugging purposes, therefore we need to
        ## see the state of the dictionary - especially the invalid values.
        if isinstance(props, Properties):
            val = props._get_unvalidated_val(key)
        else:
            val = props[key]

        if isinstance(val, dict):
            resolved[key] = to_picklable_dict(val)
        elif issequence(val):
            resolved[key] = [(to_picklable_dict(v) if isinstance(v, dict) else get_repr(v) ) for v in val]
        else:
            resolved[key] = get_repr(val)

    return resolved

def to_flat_dict(dict_obj):
    """
    Returns the result of flattening to_picklable_dict(self).
    Args:
        dict_obj: A dict object (including subclasses).
    """
    def _flatten(prefix, d, f):
        for k in d.keys():
            v = d[k]
            if isinstance(v, dict):
                _flatten(prefix+k+'.', v, f)
            elif issequence(v) and all((isinstance(v_i, dict) for v_i in v)):
                for i, v_i in enumerate(v, start=1):
                    _flatten(prefix+'%s_%d.'%(k,i), v_i, f)
            else:
                f[prefix+k] = v
        return f
    d = to_picklable_dict(dict_obj)
    return _flatten('', d, {})

def to_set(dict_obj, sep=' ===> '):
    """
    Returns:
        set(['key1 ==> val1', 'key2 ==> val2', ...])

    Returns a set of key-value pair strings separated by the sep argument. All keys
    and values are converted to their string representations. Key value pairs are
    derived by calling self.to_flat_dict.

    """
    return set(['%s%s%s'%(e[0],sep,e[1]) for e in to_flat_dict(dict_obj).iteritems()])

# def to_set(self):
#     """
#     Returns the result of flattening self.to_picklable_dict()
#     """
#     def _flatten(prefix, d, f):
#         for k in d.keys():
#             v = d[k]
#             if isinstance(v, dict):
#                 _flatten(prefix+k+'.', v, f)
#             elif issequence(v) and all((isinstance(v_i, dict) for v_i in v)):
#                 for i, v_i in enumerate(v, start=1):
#                     _flatten(prefix+'%s_%d.'%(k,i), v_i, f)
#             else:
#                 f.add('%s%s:%s'%(prefix, k, v))
#         return f
#     d = self.to_picklable_dict()
#     return _flatten('', d, set())

def diff_table(dict_obj, other):
    """
    Returns two numpy string array with differing values between the two dictionaries.
    The values are all converted to their string representations before comparison.
    Meant for importing into pandas for quick viewing. The first array has the more
    important differences whereas the second one has the less important diffs.
    """
    sep = ' ===> '
    def extract_keys(s):
        return set([''.join(e.split(sep)[:-1]) for e in s])

    s1 = to_set(dict_obj, sep)
    s2 = to_set(other, sep)
    keys = extract_keys(s1 ^ s2)

    d1 = to_flat_dict(dict_obj)
    d2 = to_flat_dict(other)

    ## Move all function and tensorflow objects to end of the list since these are spurious differences.
    def cullKey(k):
        v = d1[k] if (k in d1) else d2[k]
        v = repr(v) if not isinstance(v, str) else v
        return  (v.startswith('<function') or v.startswith('<tensorflow') or v.startswith('<logging.Logger') or ('logdir' in k))


    tail_keys = set(filter(cullKey, keys))
    head_keys = keys - tail_keys ## filter(lambda k: k not in tail_keys, keys)
    keys = sorted(list(head_keys)) + sorted(list(tail_keys))

    np_head = np.asarray([ ['%s%s%s'%(k, sep, d1[k] if d1.has_key(k) else 'undefined') , '%s%s%s'%(k,sep,d2[k] if d2.has_key(k) else 'undefined')] for k in head_keys ]).reshape(-1,2)
    np_tail = np.asarray([ ['%s%s%s'%(k, sep, d1[k] if d1.has_key(k) else 'undefined') , '%s%s%s'%(k,sep,d2[k] if d2.has_key(k) else 'undefined')] for k in tail_keys ]).reshape(-1,2)

    return np_head, np_tail

def pformat(v):
    if isinstance(v, Properties):
        return v.pformat()
    else:
        return pprint.pformat(v)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_dupes(lst):
    return set([item for item in lst if lst.count(item) > 1])