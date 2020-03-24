# todo - implemenet boundary pixel iterable, iterator, and accessor
# todo - implement cell neighbor iterable, iterator, and accessor
# todo - implement fields with support for parallelism
# todo - implement focal point plasticity iterable, iterator, and accessor
# todo - implement additional steppable methods on CC3DJitUtils (to discuss with group)

import operator
from inspect import getmembers, isclass

from cc3d.CompuCellSetup import persistent_globals as pg
from cc3d.core import iterators as cc3d_iter
from cc3d.cpp import CompuCell
from cc3d.cpp.CompuCell import CellG, CellInventory, Point3D

from llvmlite import ir
from llvmlite.llvmpy.core import Type as llvmtype
import numba
from numba import cgutils, types
from numba.extending import typeof_impl, type_callable
from numba.extending import models, register_model
from numba.extending import make_attribute_wrapper, overload, overload_method, overload_attribute
from numba.extending import lower_builtin
from numba.extending import unbox, box, NativeValue
from numba.pythonapi import PythonAPI
from numba.targets import boxing
from numba.targets.imputils import iterator_impl
from numba.types import boolean, float64, int64
from numba.typing.arraydecl import NestedArrayAttribute
from numba.typing.templates import AttributeTemplate, infer_getattr
import numpy as np

numba.config.THREADING_LAYER = 'threadsafe'
numba.npyufunc.parallel._launch_threads()

registered_functions = {}


# For jitting steppable static methods
def jit_steppable_method(*args, **kwargs):

    def wrap(fcn):
        if fcn not in registered_functions.keys():
            registered_functions[fcn] = staticmethod(numba.jit(fcn, *args, **kwargs))

        return registered_functions[fcn]

    return wrap


# Maps from Numba types to Numba-distributed Python API
_unbox_map = {
    int64: PythonAPI.long_as_longlong,
    float64: PythonAPI.float_as_double,
    boolean: PythonAPI.object_istrue
}
_box_map = {
    int64: PythonAPI.long_from_longlong,
    float64: PythonAPI.float_from_double,
    boolean: PythonAPI.bool_from_bool
}


# Convenience functions

def _get_numba_pyapi(context, builder) -> PythonAPI:
    return context.get_python_api(builder)


def _unbox_pyobj(obj):
    return boxing.unbox_pyobject(None, obj, None)


def _box_pyobj(val):
    return boxing.box_pyobject(None, val, None)


def _get_cell_inventory():
    from cc3d.cpp.CompuCell import Potts3D
    potts: Potts3D = pg.simulator.getPotts()
    return potts.getCellInventory()


# ------------------------------- Numba-compatible Swig ------------------------------


# Generic Numba Type
def _jit_type_class_factory(_class_name: str):

    class _JitTypeTemplate(types.Type):
        mutable = True

        def __init__(self):
            super().__init__(name=_class_name)

    _class_jit_name = _class_name + 'JitType'
    _class_jit = type(_class_jit_name, (_JitTypeTemplate,), {})
    return _class_jit


# Generic class representation in lower layer; it just carries a reference to the original Python object
def _jit_model_class_factory(class_jit):
    assert isclass(class_jit), ValueError

    class _JitModelTemplate(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('pyobj', types.pyobject)
            ]
            models.StructModel.__init__(self, dmm, fe_type, members)

    _class_jit_name = class_jit.__name__
    register_model(class_jit)(type(_class_jit_name, (_JitModelTemplate,), {}))


# Generic class to help with type inferencing of class attributes
def _jit_attribute_infer_factory(class_jit, cls_attr_list):
    assert isclass(class_jit), ValueError

    class _JitAttributeTemplate(AttributeTemplate):
        key = class_jit

    def _attr_method_factory(_attr_name, _numba_typ):
        _method_name = 'resolve_' + _attr_name

        def _method_impl(self, _obj):
            return _numba_typ

        return _method_name, _method_impl

    attr_dict = {}
    for attr_name, numba_typ in cls_attr_list:
        method_name, method_impl = _attr_method_factory(attr_name, numba_typ)
        attr_dict[method_name] = method_impl

    _class_model_name = class_jit.__name__.replace('JitType', '') + 'Model'
    _class_model = type(_class_model_name, (_JitAttributeTemplate,), attr_dict)

    infer_getattr(_class_model)


# Generic SWIG get/set lowering
def _jit_lower_attr_factory(_cls_cc3d, _cls_jit, cls_attr_list):
    if '__swig_getmethods__' not in dict(getmembers(_cls_cc3d)).keys():
        return

    def _lower_getattr(_attr_name, _numba_typ):

        # Lower attribute getter

        @numba.extending.lower_getattr(_cls_jit, _attr_name)
        def _lower_getattr_impl(context, builder, typ, value):
            _attr_typ = _numba_typ
            _unbox_val_fcn = _unbox_map[_attr_typ]

            _pyapi: PythonAPI = context.get_python_api(builder)

            _model = cgutils.create_struct_proxy(typ)(context, builder, value=value)

            attr_obj = _pyapi.object_getattr_string(_model.pyobj, _attr_name)
            attr_val = _unbox_val_fcn(_pyapi, attr_obj)

            _pyapi.decref(attr_obj)

            return attr_val

    def _lower_setattr(_attr_name, _numba_typ):

        # Lower attribute setter

        @numba.extending.lower_setattr(_cls_jit, _attr_name)
        def _lower_setattr_impl(context, builder, sig, args):
            _obj_val, _attr_val = args
            _obj_typ, _attr_typ = sig.args

            _model = cgutils.create_struct_proxy(_obj_typ)(context, builder, value=_obj_val)

            _pyapi = _get_numba_pyapi(context, builder)
            _box_val_fcn = _box_map[_attr_typ]
            _attr_obj = _box_val_fcn(_pyapi, _attr_val)

            _pyapi.object_setattr_string(_model.pyobj, _attr_name, _attr_obj)

            _pyapi.decref(_attr_obj)

            return

    for attr_name, numba_typ in cls_attr_list:
        if attr_name in _cls_cc3d.__swig_getmethods__.keys():
            _lower_getattr(attr_name, numba_typ)

        if attr_name in _cls_cc3d.__swig_setmethods__.keys():
            _lower_setattr(attr_name, numba_typ)


# Generic class boxing and unboxing during raising/lowering
def _box_unbox_factory(_cls_jit):
    @unbox(_cls_jit)
    def unbox_cls(typ, obj, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        _model.pyobj = obj

        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return NativeValue(_model._getvalue(), is_error=is_error)

    @box(_cls_jit)
    def box_cls(typ, val, c):
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        res = _model.pyobj
        return res


# Generic class factory for jitting. Use only for purely abstract classes (e.g., CellG). Classes with particularized
# JIT representations should be constructed manually
def _jit_class_factory(cls_cc3d, cls_attr_list):
    assert isclass(cls_cc3d), ValueError

    # Make Numba type class
    cls_jit = _jit_type_class_factory(cls_cc3d.__name__)
    cls_jit_obj = cls_jit()

    # Register typing
    @typeof_impl.register(cls_cc3d)
    def typeof_cls(val, c):
        return cls_jit_obj

    # Register constructor typing
    @type_callable(cls_cc3d)
    def type_cls_cc3d(context):
        def typer():
            return cls_jit_obj
        return typer

    # Register model
    _jit_model_class_factory(cls_jit)

    # Generate attribute inferencing
    _jit_attribute_infer_factory(cls_jit, cls_attr_list)

    # Lower SWIG attribute getters
    _jit_lower_attr_factory(cls_cc3d, cls_jit, cls_attr_list)

    # Lower constructor
    @lower_builtin(cls_cc3d, types.PyObject)
    def impl_cls_cc3d(context, builder, sig, args):
        typ = sig.return_type
        py_obj, = args
        _model = cgutils.create_struct_proxy(typ)(context, builder)
        _model.py_obj = py_obj
        return _model._getvalue()

    # Register boxing/unboxing routines
    _box_unbox_factory(cls_jit)

    return cls_jit_obj  # fin


# ------------------------------ Numba-compatible CellG ------------------------------
cell_g_attr = [
    ('volume', int64),
    ('targetVolume', float64),
    ('lambdaVolume', float64),
    ('surface', float64),
    ('targetSurface', float64),
    ('angle', float64),
    ('lambdaSurface', float64),
    ('clusterSurface', float64),
    ('targetClusterSurface', float64),
    ('lambdaClusterSurface', float64),
    ('type', int64),
    ('subtype', int64),
    ('xCM', float64),
    ('yCM', float64),
    ('zCM', float64),
    ('xCOM', float64),
    ('yCOM', float64),
    ('zCOM', float64),
    ('xCOMPrev', float64),
    ('yCOMPrev', float64),
    ('zCOMPrev', float64),
    ('iXX', float64),
    ('iXY', float64),
    ('iXZ', float64),
    ('iYY', float64),
    ('iYZ', float64),
    ('iZZ', float64),
    ('lX', float64),
    ('lY', float64),
    ('lZ', float64),
    ('ecc', float64),
    ('lambdaVecX', float64),
    ('lambdaVecY', float64),
    ('lambdaVecZ', float64),
    ('flag', int64),
    ('averageConcentration', float64),
    ('id', int64),
    ('clusterId', int64),
    ('fluctAmpl', float64),
    ('lambdaMotility', float64),
    ('biasVecX', float64),
    ('biasVecY', float64),
    ('biasVecZ', float64)
    # ('connectivityOn', boolean)  # Currently doesn't like booleans
]

_cell_g_constructed = False


def _ctor_cell_g_jit():
    try:
        if not _cell_g_constructed:
            return _jit_class_factory(CellG, cell_g_attr)
        else:
            return None
    except NameError:
        return None


CellGJit = _ctor_cell_g_jit()  # For import
del _cell_g_constructed

# ----------------------------- Numba-compatible Point3D -----------------------------

_class_name = Point3D.__name__
point_3d_attr = [('x', int64), ('y', int64), ('z', int64)]


def _ctor_point_3d():

    # Make Numba type class
    class _Point3DJitType(types.Type):
        mutable = True

        def __init__(self):
            super().__init__(name='Point3D')

    _class_jit_name = _class_name + 'JitType'
    _class_jit = type(_class_jit_name, (_Point3DJitType,), {})

    _Point3DJit = _class_jit()

    # Register typing
    @typeof_impl.register(Point3D)
    def typeof_cls(val, c):
        return _Point3DJit

    # Register constructor typing
    @type_callable(Point3D)
    def type_cls_cc3d_3args(context):
        def typer(x, y, z):
            if isinstance(x, types.Integer) and isinstance(y, types.Integer) and isinstance(z, types.Integer):
                return _Point3DJit

        return typer

    @type_callable(Point3D)
    def type_cls_cc3d_0args(context):
        def typer():
            return _Point3DJit

        return typer

    # Register model
    @register_model(_class_jit)
    class _Point3DModelTemplate(models.StructModel):

        def __init__(self, dmm, fe_type):
            members = point_3d_attr
            models.StructModel.__init__(self, dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class _Point3DAttribute(AttributeTemplate):
        key = _Point3DJitType

        def resolve_x(self, pt):
            return int64

        def resolve_y(self, pt):
            return int64

        def resolve_z(self, pt):
            return int64

    # Lower attribute getters
    @numba.extending.lower_getattr(_class_jit, "x")
    def lower_point3d_get_x(context, builder, typ, value):
        return cgutils.create_struct_proxy(typ)(context, builder, value=value).x

    @numba.extending.lower_getattr(_class_jit, "y")
    def lower_point3d_get_y(context, builder, typ, value):
        return cgutils.create_struct_proxy(typ)(context, builder, value=value).y

    @numba.extending.lower_getattr(_class_jit, "z")
    def lower_point3d_get_z(context, builder, typ, value):
        return cgutils.create_struct_proxy(typ)(context, builder, value=value).z

    # Lower constructors
    @lower_builtin(Point3D, types.VarArg(types.Integer))
    def _impl_point_3d(context, builder, sig, *args):
        typ = sig.return_type

        _model = cgutils.create_struct_proxy(typ)(context, builder)
        if len(*args) == 3:
            _model.x, _model.y, _model.z = list(*args)
        else:
            _model.x, _model.y, _model.z = context.get_constant(int64, 0),\
                                           context.get_constant(int64, 0),\
                                           context.get_constant(int64, 0)
        return _model._getvalue()

    # Register boxing/unboxing routines
    @unbox(_Point3DJitType)
    def unbox_point_3d(typ, obj, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        _x_obj = _pyapi.object_getattr_string(obj, "x")
        _y_obj = _pyapi.object_getattr_string(obj, "y")
        _z_obj = _pyapi.object_getattr_string(obj, "z")

        _unbox_func = _unbox_map[int64]
        _model.x = _unbox_func(_pyapi, _x_obj)
        _model.y = _unbox_func(_pyapi, _y_obj)
        _model.z = _unbox_func(_pyapi, _z_obj)

        _pyapi.decref(_x_obj)
        _pyapi.decref(_y_obj)
        _pyapi.decref(_z_obj)

        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return NativeValue(_model._getvalue(), is_error=is_error)

    @box(_Point3DJitType)
    def box_point_3d(typ, val, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        _box_func = _box_map[int64]
        _x_obj = _box_func(_pyapi, _model.x)
        _y_obj = _box_func(_pyapi, _model.y)
        _z_obj = _box_func(_pyapi, _model.z)

        _cls_obj = _pyapi.unserialize(_pyapi.serialize_object(Point3D))
        res = _pyapi.call_function_objargs(_cls_obj, (_x_obj, _y_obj, _z_obj))

        _pyapi.decref(_x_obj)
        _pyapi.decref(_y_obj)
        _pyapi.decref(_z_obj)
        _pyapi.decref(_cls_obj)

        return res

    # Methods

    # Element-wise binary operations (e.g., pt3 = pt1 + pt2 -> pt3.x = pt1.x + pt2.x, pt3.y = ...)

    def point_3d_lower_operation(context, builder, sig, args, pyop):
        typ = sig.return_type
        val_pt1, val_pt2 = args

        _pyapi = _get_numba_pyapi(context, builder)
        _box_func = _box_map[int64]
        _unbox_func = _unbox_map[int64]

        pt1_x = _box_func(_pyapi, builder.extract_value(val_pt1, 0))
        pt1_y = _box_func(_pyapi, builder.extract_value(val_pt1, 1))
        pt1_z = _box_func(_pyapi, builder.extract_value(val_pt1, 2))
        pt2_x = _box_func(_pyapi, builder.extract_value(val_pt2, 0))
        pt2_y = _box_func(_pyapi, builder.extract_value(val_pt2, 1))
        pt2_z = _box_func(_pyapi, builder.extract_value(val_pt2, 2))

        pt3_x = pyop(pt1_x, pt2_x)
        pt3_y = pyop(pt1_y, pt2_y)
        pt3_z = pyop(pt1_z, pt2_z)

        model_pt3 = cgutils.create_struct_proxy(typ)(context, builder)
        model_pt3.x = _unbox_func(_pyapi, pt3_x)
        model_pt3.y = _unbox_func(_pyapi, pt3_y)
        model_pt3.z = _unbox_func(_pyapi, pt3_z)

        return model_pt3._getvalue()

    @type_callable(operator.add)
    @type_callable(operator.sub)
    def point_3d_op_add_type(context):
        def typer(pt1, pt2):
            if isinstance(pt1, _Point3DJitType) and isinstance(pt2, _Point3DJitType):
                return numba.typing.templates.signature(_Point3DJit, _Point3DJit, _Point3DJit)

        return typer

    @lower_builtin(operator.add, _Point3DJit, _Point3DJit)
    def point_3d_add_lower(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)
        return point_3d_lower_operation(context, builder, sig, args, _pyapi.number_add)

    @lower_builtin(operator.sub, _Point3DJit, _Point3DJit)
    def point_3d_sub_lower(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)
        return point_3d_lower_operation(context, builder, sig, args, _pyapi.number_subtract)

    return _Point3DJit


Point3DJit = _ctor_point_3d()  # For import


# ------------------------ Numba-compatible STLPyIteratorCINV ------------------------
_stl_py_iterator_c_inv_attr = []
STLPyIteratorCINVJit = _jit_class_factory(CompuCell.STLPyIteratorCINV, _stl_py_iterator_c_inv_attr)


# -------------------------- Numba-compatible CellInventory --------------------------
cell_inv_attr = []
CellInventoryJit = _jit_class_factory(CellInventory, cell_inv_attr)


# ----------------------------- Numba-compatible CellList ----------------------------

def _ctor_cell_list():
    # Make Numba type class
    class CellListType(types.IterableType):
        def __init__(self):
            super().__init__(name="CellList")

        @property
        def iterator_type(self):
            return CellListIterType()

        @property
        def dtype(self):
            return CellGJit

    CellListJit = CellListType()

    # Register typing

    @typeof_impl.register(cc3d_iter.CellList)
    def typeof_cell_list(val, c):
        return CellListJit

    # Register constructor typing

    @type_callable(cc3d_iter.CellList)
    def type_cell_list(context):
        def typer():
            return CellListJit

        return typer

    # Register models
    @register_model(CellListType)
    class CellListModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('cinv_pyobj', types.pyobject)
            ]
            super().__init__(dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class _CellListAttribute(AttributeTemplate):
        key = CellListType

    # Lower constructor
    # n/a

    # Register boxing/unboxing routines
    @unbox(CellListType)
    def unbox_cell_list(typ, obj, c):
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        _model.cinv_pyobj = c.pyapi.object_getattr_string(obj, "inventory")
        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return NativeValue(_model._getvalue(), is_error=is_error)

    @box(CellListType)
    def box_cell_list(typ, val, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

        class_obj = _pyapi.unserialize(c.pyapi.serialize_object(cc3d_iter.CellList))
        res = _pyapi.call_function_objargs(class_obj, (_model.cinv_pyobj,))

        _pyapi.decref(class_obj)

        return res

    # Register typing and lowering of class-specific methods, built-in overloads, etc.

    #   "len()"

    @numba.extending.intrinsic
    def _len_cell_list(typingctx, clst):
        sig = numba.typing.signature(int64, CellListJit)

        def codegen(context, builder, sig, args):
            _pyapi = _get_numba_pyapi(context, builder)
            clst_obj, = args
            _clst = _pyapi.from_native_value(CellListJit, clst_obj)
            inv_obj = _pyapi.object_getattr_string(_clst, "inventory")
            res = _pyapi.call_method(inv_obj, "getSize")
            return _pyapi.to_native_value(int64, res).value

        return sig, codegen

    @overload(len)
    def len_cell_list(clst):
        if isinstance(clst, CellListJit.__class__):
            def impl(clst):
                return _len_cell_list(clst)

            return impl

    #   "getitem()"

    @type_callable(operator.getitem)
    def type_cell_list_get_item(context):
        def typer(clst, idx):
            if isinstance(clst, CellListJit.__class__):
                return numba.typing.signature(CellGJit, CellListJit, types.intp)

        return typer

    @lower_builtin(operator.getitem, CellListJit, int64)
    def cell_list_get_item(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)

        clst, idx = args
        # clst_typ, idx_typ = sig.args
        # itr_typ = sig.return_type

        # Get iterator at beginning of cell inventory
        cinv = cgutils.create_struct_proxy(CellListJit)(context, builder, value=clst).cinv_pyobj
        cinv_class = _pyapi.unserialize(_pyapi.serialize_object(CellInventory))

        # Instantiate iterator, initialize with container and set to begin
        inv_iter_class = _pyapi.unserialize(_pyapi.serialize_object(CompuCell.STLPyIteratorCINV))
        cont = _pyapi.call_method(cinv_class, "getContainer", (cinv,))
        inv_iter = _pyapi.call_function_objargs(inv_iter_class, (cont,))
        _pyapi.call_method(inv_iter_class, "setToBegin", (inv_iter,))  # Returns null

        # Move iterator to idx
        ir_one_typ = ir.IntType(1)
        ir_one = ir_one_typ(1)
        ir_zero = ir.IntType(64)(0)
        with builder.if_then(builder.icmp_signed(">", idx, ir_zero)):
            with cgutils.for_range(builder, idx) as loop:
                ret_is_end = _pyapi.call_method(inv_iter, "isEnd")
                is_end = cgutils.as_bool_bit(builder, _pyapi.object_istrue(ret_is_end))

                with builder.if_then(builder.icmp_signed("==", is_end, ir_one)):
                    loop.do_break()
                _pyapi.call_method(inv_iter, "next")

        # Get current iterator reference and return it
        current_ref = _pyapi.call_method(inv_iter, "getCurrentRef")
        res = _pyapi.to_native_value(CellGJit, current_ref).value

        return res

    # ------------------------ Numba-compatible CellListIterator -------------------------

    # Make Numba type class
    class CellListIterType(types.IteratorType):
        def __init__(self):
            super().__init__(name="CellListIterator")

        @property
        def yield_type(self):
            return CellGJit

        @property
        def iterator_type(self):
            return self

    CellListIterJit = CellListIterType()

    # Register typing
    @typeof_impl.register(cc3d_iter.CellListIterator)
    def typeof_cell_list(val, c):
        return CellListIterJit

    # Register constructor typing
    @type_callable(cc3d_iter.CellListIterator)
    def type_cell_list_iter(context):
        def typer(clst):
            if isinstance(clst, CellListType):
                return CellListIterJit

        return typer

    # Register models
    @register_model(CellListIterType)
    class CellListIterModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('inv_iter', types.pyobject),
            ]
            super().__init__(dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class _CellListIterAttribute(AttributeTemplate):
        key = CellListIterType

        def resolve_yield_type(self, clst_iter):
            return CellGJit

        def resolve_iterator_type(self, clst_iter):
            return CellListIterJit

    # Lower constructor as method getiter of class CellListType
    @lower_builtin('getiter', CellListType)
    def impl_clst_iter(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)

        clst, = args
        _model = cgutils.create_struct_proxy(CellListIterJit)(context, builder)

        # Instantiate iterator, initialize with cell inventory container and set to begin
        inv_iter_class = _pyapi.unserialize(_pyapi.serialize_object(CompuCell.STLPyIteratorCINV))
        cinv = cgutils.create_struct_proxy(CellListJit)(context, builder, value=clst).cinv_pyobj
        cont = _pyapi.call_method(cinv, "getContainer")
        inv_iter = _pyapi.call_function_objargs(inv_iter_class, (cont,))
        _pyapi.call_method(inv_iter, "setToBegin")

        # Store prepared iterator in data model
        _model.inv_iter = inv_iter

        return _model._getvalue()

    # Register boxing/unboxing routines
    # n/a

    # Implementation of cell list iterator
    @iterator_impl(CellListType, CellListIterType)
    class ImplCellListIteratorType:
        def __init__(self, context, builder, clst_iter):
            self.context = context
            self.builder = builder
            self.pyapi = _get_numba_pyapi(context, builder)
            self.inv_itr = cgutils.create_struct_proxy(CellListIterJit)(context, builder, value=clst_iter).inv_iter
            self.ir_one_typ = ir.IntType(1)
            self.ir_one = ir.Constant(self.ir_one_typ, 1)

        @property
        def is_not_end(self):
            ret_is_end = self.pyapi.call_method(self.inv_itr, "isEnd")
            res = cgutils.as_bool_bit(self.builder, self.pyapi.object_not(ret_is_end))
            self.pyapi.decref(ret_is_end)
            return res

        def _yield(self):
            _model = cgutils.create_struct_proxy(CellGJit)(self.context, self.builder)
            _model.pyobj = self.pyapi.call_method(self.inv_itr, "getCurrentRef")

            return _model._getvalue()

        def iternext(self, context, builder, result):
            result.yield_(self._yield())
            result.set_valid(self.is_not_end)
            self.pyapi.call_method(self.inv_itr, "next")

    return CellListJit, CellListIterJit


CellListJit, CellListIterJit = _ctor_cell_list()


# ------------------------ Numba-compatible PixelTrackerData -------------------------


def _ctor_pixel_tracker_data():
    # Make Numba type class
    class PixelTrackerDataType(types.Type):
        def __init__(self):
            super().__init__(name="PixelTrackerData")

    PixelTrackerDataJit = PixelTrackerDataType()

    # Register typing
    @typeof_impl.register(CompuCell.PixelTrackerData)
    def typeof_pixel_tracker_data(val, c):
        return PixelTrackerDataJit

    # Register constructor typing
    @type_callable(CompuCell.PixelTrackerData)
    def type_pixel_tracker_data(context):
        def typer():
            return PixelTrackerDataJit

        return typer

    # Register model
    @register_model(PixelTrackerDataType)
    class PixelTrackerDataModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('pixel_pyobj', types.pyobject)
            ]
            models.StructModel.__init__(self, dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class PixelTrackerDataAttribute(AttributeTemplate):
        key = PixelTrackerDataType

        def resolve_pixel(self, ptd):
            return Point3DJit

    # Lower attribute getters
    @numba.extending.lower_getattr(PixelTrackerDataType, "pixel")
    def lower_pixel_tracker_data_get_pixel(context, builder, typ, value):
        _pyapi = _get_numba_pyapi(context, builder)
        pixel_obj = cgutils.create_struct_proxy(typ)(context, builder, value=value).pixel_pyobj
        return _pyapi.to_native_value(Point3DJit, pixel_obj).value

    # Register boxing/unboxing routines
    @unbox(PixelTrackerDataType)
    def unbox_pixel_tracker_data(typ, obj, c):
        _pyapi: PythonAPI = c.pyapi
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        _model.pixel_pyobj = _pyapi.object_getattr_string(obj, "pixel")
        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return NativeValue(_model._getvalue(), is_error=is_error)

    @box(PixelTrackerDataType)
    def box_pixel_tracker_data(typ, val, c):
        _pyapi: PythonAPI = c.pyapi
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        _cls_obj = _pyapi.unserialize(_pyapi.serialize_object(CompuCell.PixelTrackerData))
        res = _pyapi.call_function_objargs(_cls_obj, (c.box(Point3DJit, _model.pixel_pyobj),))

        _pyapi.decref(_cls_obj)
        return res

    return PixelTrackerDataJit


PixelTrackerDataJit = _ctor_pixel_tracker_data()


# Register typing and lowering of class-specific methods, built-in overloads, etc.
#   n/a


# ------------------------ Numba-compatible PixelTrackerPlugin -----------------------
PixelTrackerPluginJit = _jit_class_factory(CompuCell.PixelTrackerPlugin, [])
PixelTrackerJit = _jit_class_factory(CompuCell.PixelTracker, [])


# -------------------------- Numba-compatible CellPixelList --------------------------


def _ctor_cell_pixel_list():
    # Make Numba type class
    class CellPixelListType(types.IterableType):
        def __init__(self):
            super().__init__(name="CellPixelList")

        @property
        def iterator_type(self):
            return CellPixelIterType()

        @property
        def dtype(self):
            return PixelTrackerDataJit

    CellPixelListJit = CellPixelListType()

    # Register typing

    @typeof_impl.register(cc3d_iter.CellPixelList)
    def typeof_cell_pixel_list(val, c):
        return CellPixelListJit

    # Register constructor typing

    @type_callable(cc3d_iter.CellPixelList)
    def type_cell_pixel_list(context):
        def typer(pxt_plugin, cell):
            if isinstance(pxt_plugin, PixelTrackerPluginJit.__class__) and isinstance(cell, CellGJit.__class__):
                return CellPixelListJit

        return typer

    # Register models
    @register_model(CellPixelListType)
    class CellPixelListModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('pixel_tracker_plugin', types.pyobject),
                ('cell', types.pyobject),  # Keep this as pyobject, since it's only stored for boxing/unboxing
                ('pixel_set', types.pyobject)

            ]
            super().__init__(dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class _CellPixelListAttribute(AttributeTemplate):
        key = CellPixelListType

    # Lower constructor
    #   n/a

    # Register boxing/unboxing routines
    @unbox(CellPixelListType)
    def unbox_cell_pixel_list(typ, obj, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)

        _model.pixel_tracker_plugin = _pyapi.object_getattr_string(obj, "pixelTrackerPlugin")

        _model.cell = _pyapi.object_getattr_string(obj, "cell")
        extra_attrib_ptr = _pyapi.object_getattr_string(_model.cell, "extraAttribPtr")
        pixel_tracker_accessor = _pyapi.call_method(_model.pixel_tracker_plugin, "getPixelTrackerAccessorPtr")
        pixel_tracker = _pyapi.call_method(pixel_tracker_accessor, "get", (extra_attrib_ptr,))
        _model.pixel_set = _pyapi.object_getattr_string(pixel_tracker, "pixelSet")

        # _pyapi.decref(pixel_tracker)  # Worked without this, before modifications
        _pyapi.decref(extra_attrib_ptr)
        _pyapi.decref(pixel_tracker_accessor)

        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        res = NativeValue(_model._getvalue(), is_error=is_error)

        return res

    @box(CellPixelListType)
    def box_cell_pixel_list(typ, val, c):
        _pyapi: PythonAPI = c.pyapi
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        class_obj = _pyapi.unserialize(c.pyapi.serialize_object(cc3d_iter.CellPixelList))
        res = _pyapi.call_function_objargs(class_obj, (_model.pixel_tracker_plugin, _model.cell))

        _pyapi.decref(class_obj)
        return res

    # Register typing and lowering of class-specific methods, built-in overloads, etc.

    @type_callable(operator.getitem)
    def type_cell_pixel_list_get_item(context):
        def typer(cpxlst, idx):
            if isinstance(cpxlst, CellPixelListType):
                return numba.typing.signature(PixelTrackerDataJit, CellPixelListJit, types.intp)

        return typer

    @lower_builtin(operator.getitem, CellPixelListJit, int64)
    def cell_pixel_list_get_item(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)

        cpxlst, idx = args
        cpxlst_typ, idx_typ = sig.args
        pixel_set = cgutils.create_struct_proxy(cpxlst_typ)(context, builder, value=cpxlst).pixel_set

        # Get iterator at beginning of pixel set
        pixel_itr_class = _pyapi.unserialize(_pyapi.serialize_object(CompuCell.pixelSetPyItr))
        pixel_itr = _pyapi.call_function_objargs(pixel_itr_class, (pixel_set,))
        _pyapi.call_method(pixel_itr, "setToBegin")

        # Move iterator to idx
        ir_one_typ = ir.IntType(1)
        ir_one = ir_one_typ(1)
        ir_zero = ir.IntType(64)(0)
        with builder.if_then(builder.icmp_signed(">", idx, ir_zero)):
            with cgutils.for_range(builder, idx) as loop:
                ret_is_end = _pyapi.call_method(pixel_itr, "isEnd")
                is_end = cgutils.as_bool_bit(builder, _pyapi.object_istrue(ret_is_end))

                with builder.if_then(builder.icmp_signed("==", is_end, ir_one)):
                    loop.do_break()
                _pyapi.call_method(pixel_itr, "next")

        # Get current iterator reference and return it
        ptd = _pyapi.call_method(pixel_itr, "getCurrentRef")

        res = _pyapi.to_native_value(PixelTrackerDataJit, ptd).value

        return res

    @type_callable(len)
    def type_cell_pixel_list_len(context):
        def typer(cpxlst):
            if isinstance(cpxlst, CellPixelListType):
                return numba.typing.templates.signature(int64, CellPixelListJit)

        return typer

    @lower_builtin(len, CellPixelListJit)
    def cell_pixel_list_len(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)

        cpxlst, = args
        cpxlst_typ, = sig.args
        pixel_set = cgutils.create_struct_proxy(cpxlst_typ)(context, builder, value=cpxlst).pixel_set
        size_obj = _pyapi.call_method(pixel_set, "size")  # Returns size_t
        res = _pyapi.to_native_value(int64, _pyapi.number_long(size_obj)).value
        return res

    # ----------------------- Numba-compatible CellPixelListIterator ---------------------

    # Make Numba type class
    class CellPixelIterType(types.IteratorType):
        def __init__(self):
            super().__init__(name="CellPixelIterator")

        @property
        def yield_type(self):
            return PixelTrackerDataJit

        @property
        def iterator_type(self):
            return self

    CellPixelIterJit = CellPixelIterType()

    # Register typing
    @typeof_impl.register(cc3d_iter.CellPixelIterator)
    def typeof_cell_pixel_iter(val, c):
        return CellPixelIterJit

    # Register constructor typing

    @type_callable(cc3d_iter.CellPixelIterator)
    def type_cell_pixel_iter(context):
        def typer(cpxlst):
            if isinstance(cpxlst, CellPixelListType):
                return typer

        return typer

    # Register models
    @register_model(CellPixelIterType)
    class CellPixelIterModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ('pixel_tracker_plugin', types.pyobject),
                ('cell', types.pyobject),  # Keep this as pyobject, since it's only stored for boxing/unboxing
                ('pixel_itr', types.pyobject)
            ]
            super().__init__(dmm, fe_type, members)

    # Generate attribute inferencing
    @infer_getattr
    class _CellPixelIterAttribute(AttributeTemplate):
        key = CellPixelIterType

        def resolve_yield_type(self, cpx_itr):
            return PixelTrackerDataJit

        def resolve_iterator_type(self, cpx_itr):
            return CellPixelIterJit

    # Lower constructor as method getiter of class CellPixelListType
    @lower_builtin('getiter', CellPixelListType)
    def impl_cpxlst_iter(context, builder, sig, args):
        _pyapi = _get_numba_pyapi(context, builder)

        # cpxlst, = args
        _model = cgutils.create_struct_proxy(CellPixelIterJit)(context, builder)

        # Get pixelTrackerPlugin from cell pixel list

        cpxlst_model = cgutils.create_struct_proxy(CellPixelListJit)(context, builder, value=args[0])
        _model.pixel_tracker_plugin = cpxlst_model.pixel_tracker_plugin
        _model.cell = cpxlst_model.cell

        # Instantiate and initialize iterator
        pixel_itr_class = _pyapi.unserialize(_pyapi.serialize_object(CompuCell.pixelSetPyItr))
        pixel_itr = _pyapi.call_function_objargs(pixel_itr_class, (cpxlst_model.pixel_set,))
        _pyapi.call_method(pixel_itr, "setToBegin")
        _model.pixel_itr = pixel_itr

        return _model._getvalue()

    # Register boxing/unboxing routines
    @unbox(CellPixelIterType)
    def unbox_cxplst_iter(typ, obj, c):
        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        _model.pixel_tracker_plugin = c.pyapi.object_getattr_string(obj, "pixelTrackerPlugin")
        _model.cell = c.pyapi.object_getattr_string(obj, "cell")
        _model.pixel_itr = c.pyapi.object_getattr_string(obj, "pixelItr")
        is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
        return NativeValue(_model._getvalue(), is_error=is_error)

    @box(CellPixelIterType)
    def box_cxplst_iter(typ, val, c):
        _pyapi: PythonAPI = c.pyapi

        _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

        cpxlst_class = _pyapi.unserialize(c.pyapi.serialize(cc3d_iter.CellPixelList))
        cpxlst = _pyapi.call_function_objargs(cpxlst_class, (_model.pixel_tracker_plugin, _model.cell))
        res = _pyapi.call_method(cpxlst, "__iter__")
        _pyapi.object_setattr_string(res, "pixelItr", _model.pixel_itr)

        _pyapi.decref(cpxlst_class)
        _pyapi.decref(cpxlst)
        return res

    # Implementation of cell pixel list iterator
    @iterator_impl(CellPixelListType, CellPixelIterType)
    class ImplCellPixelIteratorTypr:
        def __init__(self, context, builder, cpxlst_iter):
            self.context = context
            self.builder = builder
            self.pyapi = _get_numba_pyapi(context, builder)
            cpxlst_iter_model = cgutils.create_struct_proxy(CellPixelIterJit)(context, builder, value=cpxlst_iter)
            self.pixel_tracker_plugin = cpxlst_iter_model.pixel_tracker_plugin
            self.pixel_itr = cpxlst_iter_model.pixel_itr
            self.ir_one_typ = ir.IntType(1)
            self.ir_one = ir.Constant(self.ir_one_typ, 1)

            pixel_itr_class = self.pyapi.unserialize(self.pyapi.serialize_object(CompuCell.pixelSetPyItr))
            self.isEnd_meth = self.pyapi.object_getattr_string(pixel_itr_class, "isEnd")
            self.next_meth = self.pyapi.object_getattr_string(pixel_itr_class, "next")
            self.getCurrentRef_meth = self.pyapi.object_getattr_string(pixel_itr_class, "getCurrentRef")

        @property
        def is_not_end(self):
            ret_is_end = self.pyapi.call_function_objargs(self.isEnd_meth, (self.pixel_itr,))
            res = cgutils.as_bool_bit(self.builder, self.pyapi.object_not(ret_is_end))

            self.pyapi.decref(ret_is_end)
            return res

        @property
        def current_ref(self):
            return self.pyapi.call_function_objargs(self.getCurrentRef_meth, (self.pixel_itr,))

        def _yield(self):
            ptd_obj = self.pyapi.call_method(self.pixel_tracker_plugin, "getPixelTrackerData", (self.current_ref,))
            return self.pyapi.to_native_value(PixelTrackerDataJit, ptd_obj).value

        def _iter_next(self):
            self.pyapi.call_function_objargs(self.next_meth, (self.pixel_itr,))  # Returns null

        def iternext(self, context, builder, result):
            result.yield_(self._yield())
            result.set_valid(self.is_not_end)
            self._iter_next()

    return CellPixelListJit, CellPixelIterJit


CellPixelListJit, CellPixelIterJit = _ctor_cell_pixel_list()


class _CC3DJitUtilsBuilder:
    """
    Builds Numba support for CC3DJitUtils, a class that mimics select steppable features in a Jit-compatible way
    """
    def __init__(self):
        self.__initialized = False
        self.__data_class = None
        self.__util_class = None
        self.__data_members = [
            ("cell_list", CellListJit, types.pyobject)
        ]
        self.__methods = [
            ("get_cell_pixel_list", types.pyfunc_type, (CellPixelListJit, CellGJit))
        ]

    def get_classes(self, members):
        if not self.__initialized:
            self.__build_classes(members)

        return self.__data_class, self.__util_class

    def __build_classes(self, members):
        if self.__initialized:
            return

        class CC3DJitUtilsClass(object):
            def __init__(self, _data_obj):
                self.data_obj = _data_obj

        # User-facing convenience class
        class CC3DJitUtilsType(types.Type):
            def __init__(self, _data_obj):
                self.data_obj = _data_obj

                super().__init__(name="CC3DJitUtils")

        # Static storage class for simulation accessor references
        # All instances of CC3DJitUtilsType reference an instance of this class
        class CC3DJitUtilsDataClass(object):
            def __init__(self, _members):
                [setattr(self, _name, _inst) for _name, _inst in _members]

        class CC3DJitUtilsDataType(types.Type):
            def __init__(self, _members):
                [setattr(self, _name, _inst) for _name, _inst in _members]
                super().__init__(name="CC3DJitUtilsData")

        cc3d_jit_utils_data = CC3DJitUtilsDataType(members)
        cc3d_jit_utils = CC3DJitUtilsType(cc3d_jit_utils_data)

        @typeof_impl.register(CC3DJitUtilsClass)
        def typeof_cc3d_utils(val, c):
            return cc3d_jit_utils

        @typeof_impl.register(CC3DJitUtilsDataClass)
        def typeof_cc3d_utils_data(val, c):
            return cc3d_jit_utils_data

        @type_callable(CC3DJitUtilsType)
        def type_cc3d_utils(context):
            def typer(utils_data):
                if isinstance(utils_data, CC3DJitUtilsDataType):
                    return cc3d_jit_utils

            return typer

        @unbox(CC3DJitUtilsType)
        def unbox_utils(typ, obj, c):
            _pyapi: PythonAPI = c.pyapi
            _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
            _model.data_obj = _pyapi.object_getattr_string(obj, "data_obj")
            is_error = cgutils.is_not_null(c.builder, _pyapi.err_occurred())
            return NativeValue(_model._getvalue(), is_error=is_error)

        @box(CC3DJitUtilsType)
        def box_utils(typ, val, c):
            _pyapi: PythonAPI = c.pyapi
            _model = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
            class_obj = _pyapi.unserialize(_pyapi.serialize_object(CC3DJitUtilsClass))
            res = _pyapi.call_function_objargs(class_obj, (_model.data_obj,))
            return res

        _names = [x[0] for x in self.__data_members]
        @unbox(CC3DJitUtilsDataType)
        def unbox_utils_data(typ, obj, c):
            _pyapi: PythonAPI = c.pyapi

            _model = cgutils.create_struct_proxy(typ)(c.context, c.builder)
            [setattr(_model, _name, _pyapi.object_getattr_string(obj, _name)) for _name in _names]

            is_error = cgutils.is_not_null(c.builder, _pyapi.err_occurred())
            return NativeValue(_model._getvalue(), is_error=is_error)

        # Build models

        @register_model(CC3DJitUtilsType)
        class CC3DJitUtilsModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                _members = [
                    ('data_obj', types.pyobject)
                ]
                super().__init__(dmm, fe_type, _members)

        data_members = []
        for name, _, numba_typ in self.__data_members + self.__methods:
            data_members.append((name, numba_typ))

        @register_model(CC3DJitUtilsDataType)
        class CC3DJitUtilsDataModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                super().__init__(dmm, fe_type, data_members)

        # Build inferencing

        class _CC3DJitUtilsAttribute(AttributeTemplate):
            key = CC3DJitUtilsType

        def resolve_factory(name, jit_class):
            def impl(self, u):
                return jit_class

            return "resolve_" + name, impl

        attr_resolves = dict()
        for name, jit_class, _ in self.__data_members:
            resolve_name, resolve_impl = resolve_factory(name, jit_class)
            attr_resolves[resolve_name] = resolve_impl

        infer_getattr(type("CC3DJitUtilsAttribute", (_CC3DJitUtilsAttribute,), attr_resolves))

        # Lower attribute getters

        for name, jit_class, _ in self.__data_members:
            @numba.extending.lower_getattr(CC3DJitUtilsType, name)
            def lower_utils_get_attr(context, builder, typ, value):
                _pyapi = _get_numba_pyapi(context, builder)

                data_obj = cgutils.create_struct_proxy(typ)(context, builder, value=value).data_obj
                attr_obj = _pyapi.object_getattr_string(data_obj, name)
                res = _pyapi.to_native_value(jit_class, attr_obj).value

                # _pyapi.decref(attr_obj)

                return res

        # Define equivalent steppable methods

        #   "SteppableBasePy.get_cell_pixel_list(self, cell)"

        @numba.extending.intrinsic
        def _get_cell_pixel_list(typingctx, utils, cell):
            sig = numba.typing.signature(CellPixelListJit, cc3d_jit_utils, CellGJit)

            def codegen(context, builder, sig, args):
                _pyapi = _get_numba_pyapi(context, builder)

                utils, cell = args
                utils_typ, cell_typ = sig.args

                accessor = cgutils.create_struct_proxy(utils_typ)(context, builder, value=utils)

                data_obj = accessor.data_obj

                pixel_tracker_plugin = _pyapi.object_getattr_string(data_obj, "pixel_tracker_plugin")

                class_obj = _pyapi.unserialize(_pyapi.serialize_object(cc3d_iter.CellPixelList))
                res = _pyapi.call_function_objargs(class_obj, (pixel_tracker_plugin, cell))
                res_val = _pyapi.to_native_value(sig.return_type, res).value

                return res_val

            return sig, codegen

        @overload_method(CC3DJitUtilsType, 'get_cell_pixel_list')
        def om_get_cell_pixel_list(utils, cell):
            if not isinstance(cell, CellGJit.__class__):
                return

            def impl(utils, cell):
                return _get_cell_pixel_list(utils, cell)

            return impl

        self.__initialized = True
        self.__util_class = CC3DJitUtilsClass
        self.__data_class = CC3DJitUtilsDataClass


_utils_builder = _CC3DJitUtilsBuilder()


def init_cc3d_jit(steppable):
    """
    Attaches an instance of CC3DJitUtils to a steppable
    :param steppable: CC3D steppable to which an instance of CC3DJitUtils is attached as attribute "cc3d_jit_utils"
    :return: None
    """
    _cell_list = steppable.cell_list
    _get_cell_pixel_list = steppable.get_cell_pixel_list

    # Attach instance of CC3DJitUtils to steppable

    members = [
        ("cell_list", steppable.cell_list),
        ('pixel_tracker_plugin', steppable.pixelTrackerPlugin),
        ("get_cell_pixel_list", steppable.get_cell_pixel_list)
    ]

    data_class, utils_class = _utils_builder.get_classes(members)
    data_inst = data_class(members)
    steppable.cc3d_jit_utils = utils_class(data_inst)
