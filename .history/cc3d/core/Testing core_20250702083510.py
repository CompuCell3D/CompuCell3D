
"""
Defines python-based core specification classes
"""

# todo: add automatic type checks of spec parameter sets
# todo: add to source code doc autogen
# todo: add to python reference manual

from abc import ABC
from contextlib import AbstractContextManager
import itertools
import os
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import cc3d
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLElement, CC3DXMLListPy, Xml2Obj
from cc3d.cpp.CompuCell import Point3D


_Type = TypeVar('_Type')
"""A generic type"""


class SpecValueError(Exception):
    """ Base class for specification value errors """

    def __init__(self, *args, names: List[str] = None):
        super().__init__(*args)

        self.names = []
        if names is not None:
            self.names.extend(names)


class _SpecValueErrorContext(AbstractContextManager):
    """
    A context manager for aggregating multiple raised SpecValueError exceptions into one exception

    Typical usage is as follows,

    .. code-block:: python

       err_ctx = _SpecValueErrorContext()
       with err_ctx:
          raise SpecValueError("first message", names=["name1"])
       with err_ctx:
          raise SpecValueError("second message", names=["name2"])
       if err_ctx:  # Returns True if any SpecValueError exceptions were raised
          # Here a SpecValueError is raised equivalent to
          # raise SpecValueError("first message\\nsecond message, names=["name1", "name2"])
          err_ctx.raise_error()

    Note that :meth:`__call__` defines functionality to wrap a function for usage as follows,

    .. code-block:: python

       err_ctx = _SpecValueErrorContext()
       err_ctx(func1, 1, x=2)  # For def func1(_z, x)
       err_ctx(func2, 3, y=4)  # For def func2(_z, y)
       if err_ctx:  # Test for any raised SpecValueError exceptions by func1 and func2
          err_ctx.raise_error()  # Raise aggregated SpecValueError exception

    """
    def __init__(self):
        super().__init__()

        self._errs: List[SpecValueError] = []

    def __call__(self, func, *args, **kwargs):
        with self:
            return func(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and issubclass(exc_type, SpecValueError):
            exc_value: SpecValueError
            self._errs.append(exc_value)
            return True

    def __bool__(self):
        return len(self._errs) > 0

    def raise_error(self):
        """
        Raises all collected exceptions and aggregates messages and names into one exception

        :raises SpecValueError: when called
        :return: None
        """
        names = []
        msgs = "\n".join([str(e) for e in self._errs])
        [names.extend(e.names) for e in self._errs]
        raise SpecValueError(msgs, names=names)


class _SpecValueErrorContextBlock(AbstractContextManager):
    """
    A convenience context manager for aggregating multiple raised SpecValueError exceptions into one exception

    Typical usage is as follows,

    .. code-block:: python

       with _SpecValueErrorContextBlock() as err_ctx:
          with err_ctx.ctx:
             raise SpecValueError("first message", names=["name1"])
          with err_ctx.ctx:
             raise SpecValueError("second message", names=["name2"])
       # Here a SpecValueError is raised equivalent to
       # raise SpecValueError("first message\\nsecond message, names=["name1", "name2"])

    """
    def __init__(self):
        super().__init__()

        self.ctx = _SpecValueErrorContext()

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return self.ctx.__exit__(exc_type, exc_val, exc_tb)

        if self.ctx:
            self.ctx.raise_error()


class SpecValueCheckError(SpecValueError):
    """ Execption for when value check fails """


class SpecValueReadOnlyError(SpecValueError):
    """ Exception for when attempting to set a read-only SpecProperty """


class SpecImportError(Exception):
    """ Base class for specification import errors """


class SteerableError(Exception):
    """ Base class for steerable spec errors """


class SpecProperty(property, Generic[_Type]):
    """ Derived class to enforce checks on sets """

    def __init__(self, name: str, readonly: bool = False):

        def _fget(_self) -> _Type:
            """
            Item getter in :attr:`spec_dict` for :class:`SpecProperty` instances

            :param _self: parent
            :type _self: _PyCoreSpecsBase
            :return: value
            """
            return _self.spec_dict[name]

        def _fset(_self, val: _Type) -> None:
            """
            Item setter in :attr:`spec_dict` for :class:`SpecProperty` instances

            :param _self: parent
            :type _self: _PyCoreSpecsBase
            :param val: value
            :raises SpecValueCheckError: when check_dict criterion is True, if any
            :return: None
            """
            try:
                fcn, msg = _self.check_dict[name]
                if fcn(val):
                    raise SpecValueCheckError(msg, names=[name])
            except KeyError:
                pass
            _self.spec_dict[name] = val

        def _fset_err(_self, val: _Type) -> None:
            """
            Read-only item setter in :attr:`spec_dict` for :class:`SpecProperty` instances

            :param _self: parent
            :type _self: _PyCoreSpecsBase
            :param val: value
            :raises SpecValueReadOnlyError: when setting
            :return: None
            """
            raise SpecValueReadOnlyError(f"Setting attribute is illegal.", names=[name])

        if readonly:
            fset = _fset_err
        else:
            fset = _fset

        super().__init__(fget=_fget, fset=fset)


Point3DLike = Union[Point3D, List[int], Tuple[int, int, int]]


def _as_point3d(pt: Point3DLike):
    if isinstance(pt, list) or isinstance(pt, tuple):
        pt = Point3D(*pt)
    return pt


class _PyCoreSpecsBase:
    """
    Base class of all core specification implementations
    """

    name = ""
    """user-facing name"""

    type = ""
    """core type *e.g.*, Steppable"""

    registered_name = ""
    """name according to core"""

    core_accessor_name = ""
    """core accessor name"""

    check_dict: Dict[str, Tuple[Callable[[Any], bool], str]] = {}
    """
    input checks dictionary for spec properties
    
    key corresponds to a key in instance dictionary attribute :attr:`spec_dict`
    
    value is a tuple, where
       - the first element is a function that takes an input and raise a SpecValueError for an invalid input
       - the second element is a message string to accompany the raised SpecValueError
    
    inputs without a check are not validated, and can be omitted
    
    As a simple example, to enforce positive values for :attr:`steps`, write the following
    
    .. code-block:: python
       
       check_dict = {"steps": (lambda x: x < 1, "Steps must be positive")}

    """

    def __init__(self, *args, **kwargs):
        self.spec_dict: Dict[str, Any] = {}
        """specification dictionary"""

        self._el = None
        """(:class:`ElementCC3D` or None), for keeping a reference alive"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generates base element according to name and type

        :raises SpecValueError: spec type not recognized
        :return: Base element
        """
        if not self.name or not self.type:
            return None
        if self.type.lower() == "plugin":
            return ElementCC3D("Plugin", {"Name": self.registered_name})
        elif self.type.lower() == "steppable":
            return ElementCC3D("Steppable", {"Type": self.registered_name})
        raise SpecValueError(f"Spec type {self.type} not recognized", names=["spec_type"])

    def check_inputs(self, **kwargs) -> None:
        """
        Validates inputs against all registered checks

        Checks are defined in implementation :attr:`check_dict`

        :param kwargs: key-value pair by registered input
        :raises SpecValueError: when check criterion is True
        :return: None
        """
        with _SpecValueErrorContextBlock() as err_ctx:

            for k, v in kwargs.items():
                with err_ctx.ctx:
                    try:
                        fcn, msg = self.check_dict[k]
                        if fcn(v):
                            raise SpecValueError(msg, names=[k])
                    except KeyError:
                        pass

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary. Implementations should override this

        :return: CC3DML XML element
        """
        raise NotImplementedError

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of dependencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return [PottsCore, CellTypePlugin]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        # Validate dependencies
        with _SpecValueErrorContextBlock() as err_ctx:
            for s in self.depends_on:
                with err_ctx.ctx:
                    try:
                        CoreSpecsValidator.validate_single_instance(*specs, cls_oi=s, caller_name=self.registered_name)
                    except SpecValueError as err:
                        if hasattr(s, "registered_name"):
                            name = s.registered_name
                        else:
                            name = s.__name__
                        raise SpecValueError(str(err), names=["depends_on", name])

    @property
    def core(self):
        """

        :return: The wrapped core object if accessible, otherwise None
        """
        if not self.core_accessor_name:
            return None
        from cc3d.cpp import CompuCell
        try:
            return getattr(CompuCell, self.core_accessor_name)()
        except AttributeError:
            return None

    @core.setter
    def core(self, _val):
        raise SpecValueReadOnlyError("Core object cannot be set")


class _PyCoreParamAccessor(Generic[_Type]):
    """ Parameter accessor container """

    def __init__(self, _specs_base: _PyCoreSpecsBase, key: str):
        self._specs_base = _specs_base
        self._key = key

    def __getitem__(self, item: str) -> _Type:
        return self._specs_base.spec_dict[self._key][item]

    def __setitem__(self, key, value: _Type):
        self._specs_base.spec_dict[self._key][key] = value


class _PyCoreXMLInterface(_PyCoreSpecsBase, ABC):
    """
    Interface for specs that can import from XML

    Any class that inherits must implement :meth:`from_xml` such that, when passed a :class:`CC3DXMLElement` instance
    that contains a child element describing the data of the derived-class instance, :meth:`from_xml` returns an
    instance of the derived class initialized from the :class:`CC3DXMLElement` instance

    All implementations support serialization
    """

    def __getstate__(self) -> Tuple[str, tuple, dict]:
        return self.xml.getCC3DXMLElementString(), tuple(), dict()

    def __setstate__(self, state):
        xml_str, args, kwargs = state

        xml2_obj_converter = Xml2Obj()
        parent_el = ElementCC3D("dummy").CC3DXMLElement
        el = xml2_obj_converter.ParseString(xml_str)
        parent_el.addChild(el)
        o = self.from_xml_str(parent_el.getCC3DXMLElementString())

        super().__init__(*args, **kwargs)
        self.spec_dict.update(o.spec_dict)

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: _PyCoreSpecsBase
        """
        raise NotImplementedError

    @classmethod
    def from_xml_str(cls, _xml: str):
        """
        Instantiate an instance from a CC3DXMLElement parent string instance

        :param _xml:
        :return: python class instace
        :rtype: _PyCoreSpecsBase
        """
        xml2_obj_converter = Xml2Obj()
        el = xml2_obj_converter.ParseString(_xml)
        return cls.from_xml(el)

    @classmethod
    def find_xml_by_attr(cls,
                         _xml: CC3DXMLElement,
                         attr_name: str = None,
                         registered_name: str = None) -> CC3DXMLElement:
        """
        Returns first found xml element in parent by attribute value

        :param _xml: parent element
        :param attr_name: attribute name, default read from class type
        :param registered_name: registered name, default read from class type
        :raises SpecImportError: When xml element not found
        :return: first found element
        """
        if attr_name is None:
            if cls.type == "Plugin":
                attr_name = "Name"
            elif cls.type == "Steppable":
                attr_name = "Type"
            else:
                raise SpecImportError("Attribute name not known")

        if registered_name is None:
            registered_name = cls.registered_name

        el = None
        el_list = CC3DXMLListPy(_xml.getElements(cls.type))
        for els in el_list:
            els: CC3DXMLElement
            if els.findAttribute(attr_name):
                registered_name_el = els.getAttribute(attr_name)
                if registered_name_el == registered_name:
                    el = els
                    break

        if el is None:
            raise SpecImportError(f"{registered_name} not found")

        return el


class _PyCoreSteerableInterface(_PyCoreSpecsBase, ABC):
    """ Interface for steerable :class:`_PyCoreSpecsBase`-derived classes """

    def steer(self) -> None:
        """
        Perform steering

        :raises SteerableError: when not steerable
        :return: None
        """
        from cc3d.CompuCellSetup import persistent_globals as pg
        steerable = pg.simulator.getSteerableObject(self.registered_name)
        try:
            steerable.update(self.xml.CC3DXMLElement)
        except AttributeError:
            raise SteerableError("Not steerable")


class _PyCorePluginSpecs(_PyCoreXMLInterface, ABC):
    """
    Base class for plugins
    """

    type = "Plugin"

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D(self.type, {"Name": self.registered_name})

    def get_plugin(self):
        """

        :raises SpecValueError: when core plugin is unavailable
        :return: core plugin
        """
        try:
            from cc3d.cpp import CompuCell
            return getattr(CompuCell, f"get{self.registered_name}Plugin")()
        except AttributeError:
            raise SpecValueError("Plugin unavailable")

    @property
    def core_accessor_name(self) -> str:
        """

        :return: core accessor name
        """
        accessor_name = "get" + self.registered_name + self.type
        from cc3d.cpp import CompuCell
        if hasattr(CompuCell, accessor_name):
            return accessor_name
        else:
            return ""


class _PyCoreSteppableSpecs(_PyCoreXMLInterface, ABC):
    """
    Base class for steppables
    """

    type = "Steppable"

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D(self.type, {"Type": self.registered_name})

    def get_steppable(self):
        """

        :raises SpecValueError: when core steppable is unavailable
        :return: core steppable
        """
        try:
            from cc3d.cpp import CompuCell
            return getattr(CompuCell, f"get{self.registered_name}Steppable")()
        except AttributeError:
            raise SpecValueError("Steppable unavailable")

    @property
    def core_accessor_name(self):
        """

        :return: core accessor name
        :rtype: str
        """
        accessor_name = "get" + self.registered_name + self.type
        from cc3d.cpp import CompuCell
        if hasattr(CompuCell, accessor_name):
            return accessor_name
        else:
            return ""


    """ Secretion Data Base Class """

    def __init__(self, *_param_specs):
        """

        :param _param_specs: variable number of SecretionParameters instances, optional
        """
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _ps in _param_specs:
                with err_ctx.ctx:
                    if not isinstance(_ps, SecretionParameters):
                        raise SpecValueError("Only SecretionParameters instances can be passed",
                                             names=[_ps.__name__])

        self.spec_dict = {"param_specs": []}
        [self.params_append(_ps) for _ps in _param_specs]

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("SecretionData")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"]]
        return self._el

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        # Validate parameters
        [ps.validate(*specs) for ps in self.spec_dict["param_specs"]]

    def params_append(self, _ps: SecretionParameters) -> None:
        """
        Append a secretion spec

        :param _ps: secretion spec
        :return: None
        """
        if _ps.contact_type is None:
            for x in self.spec_dict["param_specs"]:
                x: SecretionParameters
                if x.contact_type is None and _ps.cell_type == x.cell_type:
                    raise SpecValueError(f"Duplicate specification for cell type {_ps.cell_type}")
        self.spec_dict["param_specs"].append(_ps)

    def params_remove(self, _cell_type: str, contact_type: str = None):
        """
        Remove a secretion spec

        :param _cell_type: name of cell type
        :param contact_type: name of on-contact dependent cell type, optional
        :return: None
        """
        for _ps in self.spec_dict["param_specs"]:
            _ps: SecretionParameters
            if _ps.cell_type == _cell_type and contact_type == _ps.contact_type:
                self.spec_dict["param_specs"].remove(_ps)
                return
        raise SpecValueError("SecretionParameters not specified")

    def params_new(self,
                   _cell_type: str,
                   _val: float,
                   constant: bool = False,
                   contact_type: str = None) -> SecretionParameters:
        """
        Append and return a new secretion spec

        :param _cell_type: cell type name
        :param _val: value of parameter
        :param constant: flag for constant concentration, optional
        :param contact_type: name of cell type for on-contact dependence, optional
        :return: new secretion spec
        """
        ps = SecretionParameters(_cell_type, _val, constant=constant, contact_type=contact_type)
        self.params_append(ps)
        return ps


PDEBOUNDARYVALUE = "Value"
PDEBOUNDARYFLUX = "Flux"
PDEBOUNDARYPERIODIC = "Periodic"
BOUNDARYTYPESPDE = [PDEBOUNDARYVALUE, PDEBOUNDARYFLUX, PDEBOUNDARYPERIODIC]


class PDEBoundaryConditions(_PyCoreSpecsBase):
    """ PDE Solver Boundary Conditions Spec """

    def __init__(self,
                 x_min_type: str = PDEBOUNDARYVALUE, x_min_val: float = 0.0,
                 x_max_type: str = PDEBOUNDARYVALUE, x_max_val: float = 0.0,
                 y_min_type: str = PDEBOUNDARYVALUE, y_min_val: float = 0.0,
                 y_max_type: str = PDEBOUNDARYVALUE, y_max_val: float = 0.0,
                 z_min_type: str = PDEBOUNDARYVALUE, z_min_val: float = 0.0,
                 z_max_type: str = PDEBOUNDARYVALUE, z_max_val: float = 0.0):
        super().__init__()

        self.check_inputs(x_min_type=x_min_type,
                          x_max_type=x_max_type,
                          y_min_type=y_min_type,
                          y_max_type=y_max_type,
                          z_min_type=z_min_type,
                          z_max_type=z_max_type)

        self.spec_dict = {"x_min_type": x_min_type,
                          "x_min_val": x_min_val,
                          "x_max_type": x_max_type,
                          "x_max_val": x_max_val,
                          "y_min_type": y_min_type,
                          "y_min_val": y_min_val,
                          "y_max_type": y_max_type,
                          "y_max_val": y_max_val,
                          "z_min_type": z_min_type,
                          "z_min_val": z_min_val,
                          "z_max_type": z_max_type,
                          "z_max_val": z_max_val}

        self._xml_type_labels = {PDEBOUNDARYVALUE: "ConstantValue",
                                 PDEBOUNDARYFLUX: "ConstantDerivative",
                                 BOUNDARYTYPESPDE[2]: "Periodic"}

    x_min_val: float = SpecProperty(name="x_min_val")
    """boundary condition value along lower x-orthogonal boundary"""

    x_max_val: float = SpecProperty(name="x_max_val")
    """boundary condition value along upper x-orthogonal boundary"""

    y_min_val: float = SpecProperty(name="y_min_val")
    """boundary condition value along lower y-orthogonal boundary"""

    y_max_val: float = SpecProperty(name="y_max_val")
    """boundary condition value along upper y-orthogonal boundary"""

    z_min_val: float = SpecProperty(name="z_min_val")
    """boundary condition value along lower z-orthogonal boundary"""

    z_max_val: float = SpecProperty(name="z_max_val")
    """boundary condition value along upper z-orthogonal boundary"""

    @property
    def x_min_type(self) -> str:
        """

        :return: boundary condition type along lower x-orthogonal boundary
        """
        return self.spec_dict["x_min_type"]

    @x_min_type.setter
    def x_min_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["x_max_type"]]:
            self.spec_dict["x_max_type"] = _val
        self.spec_dict["x_min_type"] = _val

    @property
    def x_max_type(self) -> str:
        """

        :return: boundary condition type along upper x-orthogonal boundary
        """
        return self.spec_dict["x_max_type"]

    @x_max_type.setter
    def x_max_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["x_min_type"]]:
            self.spec_dict["x_min_type"] = _val
        self.spec_dict["x_max_type"] = _val

    @property
    def y_min_type(self) -> str:
        """

        :return: boundary condition type along lower y-orthogonal boundary
        """
        return self.spec_dict["y_min_type"]

    @y_min_type.setter
    def y_min_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["y_max_type"]]:
            self.spec_dict["y_max_type"] = _val
        self.spec_dict["y_min_type"] = _val

    @property
    def y_max_type(self) -> str:
        """

        :return: boundary condition type along upper y-orthogonal boundary
        """
        return self.spec_dict["y_max_type"]

    @y_max_type.setter
    def y_max_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["y_min_type"]]:
            self.spec_dict["y_min_type"] = _val
        self.spec_dict["y_max_type"] = _val

    @property
    def z_min_type(self) -> str:
        """

        :return: boundary condition type along lower z-orthogonal boundary
        """
        return self.spec_dict["z_min_type"]

    @z_min_type.setter
    def z_min_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["z_max_type"]]:
            self.spec_dict["z_max_type"] = _val
        self.spec_dict["z_min_type"] = _val

    @property
    def z_max_type(self) -> str:
        """

        :return: boundary condition type along upper z-orthogonal boundary
        """
        return self.spec_dict["z_max_type"]

    @z_max_type.setter
    def z_max_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["z_min_type"]]:
            self.spec_dict["z_min_type"] = _val
        self.spec_dict["z_max_type"] = _val

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("BoundaryConditions")

    def _xml_plane(self, name: str) -> ElementCC3D:
        el = ElementCC3D("Plane", {"Axis": name})
        b_types = {"X": (self.x_min_type, self.x_max_type),
                   "Y": (self.y_min_type, self.y_max_type),
                   "Z": (self.z_min_type, self.z_max_type)}[name]
        b_vals = {"X": (self.x_min_val, self.x_max_val),
                  "Y": (self.y_min_val, self.y_max_val),
                  "Z": (self.z_min_val, self.z_max_val)}[name]
        attr_dict = {"PlanePosition": "Min"}
        if b_types[0] != BOUNDARYTYPESPDE[2]:
            attr_dict["Value"] = b_vals[0]
        el.ElementCC3D(self._xml_type_labels[b_types[1]], attr_dict)
        attr_dict = {"PlanePosition": "Max"}
        if b_types[1] != BOUNDARYTYPESPDE[2]:
            attr_dict["Value"] = b_vals[1]
        el.ElementCC3D(self._xml_type_labels[b_types[1]], attr_dict)
        return el

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(self._xml_plane(x)) for x in ["X", "Y", "Z"]]
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return [PottsCore]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """

        with _SpecValueErrorContextBlock() as err_ctx:

            with err_ctx.ctx:
                super().validate(*specs)

            for s in specs:
                # Validate against cell types defined in PottsCore
                # Constant value and flux are compatible with no flux
                # Perdioc is compatible with periodic
                if isinstance(s, PottsCore):
                    for c in ["x", "y", "z"]:
                        with err_ctx.ctx:
                            if getattr(s, f"boundary_{c}") == BOUNDARYTYPESPOTTS[0]:
                                if getattr(self, f"{c}_min_type") not in BOUNDARYTYPESPDE[0:2]:
                                    raise SpecValueError(f"Min-{c} boundary could not be validated",
                                                         names=[f"{c}_min_type"])
                                elif getattr(self, f"{c}_max_type") not in BOUNDARYTYPESPDE[0:2]:
                                    raise SpecValueError(f"Max-{c} boundary could not be validated",
                                                         names=[f"{c}_max_type"])
                            elif getattr(self, f"{c}_min_type") != BOUNDARYTYPESPDE[2]:
                                raise SpecValueError(f"Periodic-{c} boundary could not be validated",
                                                     names=[f"{c}_min_type"])


_DD = TypeVar('_DD')
"""Diffusion data generic"""

_SD = TypeVar('_SD')
"""Secretion data generic"""


class _PDESolverFieldSpecs(_PyCoreSpecsBase, Generic[_DD, _SD]):
    """ PDE Field Spec Base Class """

    def __init__(self,
                 field_name: str,
                 diff_data: _DD,
                 secr_data: _SD):
        super().__init__()

        self.spec_dict = {"field_name": field_name,
                          "diff_data": diff_data(field_name=field_name),
                          "secr_data": secr_data(),
                          "bc_specs": PDEBoundaryConditions()}

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    diff_data: _DD = SpecProperty(name="diff_data")
    """diffusion data"""

    bcs: PDEBoundaryConditions = SpecProperty(name="bc_specs")
    """boundary conditions"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("DiffusionField", {"Name": self.field_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.add_child(self.spec_dict["diff_data"].xml)
        self._el.add_child(self.spec_dict["secr_data"].xml)
        self._el.add_child(self.spec_dict["bc_specs"].xml)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return []

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances
        Does not validate field name or solver existence

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        # Validate constituent data
        self.diff_data.validate(*specs)
        self.spec_dict["secr_data"].validate(*specs)
        self.bcs.validate(*specs)

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> SecretionParameters:
        """
        Specify and return a new secretion data spec

        :param _cell_type: name of cell type
        :param _val: value
        :param kwargs:
        :return: new secretion data spec
        """
        p = self.spec_dict["secr_data"].params_new(_cell_type, _val, **kwargs)
        return p

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        self.spec_dict["secr_data"].param_remove(_cell_type, **kwargs)


class _PDESolverSpecs(_PyCoreSteppableSpecs, _PyCoreSteerableInterface, ABC, Generic[_DD, _SD]):

    _field_spec = _PDESolverFieldSpecs
    _diff_data: _DD = _PDEDiffusionDataSpecs
    _secr_data: _SD = _PDESecretionDataSpecs

    def __init__(self):
        super().__init__()

        self.spec_dict = {"fluc_comp": False,
                          "fields": dict()}

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return [PottsCore]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances
        Does not validate field name or solver existence

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        with _SpecValueErrorContextBlock() as err_ctx:

            with err_ctx.ctx:
                super().validate(*specs)

            # Validate fields
            for field_name in self.field_names:
                with err_ctx.ctx:
                    self.fields[field_name].validate(*specs)

            # Validate field names against self
            for field_name in self.field_names:
                with err_ctx.ctx:
                    f = self.fields[field_name]
                    if f.field_name not in self.field_names:
                        raise SpecValueError("Field name could not be validated: " + f.field_name)

            # Validate field names
            with err_ctx.ctx:
                CoreSpecsValidator.validate_field_names(*specs, field_names=self.field_names)

            for f in self.field_names:
                with err_ctx.ctx:
                    CoreSpecsValidator.validate_field_name_unique(*specs, field_name=f)

    @property
    def fields(self) -> _PyCoreParamAccessor[_PDESolverFieldSpecs[_DD, _SD]]:
        """

        :return: accessor to field parameters with field names as keys
        """
        return _PyCoreParamAccessor(self, "fields")

    @property
    def field_names(self) -> List[str]:
        """

        :return: list of registered field names
        """
        return [x for x in self.spec_dict["fields"].keys()]

    def field_new(self, _field_name: str):
        """
        Append and return a new field spec

        :param _field_name: name of field
        :return: new field spec
        """
        if _field_name in self.field_names:
            raise SpecValueError(f"Field with name {_field_name} already specified")
        f: _PDESolverFieldSpecs[_DD, _SD] = self._field_spec(field_name=_field_name,
                                                             diff_data=self._diff_data,
                                                             secr_data=self._secr_data)
        self.spec_dict["fields"][_field_name] = f
        return f

    def field_remove(self, _field_name: str) -> None:
        """
        Remove a field spec

        :param _field_name: name of field
        :return: None
        """
        if _field_name not in self.field_names:
            raise SpecValueError(f"Field with name {_field_name} not specified")
        self.spec_dict["fields"].pop(_field_name)


class PyCoreSpecsRoot:
    """
    Root simulation specs
    """
    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        return ElementCC3D("CompuCell3D", {"Version": cc3d.__version__,
                                           "Revision": cc3d.__revision__})

    @staticmethod
    def get_simulator() -> cc3d.cpp.CompuCell.Simulator:
        """

        :return: simulator
        """
        from cc3d.CompuCellSetup import persistent_globals
        return persistent_globals.simulator


class Metadata(_PyCoreXMLInterface):
    """
    Metadata Specs
    """

    name = "metadata"
    registered_name = "Metadata"

    check_dict = {
        "num_processors": (lambda x: x < 0, "Number of processors must be non-negative"),
        "debug_output_frequency": (lambda x: x < 0, "Debug output frequency must be non-negative")
    }

    def __init__(self,
                 num_processors: int = 1,
                 debug_output_frequency: int = 0):
        """

        :param num_processors: number of processors, defaults to 1
        :param debug_output_frequency: debug output frequency, defaults to 0
        """
        super().__init__()

        self.check_inputs(num_processors=num_processors,
                          debug_output_frequency=debug_output_frequency)

        self.spec_dict = {"num_processors": num_processors,
                          "debug_output_frequency": debug_output_frequency}

    num_processors: int = SpecProperty("num_processors")
    """Number of processors"""

    debug_output_frequency: int = SpecProperty("debug_output_frequency")
    """Debug output frequency"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generates base element according to name and type

        :raises SpecValueError: spec type not recognized
        :return: Base element
        """
        return ElementCC3D(self.registered_name)

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary. Implementations should override this

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.add_child(ElementCC3D("NumberOfProcessors", {}, str(self.num_processors)))
        self._el.add_child(ElementCC3D("DebugOutputFrequency", {}, str(self.debug_output_frequency)))
        return self._el

    @classmethod
    def find_xml_by_attr(cls,
                         _xml: CC3DXMLElement,
                         attr_name: str = None,
                         registered_name: str = None) -> CC3DXMLElement:
        """
        Returns first found xml element in parent by attribute value

        :param _xml: parent element
        :param attr_name: attribute name, default read from class type
        :param registered_name: registered name, default read from class type
        :raises SpecImportError: When xml element not found
        :return: first found element
        """
        if not _xml.findElement(cls.registered_name):
            raise SpecImportError(f"{cls.registered_name} not found")
        return _xml.getFirstElement(cls.registered_name)

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: _PyCoreSpecsBase
        """
        el = cls.find_xml_by_attr(_xml)
        o = cls()
        if el.findElement("NumberOfProcessors"):
            o.num_processors = el.getFirstElement("NumberOfProcessors").getUInt()
        if el.findElement("DebugOutputFrequency"):
            o.debug_output_frequency = el.getFirstElement("DebugOutputFrequency").getUInt()
        return o


LATTICETYPECAR = "Cartesian"
LATTICETYPEHEX = "Hexagonal"
LATTICETYPES = [LATTICETYPECAR, LATTICETYPEHEX]
"""Supported lattice types"""

POTTSBOUNDARYNOFLUX = "NoFlux"
POTTSBOUNDARYPERIODIC = "Periodic"

BOUNDARYTYPESPOTTS = [POTTSBOUNDARYNOFLUX, POTTSBOUNDARYPERIODIC]
"""Supported boundary types"""

FLUCAMPFCNMIN = "Min"
FLUCAMPFCNMAX = "Max"
FLUCAMPFCNAVG = "ArithmeticAverage"
FLUCAMPFCNS = [FLUCAMPFCNMIN, FLUCAMPFCNMAX, FLUCAMPFCNAVG]
"""Supported fluctuation amplitude function names"""


class PottsCore(_PyCoreSteerableInterface, _PyCoreXMLInterface):
    """
    Potts specs
    """

    name = "potts"
    registered_name = "Potts"

    check_dict = {
        "dim_x": (lambda x: x < 1, "Dimension must be positive"),
        "dim_y": (lambda x: x < 1, "Dimension must be positive"),
        "dim_z": (lambda x: x < 1, "Dimension must be positive"),
        "steps": (lambda x: x < 0, "Steps must be non-negative"),
        "anneal": (lambda x: x < 0, "Anneal steps must be non-negative"),
        "fluctuation_amplitude_function": (lambda x: x not in FLUCAMPFCNS,
                                           f"Invalid function. Valid functions are {FLUCAMPFCNS}"),
        "boundary_x": (lambda x: x not in BOUNDARYTYPESPOTTS,
                       f"Invalid boundary specification. Valid boundaries are {BOUNDARYTYPESPOTTS}"),
        "boundary_y": (lambda x: x not in BOUNDARYTYPESPOTTS,
                       f"Invalid boundary specification. Valid boundaries are {BOUNDARYTYPESPOTTS}"),
        "boundary_z": (lambda x: x not in BOUNDARYTYPESPOTTS,
                       f"Invalid boundary specification. Valid boundaries are {BOUNDARYTYPESPOTTS}"),
        "neighbor_order": (lambda x: x < 1, "Neighbor order must be positive"),
        "random_seed": (lambda x: x is not None and x < 0, "Invalid random seed. Must be non-negative"),
        "lattice_type": (lambda x: x not in LATTICETYPES,
                         f"Invalid lattice type. Valid types are {LATTICETYPES}")
    }

    def __init__(self,
                 dim_x: int = 1,
                 dim_y: int = 1,
                 dim_z: int = 1,
                 steps: int = 0,
                 anneal: int = 0,
                 fluctuation_amplitude: float = 10.0,
                 fluctuation_amplitude_function: str = FLUCAMPFCNMIN,
                 boundary_x: str = POTTSBOUNDARYNOFLUX,
                 boundary_y: str = POTTSBOUNDARYNOFLUX,
                 boundary_z: str = POTTSBOUNDARYNOFLUX,
                 neighbor_order: int = 1,
                 random_seed: int = None,
                 lattice_type: str = LATTICETYPECAR,
                 offset: float = 0):
        """

        :param dim_x: x-dimension of simulation domain, defaults to 1
        :param dim_y: y-dimension of simulation domain, defaults to 1
        :param dim_z: z-dimension of simulation domain, defaults to 1
        :param steps: number of simulation steps, defaults to 0
        :param anneal: number of annealing steps, defaults to 0
        :param fluctuation_amplitude: constant fluctuation amplitude, defaults to 10
        :param fluctuation_amplitude_function:
            fluctuation amplitude function for heterotypic fluctuation amplitudes, defaults to "Min"
        :param boundary_x: boundary conditions orthogonal to x-direction, defaults to "NoFlux"
        :param boundary_y: boundary conditions orthogonal to y-direction, defaults to "NoFlux"
        :param boundary_z: boundary conditions orthogonal to z-direction, defaults to "NoFlux"
        :param neighbor_order: neighbor order of flip attempts, defaults to 1
        :param random_seed: random seed, optional
        :param lattice_type: type of lattice, defaults to "Cartesian"
        :param offset: offset in Boltzmann acceptance function
        """

        super().__init__()

        self.check_inputs(dim_x=dim_x,
                          dim_y=dim_y,
                          dim_z=dim_z,
                          steps=steps,
                          anneal=anneal,
                          fluctuation_amplitude_function=fluctuation_amplitude_function,
                          boundary_x=boundary_x,
                          boundary_y=boundary_y,
                          boundary_z=boundary_z,
                          neighbor_order=neighbor_order,
                          random_seed=random_seed,
                          lattice_type=lattice_type)

        self.spec_dict = {"dim_x": dim_x,
                          "dim_y": dim_y,
                          "dim_z": dim_z,
                          "steps": steps,
                          "anneal": anneal,
                          "fluctuation_amplitude": fluctuation_amplitude,
                          "fluctuation_amplitude_function": fluctuation_amplitude_function,
                          "boundary_x": boundary_x,
                          "boundary_y": boundary_y,
                          "boundary_z": boundary_z,
                          "neighbor_order": neighbor_order,
                          "debug_output_frequency": 0,  # Legacy support; standard is to use Metadata
                          "random_seed": random_seed,
                          "lattice_type": lattice_type,
                          "offset": offset,
                          "energy_function_calculator": None}

    dim_x: int = SpecProperty(name="dim_x")
    """x-dimension of simulation domain"""

    dim_y: int = SpecProperty(name="dim_y")
    """y-dimension of simulation domain"""

    dim_z: int = SpecProperty(name="dim_z")
    """z-dimension of simulation domain"""

    steps: int = SpecProperty(name="steps")
    """number of simulation steps"""

    anneal: int = SpecProperty(name="anneal")
    """number of annealing steps"""

    fluctuation_amplitude_function: str = SpecProperty(name="fluctuation_amplitude_function")
    """fluctuation amplitude function for heterotypic fluctuation amplitudes"""

    boundary_x: str = SpecProperty(name="boundary_x")
    """boundary conditions orthogonal to x-direction"""

    boundary_y: str = SpecProperty(name="boundary_y")
    """boundary conditions orthogonal to y-direction"""

    boundary_z: str = SpecProperty(name="boundary_z")
    """boundary conditions orthogonal to z-direction"""

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order of flip attempts"""

    random_seed: Optional[int] = SpecProperty(name="random_seed")
    """random seed"""

    lattice_type: str = SpecProperty(name="lattice_type")
    """type of lattice"""

    offset: float = SpecProperty(name="offset")
    """offset in Boltzmann acceptance function"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D(self.registered_name)

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("Dimensions", {"x": str(self.dim_x), "y": str(self.dim_y), "z": str(self.dim_z)})

        self._el.ElementCC3D("Steps", {}, str(self.steps))

        if self.anneal > 0:
            self._el.ElementCC3D("Anneal", {}, str(self.anneal))

        if isinstance(self.spec_dict["fluctuation_amplitude"], dict):
            fa_el = self._el.ElementCC3D("FluctuationAmplitude")
            for _type, _fa in self.spec_dict["fluctuation_amplitude"].items():
                fa_el.ElementCC3D("FluctuationAmplitudeParameters", {"CellType": _type,
                                                                     "FluctuationAmplitude": str(_fa)})
        else:
            self._el.ElementCC3D("FluctuationAmplitude", {}, str(self.spec_dict["fluctuation_amplitude"]))

        if self.fluctuation_amplitude_function != FLUCAMPFCNS[0]:
            self._el.ElementCC3D("FluctuationAmplitudeFunctionName",
                                 {},
                                 self.fluctuation_amplitude_function)

        for c in ["x", "y", "z"]:
            if getattr(self, f"boundary_{c}") != BOUNDARYTYPESPOTTS[0]:
                self._el.ElementCC3D(f"Boundary_{c}", {}, getattr(self, f"boundary_{c}"))

        if self.neighbor_order > 1:
            self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))

        # Legacy support; standard is to use Metadata
        if self.spec_dict["debug_output_frequency"] > 0:
            self._el.ElementCC3D("DebugOutputFrequency", {}, str(self.spec_dict["debug_output_frequency"]))

        if self.random_seed is not None:
            self._el.ElementCC3D("RandomSeed", {}, str(self.random_seed))

        if self.lattice_type != LATTICETYPES[0]:
            self._el.ElementCC3D("LatticeType", {}, self.lattice_type)

        if self.offset != 0:
            self._el.ElementCC3D("Offset", {}, str(self.offset))

        if self.spec_dict["energy_function_calculator"] is not None:
            efc_dict = self.spec_dict["energy_function_calculator"]

            if isinstance(efc_dict, dict) and efc_dict["type"] == "Statistics":

                efc_el = self._el.ElementCC3D("EnergyFunctionCalculator", {"Type": "Statistics"})
                efc_el.ElementCC3D("OutputFileName",
                                   {"Frequency": str(efc_dict["output_file_name"]["frequency"])},
                                   str(efc_dict["output_file_name"]["file_name"]))

                if efc_dict["output_core_file_name_spin_flips"] is not None:
                    esf_dict = efc_dict["output_core_file_name_spin_flips"]
                    attr_dict = {"Frequency": str(esf_dict["frequency"])}
                    if esf_dict["gather_results"]:
                        attr_dict["GatherResults"] = ""
                    if esf_dict["output_accepted"]:
                        attr_dict["OutputAccepted"] = ""
                    if esf_dict["output_rejected"]:
                        attr_dict["OutputRejected"] = ""
                    if esf_dict["output_total"]:
                        attr_dict["OutputTotal"] = ""
                    efc_el.ElementCC3D("OutputCoreFileNameSpinFlips", efc_dict)

        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return [CellTypePlugin]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                if isinstance(self.spec_dict["fluctuation_amplitude"], dict):
                    CoreSpecsValidator.validate_cell_type_names(
                        type_names=self.spec_dict["fluctuation_amplitude"].keys(), cell_type_spec=s)

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: PottsCore
        """
        el: CC3DXMLElement = _xml.getFirstElement("Potts")

        o = cls()

        if el.findElement("Dimensions"):
            dim_el: CC3DXMLElement = el.getFirstElement("Dimensions")
            [setattr(o, "dim_" + c, dim_el.getAttributeAsUInt(c)) for c in ["x", "y", "z"] if dim_el.findAttribute(c)]

        if el.findElement("Steps"):
            o.steps = el.getFirstElement("Steps").getUInt()

        if el.findElement("Anneal"):
            o.anneal = el.getFirstElement("Anneal").getUInt()

        if el.findElement("FluctuationAmplitude"):
            amp_el: CC3DXMLElement = el.getFirstElement("FluctuationAmplitude")
            if amp_el.findElement("FluctuationAmplitudeParameters"):
                amp_el_list = CC3DXMLListPy(amp_el.getElements("FluctuationAmplitudeParameters"))
                for ampt_el in amp_el_list:
                    ampt_el: CC3DXMLElement
                    o.fluctuation_amplitude_type(ampt_el.getAttribute("CellType"),
                                                 ampt_el.getAttributeAsDouble("FluctuationAmplitude"))
            else:
                o.fluctuation_amplitude_const(amp_el.getDouble())

        if el.findElement("FluctuationAmplitudeFunctionName"):
            o.fluctuation_amplitude_function = el.getFirstElement("FluctuationAmplitudeFunctionName").getText()

        for c in ["x", "y", "z"]:
            if el.findElement(f"Boundary_{c}"):
                setattr(o, f"boundary_{c}", el.getFirstElement(f"Boundary_{c}").getText())

        if el.findElement("NeighborOrder"):
            o.neighbor_order = el.getFirstElement("NeighborOrder").getUInt()

        if el.findElement("DebugOutputFrequency"):
            # Legacy support; standard is to use Metadata
            o.spec_dict["debug_output_frequency"] = el.getFirstElement("DebugOutputFrequency").getUInt()

        if el.findElement("RandomSeed"):
            o.random_seed = el.getFirstElement("RandomSeed").getUInt()

        if el.findElement("LatticeType"):
            o.lattice_type = el.getFirstElement("LatticeType").getText()

        if el.findElement("Offset"):
            o.offset = el.getFirstElement("Offset").getDouble()

        if el.findElement("EnergyFunctionCalculator"):
            func_el: CC3DXMLElement = el.getFirstElement("EnergyFunctionCalculator")

            if func_el.getAttribute("Type") == "Statistics":

                o_el: CC3DXMLElement = func_el.getFirstElement("OutputFileName")

                kwargs = {}
                if func_el.findElement("OutputCoreFileNameSpinFlips"):
                    c_el: CC3DXMLElement = func_el.getFirstElement("OutputCoreFileNameSpinFlips")
                    if c_el.findAttribute("Frequency"):
                        kwargs['flip_frequency'] = c_el.getAttributeAsUInt("Frequency")
                        kwargs['gather_results'] = el.findAttribute("GatherResults")
                        kwargs['output_accepted'] = el.findAttribute("OutputAccepted")
                        kwargs['output_rejected'] = el.findAttribute("OutputRejected")
                        kwargs['output_total'] = el.findAttribute("OutputTotal")

                o.energy_function_calculator_stat(output_filename=o_el.getText(),
                                                  frequency=o_el.getAttributeAsUInt("Frequency"),
                                                  **kwargs)

        return o

    @property
    def core(self):
        """

        :return: The wrapped core object if accessible, otherwise None
        """
        return self.get_potts()

    def fluctuation_amplitude_const(self, _fa: float) -> None:
        """
        Set fluctuation amplitude for all cell types

        :param _fa: fluctuation amplitude
        :return: None
        """
        self.spec_dict["fluctuation_amplitude"] = float(_fa)

    def fluctuation_amplitude_type(self, _type: str, _fa: float) -> None:
        """
        Set fluctuation amplitude by cell type

        :param _type: cell type
        :param _fa: fluctuation amplitude
        :return: None
        """
        if not isinstance(self.spec_dict["fluctuation_amplitude"], dict):
            self.spec_dict["fluctuation_amplitude"] = {}
        self.spec_dict["fluctuation_amplitude"][_type] = float(_fa)

    def energy_function_calculator_stat(self, output_filename: str,
                                        frequency: int,
                                        flip_frequency: int = None,
                                        gather_results: bool = False,
                                        output_accepted: bool = False,
                                        output_rejected: bool = False,
                                        output_total: bool = False) -> None:
        """
        Specify energy function calculator of type Statistics

        :param output_filename:
            name of the file to which CC3D will write average changes in energies returned by each plugin and
            corresponding standard deviations for those steps whose values are divisible by :attr:`frequency`.
        :param frequency: frequency at which to output calculations
        :param flip_frequency: frequency at which to output flip attempt data, optional
        :param gather_results: ensures one file written, defaults to False
        :param output_accepted: write data on accepted flip attempts, defaults to False
        :param output_rejected: write data on rejected flip attempts, defaults to False
        :param output_total: write data on all flip attempts, defaults to False
        :return: None
        """
        if frequency < 1:
            raise ValueError("Invalid frequency. Must be positive")
        if flip_frequency is not None and flip_frequency < 1:
            raise ValueError("Invalid flip frequency. Must be positive if specified")

        output_dict = {"frequency": frequency,
                       "file_name": output_filename}
        flip_dict = None
        if flip_frequency is not None:
            flip_dict = {"frequency": flip_frequency,
                         "gather_results": gather_results,
                         "output_accepted": output_accepted,
                         "output_rejected": output_rejected,
                         "output_total": output_total}

        self.spec_dict["energy_function_calculator"] = {
            "type": "Statistics",
            "output_file_name": output_dict,
            "output_core_file_name_spin_flips": flip_dict
        }

    @staticmethod
    def get_potts() -> cc3d.cpp.CompuCell.Potts3D:
        """

        :raises SpecValueError: when Potts is unavailable
        :return: Potts instance
        """
        from cc3d.CompuCellSetup import persistent_globals
        try:
            return persistent_globals.simulator.getPotts()
        except AttributeError:
            raise SpecValueError("Potts unavailable")


class CellTypePlugin(_PyCorePluginSpecs):
    """
    CellType Plugin specs

    Cell type ids are automatically defined unless explicitly passed.
    """

    name = "cell_type"
    registered_name = "CellType"

    def __init__(self, *_cell_types):
        """

        :param _cell_types: variable number of cell type names
        """
        super().__init__()

        self.spec_dict = {"cell_types": [("Medium", 0, False)]}

        [self.cell_type_append(_ct) for _ct in _cell_types]

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()

        for type_name, type_id, frozen in self.spec_dict["cell_types"]:
            td = {"TypeName": type_name, "TypeId": str(type_id)}
            if frozen:
                td["Freeze"] = ""
            self._el.ElementCC3D("CellType", td)

        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: CellTypePlugin
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("CellType"))

        for tp_el in el_list:
            kwargs = {"frozen": tp_el.findAttribute("Freeze")}
            if tp_el.findAttribute("TypeId"):
                kwargs["type_id"] = tp_el.getAttributeAsInt("TypeId")
            o.cell_type_append(tp_el.getAttribute("TypeName"), **kwargs)
        return o

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return [x[0] for x in self.spec_dict["cell_types"]]

    @property
    def cell_type_ids(self) -> List[int]:
        """

        :return: list of cell type ids
        """
        return [x[1] for x in self.spec_dict["cell_types"]]

    def is_frozen(self, _name: str) -> Optional[bool]:
        """

        :param _name: name of cell type
        :return: True if frozen, False if not frozen, None if cell type not found
        """
        for x in self.spec_dict["cell_types"]:
            if x[0] == _name:
                return x[2]
        return None

    def cell_type_append(self, _name: str, type_id: int = None, frozen: bool = False) -> None:
        """
        Add a cell type

        :param _name: name of cell type
        :param type_id: id of cell type, optional
        :param frozen: freeze cell type if True, defaults to False
        :raises SpecValueError:
        when name or id is already specified, or when setting the id of the medium to a nonzero value
        :return: None
        """

        if _name in self.cell_types:
            if _name == "Medium":
                if type_id is not None and type_id != 0:
                    raise SpecValueError(f"Medium always had id 0")
                return
            else:
                raise SpecValueError(f"Type name {_name} already specified")

        if type_id is not None:
            if type_id in self.cell_type_ids:
                raise SpecValueError(f"Type id {type_id} already specified")
        else:
            type_id = max(self.cell_type_ids) + 1

        self.spec_dict["cell_types"].append((_name, type_id, frozen))

    def cell_type_remove(self, _name: str) -> None:
        """
        Remove a cell type

        :param _name: name of cell type
        :raises SpecValueError: when attempting to remove the medium
        :return: None
        """
        if _name == "Medium":
            raise SpecValueError("Cannot remove the medium")

        for _i in range(len(self.spec_dict["cell_types"])):
            if self.spec_dict["cell_types"][_i][0] == _name:
                self.spec_dict["cell_types"].pop(_i)
                return

    def cell_type_rename(self, old_name: str, new_name: str) -> None:
        """
        Rename a cell type

        :param old_name: old cell type name
        :param new_name: new cell type name
        :raises SpecValueError: when old name is not specified or new name is already specified
        :return: None
        """
        if old_name not in self.cell_types:
            raise SpecValueError(f"Type name {old_name} not specified")
        elif new_name in self.cell_types:
            raise SpecValueError(f"Type name {new_name} already specified")

        x = self.cell_types.index(old_name)
        y = self.spec_dict["cell_types"][x]
        self.spec_dict["cell_types"][x] = (new_name, y[1], y[2])

    def type_id_relabel(self, old_id: int, new_id: int) -> None:
        """
        Change id of a cell type

        :param old_id: old id
        :param new_id: new id
        :raises SpecValueError: when old id is not specified or new id is already specified
        :return: None
        """
        if old_id not in self.cell_type_ids:
            raise SpecValueError(f"Type id {old_id} not specified")
        elif new_id in self.cell_type_ids:
            raise SpecValueError(f"Type id {new_id} already specified")

        x = self.cell_type_ids.index(old_id)
        y = self.spec_dict["cell_types"][x]
        self.spec_dict["cell_types"][x] = (y[0], new_id, y[2])

    def frozen_set(self, _type_name: str, freeze: bool) -> None:
        """
        Set frozen state of a cell type

        :param _type_name: cell type name
        :param freeze: frozen state
        :raises SpecValueError: when cell type name is not specified
        :return: None
        """
        if _type_name not in self.cell_types:
            raise SpecValueError(f"Type name {_type_name} not specified")
        x = self.cell_types.index(_type_name)
        y = self.spec_dict["cell_types"][x]
        self.spec_dict["cell_types"][x] = (y[0], y[1], freeze)


class VolumeEnergyParameter(_PyCoreSpecsBase):
    """ Volume Energy Parameter """

    check_dict = {"target_volume": (lambda x: x < 0, "Target volume must be non-negative")}

    def __init__(self, _cell_type: str, target_volume: float, lambda_volume: float):
        """

        :param _cell_type: cell type name
        :param target_volume: target volume
        :param lambda_volume: lambda value
        """
        super().__init__()

        self.check_inputs(target_volume=target_volume)

        self.spec_dict = {"cell_type": _cell_type,
                          "target_volume": target_volume,
                          "lambda_volume": lambda_volume}

    cell_type: str = SpecProperty(name="cell_type")
    """name of cell type"""

    target_volume: float = SpecProperty(name="target_volume")
    """target volume"""

    lambda_volume: float = SpecProperty(name="lambda_volume")
    """lambda volume"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = ElementCC3D("VolumeEnergyParameters", {"CellType": self.cell_type,
                                                          "TargetVolume": str(self.target_volume),
                                                          "LambdaVolume": str(self.lambda_volume)})
        return self._el

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                CoreSpecsValidator.validate_cell_type_names(type_names=[self.cell_type], cell_type_spec=s)


class VolumePlugin(_PyCorePluginSpecs):
    """
    Volume Plugin

    VolumeEnergyParameter instances can be accessed as follows,

    .. code-block:: python

       spec: VolumePlugin
       cell_type_name: str
       params: VolumeEnergyParameter = spec[cell_type_name]
       target_volume: float = params.target_volume
       lambda_volume: float = params.lambda_volume

    """

    name = "volume"
    registered_name = "Volume"

    def __init__(self, *_params):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _p in _params:
                with err_ctx.ctx:
                    if not isinstance(_p, VolumeEnergyParameter):
                        raise SpecValueError("Only VolumeEnergyParameter instances can be passed",
                                             names=[_p.__name__])

        self.spec_dict = {"params": {_p.cell_type: _p for _p in _params}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        for _p in self.spec_dict["params"].values():
            self._el.add_child(_p.xml)
        return self._el

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                CoreSpecsValidator.validate_cell_type_names(type_names=self.cell_types, cell_type_spec=s)

        # Validate parameters
        [param.validate(*specs) for param in self.spec_dict["params"].values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: VolumePlugin
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("VolumeEnergyParameters"))

        for p_el in el_list:
            o.param_new(p_el.getAttribute("CellType"),
                        target_volume=p_el.getAttributeAsDouble("TargetVolume"),
                        lambda_volume=p_el.getAttributeAsDouble("LambdaVolume"))
        return o

    def __getitem__(self, item):
        if item not in self.cell_types:
            raise SpecValueError(f"Cell type {item} not specified")
        return self.spec_dict["params"][item]

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return list(self.spec_dict["params"].keys())

    def param_append(self, _p: VolumeEnergyParameter) -> None:
        """
        Appends a volume energy parameter

        :param _p: volume energy parameter
        :return: None
        """
        if _p.cell_type in self.cell_types:
            raise SpecValueError(f"Cell type {_p.cell_type} already specified")
        self.spec_dict["params"][_p.cell_type] = _p

    def param_remove(self, _cell_type: str) -> None:
        """
        Remove a parameter by cell type

        :param _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Cell type {_cell_type} not specified")
        self.spec_dict["params"].pop(_cell_type)

    def param_new(self, _cell_type: str, target_volume: float, lambda_volume: float) -> VolumeEnergyParameter:
        """
        Appends and returns a new volume energy parameter

        :param _cell_type: cell type name
        :param target_volume: target volume
        :param lambda_volume: lambda value
        :return: new volume energy parameter
        """
        p = VolumeEnergyParameter(_cell_type, target_volume=target_volume, lambda_volume=lambda_volume)
        self.param_append(p)
        return p

