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


class _PDEDiffusionDataSpecs(_PyCoreSpecsBase, ABC):
    """
    Diffusion Data Base Class

    Difussion data varies significantly by solver

    This base class aggregates all available options for consistent usage throughout various implementations

    Implementations should make options available as appropriate to a particular solver
    """

    check_dict = {
        "diff_global": (lambda x: x < 0, "Diffusion coefficient must be non-negative"),
        "decay_global": (lambda x: x < 0, "Decay coefficient must be non-negative"),
        "diff_const": (lambda x: x < 0, "Diffusion coefficient must be non-negative"),
        "decay_const": (lambda x: x < 0, "Decay coefficient must be non-negative")
    }

    def __init__(self, field_name: str):
        super().__init__()

        self.spec_dict = {"field_name": field_name,  # FieldName
                          "diff_global": 0.0,  # GlobalDiffusionConstant / DiffusionConstant
                          "decay_global": 0.0,  # GlobalDecayConstant / DecayConstant
                          "diff_types": dict(),  # DiffusionCoefficient
                          "decay_types": dict(),  # DecayCoefficient
                          "additional_term": "",  # AdditionalTerm
                          "init_expression": "",  # InitialConcentrationExpression
                          "init_filename": "",  # ConcentrationFileName
                          }

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    def generate_header(self) -> ElementCC3D:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("DiffusionData")


class SecretionParameters(_PyCoreSpecsBase):
    """ Secretion Specs """

    def __init__(self,
                 _cell_type: str,
                 _val: float,
                 constant: bool = False,
                 contact_type: str = None):
        """

        :param _cell_type: cell type name
        :param _val: value of parameter
        :param constant: flag for constant concentration, optional
        :param contact_type: name of cell type for on-contact dependence, optional
        :raises SpecValueError: when using both constant concentration and on-contact dependence
        """
        super().__init__()

        if constant and contact_type is not None:
            raise SpecValueError("SecretionOnContact and ConstantConcentration cannot both be employed")

        self.spec_dict = {"cell_type": _cell_type,
                          "value": _val,
                          "constant": constant,
                          "contact_type": contact_type}

    cell_type: str = SpecProperty(name="cell_type")
    """cell type name"""

    value: float = SpecProperty(name="value")
    """value of rate or constant concentration"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        attr = {"Type": self.cell_type}
        if self.contact_type is not None:
            n = "SecretionOnContact"
            attr["SecreteOnContactWith"] = self.contact_type
        elif self.constant:
            n = "ConstantConcentration"
        else:
            n = "Secretion"

        self._el = ElementCC3D(n, attr, str(self.value))
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
        err_ctx = _SpecValueErrorContext()

        # Validate dependencies
        with err_ctx:
            super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                with err_ctx:
                    CoreSpecsValidator.validate_cell_type_names(type_names=[self.cell_type], cell_type_spec=s)
                if self.contact_type is not None:
                    with err_ctx:
                        CoreSpecsValidator.validate_cell_type_names(type_names=[self.contact_type], cell_type_spec=s)

        if err_ctx:
            err_ctx.raise_error()

    @property
    def contact_type(self) -> Optional[str]:
        """

        :return: name of cell type if using secretion on contact with
        """
        return self.spec_dict["contact_type"]

    @contact_type.setter
    def contact_type(self, _name: Optional[str]) -> None:
        if _name is not None:
            self.constant = False
        self.spec_dict["contact_type"] = _name

    @property
    def constant(self) -> bool:
        """

        :return: flag whether using constant concentration
        """
        return self.spec_dict["constant"]

    @constant.setter
    def constant(self, _val: bool) -> None:
        if _val:
            self.contact_type = None
        self.spec_dict["constant"] = _val


class _PDESecretionDataSpecs(_PyCoreSpecsBase):
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


# Surface Plugin


class SurfaceEnergyParameter(_PyCoreSpecsBase):
    """ Surface Energy Parameter """

    check_dict = {"target_surface": (lambda x: x < 0, "Target surface must be non-negative")}

    def __init__(self, _cell_type: str, target_surface: float, lambda_surface: float):
        """

        :param _cell_type: cell type name
        :param target_surface: target surface
        :param lambda_surface: lambda value
        """
        super().__init__()

        self.check_inputs(target_surface=target_surface)

        self.spec_dict = {"cell_type": _cell_type,
                          "target_surface": target_surface,
                          "lambda_surface": lambda_surface}

    cell_type: str = SpecProperty(name="cell_type")
    """name of cell type"""

    target_surface: float = SpecProperty(name="target_surface")
    """target surface"""

    lambda_surface: float = SpecProperty(name="lambda_surface")
    """lambda surface"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = ElementCC3D("SurfaceEnergyParameters", {"CellType": self.cell_type,
                                                           "TargetSurface": str(self.target_surface),
                                                           "LambdaSurface": str(self.lambda_surface)})
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


class SurfacePlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    Surface Plugin

    SurfaceEnergyParameter instances can be accessed as follows,

    .. code-block:: python

       spec: SurfacePlugin
       cell_type_name: str
       params: SurfaceEnergyParameter = spec[cell_type_name]
       target_surface: float = params.target_surface
       lambda_surface: float = params.lambda_surface

    """

    name = "surface"
    registered_name = "Surface"

    def __init__(self, *_params):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _p in _params:
                with err_ctx.ctx:
                    if not isinstance(_p, SurfaceEnergyParameter):
                        raise SpecValueError("Only SurfaceEnergyParameter instances can be passed",
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
        :rtype: SurfacePlugin
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("SurfaceEnergyParameters"))

        for p_el in el_list:
            o.param_new(p_el.getAttribute("CellType"),
                        target_surface=p_el.getAttributeAsDouble("TargetSurface"),
                        lambda_surface=p_el.getAttributeAsDouble("LambdaSurface"))
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

    def param_append(self, _p: SurfaceEnergyParameter):
        """
        Appends a surface energy parameter
        
        :param _p: surface energy parameter
        :return: None
        """
        if _p.cell_type in self.cell_types:
            raise SpecValueError(f"Cell type {_p.cell_type} already specified")
        self.spec_dict["params"][_p.cell_type] = _p

    def param_remove(self, _cell_type: str):
        """
        Remove a parameter by cell type

        :param _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Cell type {_cell_type} not specified")
        self.spec_dict["params"].pop(_cell_type)

    def param_new(self, cell_type: str, target_surface: float, lambda_surface: float) -> SurfaceEnergyParameter:
        """
        Appends and returns a new surface energy parameter

        :param cell_type: cell type name
        :param target_surface: target surface
        :param lambda_surface: lambda value
        """
        p = SurfaceEnergyParameter(cell_type, target_surface=target_surface, lambda_surface=lambda_surface)
        self.param_append(p)
        return p


class NeighborTrackerPlugin(_PyCorePluginSpecs):
    """ NeighborTracker Plugin """

    name = "neighbor_tracker"
    registered_name = "NeighborTracker"

    def __init__(self):
        super().__init__()

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: NeighborTrackerPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Chemotaxis Plugin


class ChemotaxisTypeParameters(_PyCoreSpecsBase):
    """ Chemotaxis cell type parameter specs """

    check_dict = {
        "cell_type": (lambda x: not isinstance(x, str), "Cell type name must be a string"),
        "lambda_chemo": (lambda x: not isinstance(x, float), "Chemotaxis lambda must be a float"),
        "sat_cf": (lambda x: x is not None and not isinstance(x, float), "Chemotaxis saturation must be a float"),
        "linear_sat_cf": (lambda x: x is not None and not isinstance(x, float),
                          "Chemotaxis linear saturation must be a float"),
        "towards": (lambda x: x is not None and not isinstance(x, str), "Towards cell type name must be a string")
    }

    def __init__(self,
                 _cell_type: str,
                 lambda_chemo: float = 0.0,
                 sat_cf: float = None,
                 linear_sat_cf: float = None,
                 towards: str = None):
        """

        :param _cell_type: name of cell type
        :param lambda_chemo: lambda value
        :param sat_cf: saturation coefficient
        :param linear_sat_cf: linear saturation coefficient
        :param towards: cell type name to chemotax towards, optional
        """
        super().__init__()

        self.spec_dict = {"cell_type": _cell_type,
                          "lambda_chemo": lambda_chemo,
                          "sat_cf": sat_cf,
                          "linear_sat_cf": linear_sat_cf,
                          "towards": towards}

    cell_type: str = SpecProperty(name="cell_type")
    """name of cell type"""

    lambda_chemo: float = SpecProperty(name="lambda_chemo")
    """lambda chemotaxis"""

    sat_cf: Optional[float] = SpecProperty(name="sat_cf")
    """saturation coefficient, Optional, None if not set"""

    linear_sat_cf: Optional[float] = SpecProperty(name="linear_sat_cf")
    """linear saturation coefficient, Optional, None if not set"""

    towards: Optional[str] = SpecProperty(name="towards")
    """name of cell type if chemotaxing towards, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        attr_dict = {"Type": self.cell_type,
                     "Lambda": str(self.lambda_chemo)}
        if self.sat_cf is not None:
            attr_dict["SaturationCoef"] = str(self.sat_cf)
        elif self.linear_sat_cf is not None:
            attr_dict["SaturationLinearCoef"] = str(self.linear_sat_cf)
        if self.towards is not None:
            attr_dict["ChemotactTowards"] = self.towards
        self._el = ElementCC3D("ChemotaxisByType", attr_dict)
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


class ChemotaxisParameters(_PyCoreSpecsBase):
    """ Chemotaxis parameter specs"""

    check_dict = {
        "field_name": (lambda x: not isinstance(x, str), "Field name must be a string"),
        "solver_name": (lambda x: not isinstance(x, str), "Solver name must be a string")
    }

    def __init__(self,
                 field_name: str,
                 solver_name: str,
                 *_type_specs):
        """

        :param field_name: name of field
        :param solver_name: name of solver
        :param _type_specs: variable number of ChemotaxisTypeParameters instances
        """
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _ts in _type_specs:
                with err_ctx.ctx:
                    if not isinstance(_ts, ChemotaxisTypeParameters):
                        raise SpecValueError("Only ChemotaxisTypeParameters instances can specify chemotaxis type data",
                                             names=[_ts.__name__])

        self.spec_dict = {"field_name": field_name,
                          "solver_name": solver_name,
                          "type_specs": {_ts.cell_type: _ts for _ts in _type_specs}}

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    solver_name: str = SpecProperty(name="solver_name")
    """name of PDE solver of field"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        attrs = {"Name": self.field_name}
        if self.solver_name:
            attrs["Source"] = self.solver_name
        return ElementCC3D("ChemicalField", attrs)

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(_ts.xml) for _ts in self.spec_dict["type_specs"].values()]
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

        # Validate field name against solvers
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        # Validate type specs
        [param.validate(*specs) for param in self.spec_dict["type_specs"].values()]

    def __getitem__(self, item):
        if item not in self.spec_dict["type_specs"].keys():
            raise SpecValueError(f"Type name {item} not specified")
        return self.spec_dict["type_specs"][item]

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return list(self.spec_dict["type_specs"].keys())

    def params_append(self, type_spec: ChemotaxisTypeParameters) -> None:
        """
        Appends a chemotaxis type spec

        :param type_spec: chemotaxis type spec
        :return: None
        """
        if type_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {type_spec.cell_type} already specified")
        self.spec_dict["type_specs"][type_spec.cell_type] = type_spec

    def params_remove(self, _cell_type: str) -> None:
        """
        Removes a chemotaxis type spec

        :param _cell_type: cell type name
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Type name {_cell_type} not specified")
        self.spec_dict["type_specs"].pop(_cell_type)

    def params_new(self,
                   _cell_type: str,
                   lambda_chemo: float = 0.0,
                   sat_cf: float = None,
                   linear_sat_cf: float = None,
                   towards: str = None) -> ChemotaxisTypeParameters:
        """
        Appends and returns a new chemotaxis type spec

        :param _cell_type: name of cell type
        :param lambda_chemo: lambda value
        :param sat_cf: saturation coefficient
        :param linear_sat_cf: linear saturation coefficient
        :param towards: cell type name to chemotax towards, optional
        :return: new chemotaxis type spec
        """
        p = ChemotaxisTypeParameters(_cell_type,
                                     lambda_chemo=lambda_chemo,
                                     sat_cf=sat_cf,
                                     linear_sat_cf=linear_sat_cf,
                                     towards=towards)
        self.params_append(p)
        return p


class ChemotaxisPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    Chemotaxis Plugin

    ChemotaxisTypeParameters instances can be accessed by field name as follows,

    .. code-block:: python

       spec: ChemotaxisPlugin
       field_name: str
       cell_type_name: str
       field_params: ChemotaxisParameters = spec[field_name]
       type_params: ChemotaxisTypeParameters = field_params[cell_type_name]
       lambda_chemo: float = type_params.lambda_chemo
       sat_cf: float = type_params.sat_cf
       linear_sat_cf: float = type_params.linear_sat_cf
       towards: str = type_params.towards

    """

    name = "chemotaxis"
    registered_name = "Chemotaxis"

    def __init__(self, *_field_specs):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _fs in _field_specs:
                with err_ctx.ctx:
                    if not isinstance(_fs, ChemotaxisParameters):
                        raise SpecValueError("Can only pass ChemotaxisParameters instances",
                                             names=[_fs.__name__])

        self.spec_dict = {"field_specs": {_fs.field_name: _fs for _fs in _field_specs}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(_fs.xml) for _fs in self.spec_dict["field_specs"].values()]
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

        # Validate field name against solvers
        CoreSpecsValidator.validate_field_names(*specs, field_names=self.fields)
        [CoreSpecsValidator.validate_field_name_unique(*specs, field_name=f) for f in self.fields]

        # Validate fields
        [fs.validate(*specs) for fs in self.spec_dict["field_specs"].values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ChemotaxisPlugin
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("ChemicalField"))

        for f_el in el_list:
            field_name = f_el.getAttribute("Name")
            # Allowing unspecified Source
            solver_name = ""
            if f_el.findAttribute("Source"):
                solver_name = f_el.getAttribute("Source")
            f = ChemotaxisParameters(field_name=field_name, solver_name=solver_name)
            f_el_list = CC3DXMLListPy(f_el.getElements("ChemotaxisByType"))

            for p_el in f_el_list:
                cell_type = p_el.getAttribute("Type")
                kwargs = {"lambda_chemo": p_el.getAttributeAsDouble("Lambda")}
                if p_el.findAttribute("SaturationCoef"):
                    kwargs["sat_cf"] = p_el.getAttributeAsDouble("SaturationCoef")
                elif p_el.findAttribute("SaturationLinearCoef"):
                    kwargs["linear_sat_cf"] = p_el.getAttributeAsDouble("SaturationLinearCoef")
                if p_el.findAttribute("ChemotactTowards"):
                    kwargs["towards"] = p_el.getAttribute("ChemotactTowards")
                p = ChemotaxisTypeParameters(cell_type, **kwargs)
                f.params_append(p)

            o.param_append(f)

        return o

    @property
    def fields(self) -> List[str]:
        """

        :return: list of registered field names
        """
        return [_fs.field_name for _fs in self.spec_dict["field_specs"].values()]

    def __getitem__(self, item) -> ChemotaxisParameters:
        if item not in self.fields:
            raise SpecValueError(f"Field name {item} not specified")
        return self.spec_dict["field_specs"][item]

    def param_append(self, _field_specs: ChemotaxisParameters) -> None:
        """
        Appends a chemotaxis parameter

        :param _field_specs: chemotaxis parameter
        :return: None
        """
        if _field_specs.field_name in self.fields:
            raise SpecValueError(f"Field name {_field_specs.field_name} already specified")
        self.spec_dict["field_specs"][_field_specs.field_name] = _field_specs

    def param_remove(self, _field_name: str) -> None:
        """
        Removes a new chemotaxis parameter

        :param _field_name: name of field
        :return: None
        """
        if _field_name not in self.fields:
            raise SpecValueError(f"Field name {_field_name} not specified")
        self.spec_dict["field_specs"].pop(_field_name)

    def param_new(self,
                  field_name: str,
                  solver_name: str,
                  *_type_specs) -> ChemotaxisParameters:
        """
        Appends and returns a new chemotaxis parameter

        :param field_name: name of field
        :param solver_name: name of solver
        :param _type_specs: variable number of ChemotaxisTypeParameters instances
        :return: new chemotaxis parameter
        """
        p = ChemotaxisParameters(field_name=field_name, solver_name=solver_name, *_type_specs)
        self.param_append(p)
        return p


# ExternalPotential Plugin


class ExternalPotentialParameter(_PyCoreSpecsBase):
    """ External Potential Parameter """

    def __init__(self,
                 _cell_type: str,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0):
        super().__init__()

        self.spec_dict = {"cell_type": _cell_type,
                          "x": x,
                          "y": y,
                          "z": z}

    cell_type: str = SpecProperty(name="cell_type")
    """cell type name"""

    x: float = SpecProperty(name="x")
    """x-component of external potential lambda vector"""

    y: float = SpecProperty(name="y")
    """y-component of external potential lambda vector"""

    z: float = SpecProperty(name="z")
    """z-component of external potential lambda vector"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("ExternalPotentialParameters", {"CellType": self.cell_type,
                                                           "x": self.x, "y": self.y, "z": self.z})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
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


class ExternalPotentialPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    ExternalPotential Plugin

    ExternalPotentialParameter instances can be accessed by cell type name as follows,

    .. code-block:: python

       spec: ExternalPotentialPlugin
       cell_type_name: str
       params: ExternalPotentialParameter = spec.cell_type[cell_type_name]
       lambda_x: float = params.x
       lambda_y: float = params.y
       lambda_z: float = params.z

    """

    name = "external_potential"
    registered_name = "ExternalPotential"

    def __init__(self,
                 lambda_x: float = None,
                 lambda_y: float = None,
                 lambda_z: float = None,
                 com_based: bool = False
                 ):
        super().__init__()

        self.spec_dict = {"lambda_x": lambda_x,
                          "lambda_y": lambda_y,
                          "lambda_z": lambda_z,
                          "com_based": com_based,
                          "param_specs": dict()}

    com_based: bool = SpecProperty(name="com_based")
    """center-of-mass-based flag"""

    lambda_x: float = SpecProperty(name="lambda_x")
    """global x-component of external potential lambda vector"""

    lambda_y: float = SpecProperty(name="lambda_y")
    """global y-component of external potential lambda vector"""

    lambda_z: float = SpecProperty(name="lambda_z")
    """global z-component of external potential lambda vector"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.lambda_x is not None or self.lambda_y is not None or self.lambda_z is not None:
            attr_dict = {"x": str(0.0), "y": str(0.0), "z": str(0.0)}
            if self.lambda_x is not None:
                attr_dict["x"] = str(self.lambda_x)
            if self.lambda_y is not None:
                attr_dict["y"] = str(self.lambda_y)
            if self.lambda_z is not None:
                attr_dict["z"] = str(self.lambda_z)
            self._el.ElementCC3D("Lambda", attr_dict)

        if self.com_based:
            self._el.ElementCC3D("Algorithm", {}, "CenterOfMassBased")

        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"].values()]

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
        [param.validate(*specs) for param in self.spec_dict["param_specs"].values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ExternalPotentialPlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        if el.findElement("Lambda"):
            lam_el: CC3DXMLElement = el.getFirstElement("Lambda")
            if lam_el.findAttribute("x"):
                o.lambda_x = lam_el.getAttributeAsDouble("x")
            if lam_el.findAttribute("y"):
                o.lambda_y = lam_el.getAttributeAsDouble("y")
            if lam_el.findAttribute("z"):
                o.lambda_z = lam_el.getAttributeAsDouble("z")

        if el.findElement("Algorithm"):
            if el.getFirstElement("Algorithm").getText().lower() == "centerofmassbased":
                o.com_based = True

        el_list = CC3DXMLListPy(el.getElements("ExternalPotentialParameters"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            kwargs = {comp: p_el.getAttributeAsDouble(comp) for comp in ["x", "y", "z"] if p_el.findAttribute(comp)}
            o.param_append(ExternalPotentialParameter(p_el.getAttribute("CellType"), **kwargs))

        return o

    @property
    def cell_type(self) -> _PyCoreParamAccessor[ExternalPotentialParameter]:
        """

        :return: accessor to external potential parameters with cell types as keys
        """
        return _PyCoreParamAccessor(self, "param_specs")

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return list(self.spec_dict["param_specs"].keys())

    def param_append(self, param_spec: ExternalPotentialParameter = None) -> None:
        """
        Append external potential parameter

        :param param_spec: external potential parameters
        :return: None
        """
        if param_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {param_spec.cell_type} already specified")
        self.spec_dict["param_specs"][param_spec.cell_type] = param_spec

    def param_remove(self, _cell_type: str) -> None:
        """
        Remove a :class:`ExternalPotentialParameter` instance

        :param _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Type name {_cell_type} not specified")
        self.spec_dict["param_specs"].pop(_cell_type)

    def param_new(self,
                  _cell_type: str = None,
                  x: float = 0.0,
                  y: float = 0.0,
                  z: float = 0.0) -> ExternalPotentialParameter:
        """
        Appends and returns a new external potential parameter

        :param _cell_type: name of cell type
        :param x: x-component value
        :param y: y-component value
        :param z: z-component value
        :return: new parameter
        """
        p = ExternalPotentialParameter(_cell_type, x=x, y=y, z=z)
        self.param_append(p)
        return p


class CenterOfMassPlugin(_PyCorePluginSpecs):
    """ CenterOfMass Plugin """

    name = "center_of_mass"
    registered_name = "CenterOfMass"

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: CenterOfMassPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Contact Plugin


class ContactEnergyParameter(_PyCoreSpecsBase):
    """ Contact Energy Parameter """

    def __init__(self, type_1: str, type_2: str, energy: float):
        """

        :param type_1: first cell type name
        :param type_2: second cell type name
        :param energy: parameter value
        """
        super().__init__()

        self.spec_dict = {"type_1": type_1,
                          "type_2": type_2,
                          "energy": energy}

    type_1: str = SpecProperty(name="type_1")
    """first cell type"""

    type_2: str = SpecProperty(name="type_2")
    """second cell type"""

    energy: float = SpecProperty(name="energy")
    """contact energy parameter"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = ElementCC3D("Energy", {"Type1": self.type_1, "Type2": self.type_2}, str(self.energy))
        return self._el

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        # Validate dependencies
        super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                CoreSpecsValidator.validate_cell_type_names(type_names=[self.type_1, self.type_2], cell_type_spec=s)


class ContactPlugin(_PyCorePluginSpecs):
    """
    Contact Plugin

    ContactEnergyParameter instances can be accessed as follows,

    .. code-block:: python

       spec: ContactPlugin
       cell_type_1: str
       cell_type_2: str
       params: ContactEnergyParameter = spec[cell_type_1][cell_type_2]
       energy: float = params.energy

    """

    name = "contact"
    registered_name = "Contact"

    check_dict = {"neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")}

    def __init__(self, neighbor_order: int = 1, *_params):
        """

        :param neighbor_order: neighbor order
        :param _params: variable number of ContactEnergyParameter instances
        """
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _p in _params:
                with err_ctx.ctx:
                    if not isinstance(_p, ContactEnergyParameter):
                        raise SpecValueError("Only ContactEnergyParameter instances can be passed",
                                             names=[_p.__name__])

            self.check_inputs(neighbor_order=neighbor_order)

        self.spec_dict = {"energies": {},
                          "neighbor_order": neighbor_order}
        [self.param_append(_p) for _p in _params]

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        for _pdict in self.spec_dict["energies"].values():
            for _p in _pdict.values():
                self._el.add_child(_p.xml)
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        return self._el

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        # Validate dependencies
        super().validate(*specs)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                for t1 in self.spec_dict["energies"].keys():
                    CoreSpecsValidator.validate_cell_type_names(type_names=self.types_specified(t1) + [t1],
                                                                cell_type_spec=s)

        # Validate parameters
        for _pdict in self.spec_dict["energies"].values():
            [_p.validate(*specs) for _p in _pdict.values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ContactPlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = CC3DXMLListPy(el.getElements("Energy"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            o.param_append(ContactEnergyParameter(type_1=p_el.getAttribute("Type1"),
                                                  type_2=p_el.getAttribute("Type2"),
                                                  energy=p_el.getDouble()))

        if el.findElement("NeighborOrder"):
            o.neighbor_order = el.getFirstElement("NeighborOrder").getInt()

        return o

    def __getitem__(self, item: str) -> Dict[str, ContactEnergyParameter]:
        return self.spec_dict["energies"][item]

    def types_specified(self, _type_name: str) -> List[str]:
        """

        :return: list of cell type names
        """
        d = self.spec_dict["energies"]
        o = []
        for k, v in d.items():
            for kk in v.keys():
                if k == _type_name:
                    o.append(kk)
                elif kk == _type_name:
                    o.append(k)
        return o

    def param_append(self, _p: ContactEnergyParameter) -> None:
        """
        Add a contact energy parameter

        :param _p: the parameter
        :return: None
        """
        if _p.type_2 in self.types_specified(_p.type_1):
            raise SpecValueError(f"Contact parameter already specified for ({_p.type_1}, {_p.type_2})")
        if _p.type_1 not in self.spec_dict["energies"]:
            self.spec_dict["energies"][_p.type_1] = {}
        self.spec_dict["energies"][_p.type_1][_p.type_2] = _p

    def param_remove(self, type_1: str, type_2: str) -> None:
        """
        Removes contact energy parameter

        :param type_1: name of first cell type
        :param type_2: name of second cell type
        :return: None
        """
        if type_2 not in self.types_specified(type_1):
            raise SpecValueError(f"Contact parameter not specified for ({type_1}, {type_2})")
        if type_1 in self.spec_dict["energies"].keys():
            key, val = type_1, type_2
        else:
            key, val = type_2, type_1
        self.spec_dict["energies"][key].pop(val)

    def param_new(self, type_1: str, type_2: str, energy: float) -> ContactEnergyParameter:
        """
        Appends and returns a new contact energy parameter

        :param type_1: first cell type name
        :param type_2: second cell type name
        :param energy: parameter value
        :return: new parameter
        """
        p = ContactEnergyParameter(type_1=type_1, type_2=type_2, energy=energy)
        self.param_append(p)
        return p


class ContactLocalFlexPlugin(ContactPlugin, _PyCoreSteerableInterface):
    """
    ContactLocalFlex Plugin

    A steerable version of :class:`ContactPlugin`
    """

    name = "contact_local_flex"
    registered_name = "ContactLocalFlex"


class ContactInternalPlugin(ContactLocalFlexPlugin):
    """
    ContactInternal Plugin

    Like :class:`ContactLocalFlexPlugin`, but for contact between compartments
    """

    name = "contact_internal"
    registered_name = "ContactInternal"


# AdhesionFlex Plugin


class AdhesionFlexBindingFormula(_PyCoreSpecsBase):
    """
    AdhesionFlex Binding Formula

    Binding parameters can be set like a dictionary as follows

    .. code-block:: python

       spec: AdhesionFlexBindingFormula
       molecule_1_name: str
       molecule_2_name: str
       density: float
       spec[molecule_1_name][molecule_2_name] = density

    """

    def __init__(self, formula_name: str, formula: str):
        """

        :param formula_name: name of forumla
        :param formula: formula
        """
        super().__init__()

        self.spec_dict = {"formula_name": formula_name,
                          "formula": formula,
                          "interactions": dict()}

    formula_name: str = SpecProperty(name="formula_name")
    """name of formula"""

    formula: str = SpecProperty(name="formula")
    """formula"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("BindingFormula", {"Name": self.formula_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("Formula", {}, self.formula)
        var_el = self._el.ElementCC3D("Variables")
        mat_el = var_el.ElementCC3D("AdhesionInteractionMatrix")
        for m1, m1_dict in self.spec_dict["interactions"].items():
            for m2 in m1_dict.keys():
                mat_el.ElementCC3D("BindingParameter", {"Molecule1": m1, "Molecule2": m2}, str(m1_dict[m2]))

        return self._el

    def __getitem__(self, item: str) -> dict:
        """

        :param item: molecule 1 name
        :return: dictionary of densities
        """
        if item not in self.spec_dict["interactions"].keys():
            self.spec_dict["interactions"][item] = dict()
        return self.spec_dict["interactions"][item]

    def param_set(self, _mol1: str, _mol2: str, _val: float) -> None:
        """
        Sets an adhesion binding parameter

        :param _mol1: name of first molecule
        :param _mol2: name of second molecule
        :param _val: binding parameter value
        :return: None
        """
        if _mol1 not in self.spec_dict["interactions"].keys():
            self.spec_dict["interactions"][_mol1] = dict()
        self.spec_dict["interactions"][_mol1][_mol2] = _val

    def param_remove(self, _mol1: str, _mol2: str) -> None:
        """
        Removes an adhesion binding parameter

        :param _mol1: name of first molecule
        :param _mol2: name of second molecule
        :return: None
        """
        if _mol1 not in self.spec_dict["interactions"].keys() \
                or _mol2 not in self.spec_dict["interactions"][_mol1].keys():
            raise SpecValueError(f"Parameter not defined for types ({_mol1}, {_mol2})")
        self.spec_dict["interactions"].pop(_mol2)


class AdhesionFlexMoleculeDensity(_PyCoreSpecsBase):
    """ AdhesionFlex molecule density spec """

    check_dict = {"density": (lambda x: x < 0.0, "Molecule density must be non-negative")}

    def __init__(self, molecule: str, cell_type: str, density: float):
        """

        :param molecule: name of molecule
        :param cell_type: name of cell type
        :param density: molecule density
        """
        super().__init__()

        self.spec_dict = {"molecule": molecule,
                          "cell_type": cell_type,
                          "density": density}

    molecule: str = SpecProperty(name="molecule")
    """name of molecule"""

    cell_type: str = SpecProperty(name="cell_type")
    """name of cell type"""

    density: float = SpecProperty(name="density")
    """molecule density"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("AdhesionMoleculeDensity", {"Molecule": self.molecule,
                                                       "CellType": self.cell_type,
                                                       "Density": str(self.density)})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
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


class AdhesionFlexPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    AdhesionFlex Plugin

    AdhesionFlexMoleculeDensity instances can be accessed by molecule and cell type as follows,

    .. code-block:: python

       specs_adhesion_flex: AdhesionFlexPlugin
       molecule_name: str
       cell_type_name: str
       x: dict = specs_adhesion_flex.density.cell_type[cell_type_name]  # keys are molecule names
       y: dict = specs_adhesion_flex.density.molecule[molecule_name]  # keys are cell type names
       a: AdhesionFlexMoleculeDensity = x[molecule_name]
       b: AdhesionFlexMoleculeDensity = y[cell_type_name]
       a is b  # Evaluates to True

    AdhesionFlexBindingFormula instances can be accessed as follows,

    .. code-block:: python

       specs_adhesion_flex: AdhesionFlexPlugin
       formula_name: str
       molecule1_name: str
       molecule2_name: str
       x: AdhesionFlexBindingFormula = specs_adhesion_flex.formula[formula_name]
       y: dict = x[molecule1_name]  # Keys are molecule names
       binding_param: float = y[molecule2_name]

    """

    name = "adhesion_flex"
    registered_name = "AdhesionFlex"

    check_dict = {"neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")}

    def __init__(self, neighbor_order: int = 1):
        super().__init__()

        self.check_inputs(neighbor_order=neighbor_order)

        self.spec_dict = {"neighbor_order": neighbor_order,
                          "molecules": list(),
                          "densities": dict(),
                          "binding_formulas": dict()}

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.ElementCC3D("AdhesionMolecule", {"Molecule": m}) for m in self.molecules]
        [[self._el.add_child(e.xml) for e in e_dict.values()] for e_dict in self.spec_dict["densities"].values()]
        [self._el.add_child(e.xml) for e in self.spec_dict["binding_formulas"].values()]
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
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
                for param_dict in self.spec_dict["densities"].values():
                    CoreSpecsValidator.validate_cell_type_names(type_names=param_dict.keys(), cell_type_spec=s)

        # Validate density parameters
        for param_dict in self.spec_dict["densities"].values():
            [param.validate(*specs) for param in param_dict.values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: AdhesionFlexPlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = CC3DXMLListPy(el.getElements("AdhesionMolecule"))

        for m_el in el_list:
            o.molecule_append(m_el.getAttribute("Molecule"))

        el_list = CC3DXMLListPy(el.getElements("AdhesionMoleculeDensity"))

        for d_el in el_list:
            d_el: CC3DXMLElement
            o.density_append(AdhesionFlexMoleculeDensity(molecule=d_el.getAttribute("Molecule"),
                                                         cell_type=d_el.getAttribute("CellType"),
                                                         density=d_el.getAttributeAsDouble("Density")))

        el_list = CC3DXMLListPy(el.getElements("BindingFormula"))

        for f_el in el_list:
            f_el: CC3DXMLElement
            f = AdhesionFlexBindingFormula(formula_name=f_el.getAttribute("Name"),
                                           formula=f_el.getFirstElement("Formula").getText())
            p_mat_el = f_el.getFirstElement("Variables").getFirstElement("AdhesionInteractionMatrix")

            p_el_list = CC3DXMLListPy(p_mat_el.getElements("BindingParameter"))

            for p_el in p_el_list:
                p_el: CC3DXMLElement
                f.param_set(p_el.getAttribute("Molecule1"), p_el.getAttribute("Molecule2"), p_el.getDouble())

            o.formula_append(f)

        return o

    @property
    def molecules(self) -> List[str]:
        """

        :return: list of registered molecule names
        """
        return [x for x in self.spec_dict["molecules"]]

    @property
    def formulas(self) -> List[str]:
        """

        :return: list of register formula names
        """
        return [x for x in self.spec_dict["binding_formulas"].keys()]

    @property
    def formula(self) -> _PyCoreParamAccessor[AdhesionFlexBindingFormula]:
        """

        :return: accessor to binding formulas with formula names as keys
        """
        return _PyCoreParamAccessor(self, "binding_formulas")

    @property
    def density(self):
        """

        :return: accessor to adhesion molecule density accessor
        :rtype: _AdhesionFlexMoleculeDensityAccessor
        """
        return _AdhesionFlexMoleculeDensityAccessor(self)

    def molecule_append(self, _name: str) -> None:
        """
        Append a molecule name

        :param _name: name of molecule
        :return: None
        """
        if _name in self.molecules:
            raise SpecValueError(f"Molecule with name {_name} already specified")
        self.spec_dict["molecules"].append(_name)

    def molecule_remove(self, _name: str) -> None:
        """
        Remove molecule and all associated binding parameters and densities

        :param _name: name of molecule
        :return: None
        """
        if _name not in self.molecules:
            raise SpecValueError(f"Molecule with name {_name} not specified")

        if _name in self.spec_dict["densities"].keys():
            self.spec_dict["densities"].pop(_name)

        for k, v in self.spec_dict["binding_formulas"].items():
            for mol1 in self.molecules:
                for mol2 in self.molecules:
                    try:
                        v.param_remove(mol1, mol2)
                    except SpecValueError:
                        pass
                    try:
                        v.param_remove(mol2, mol1)
                    except SpecValueError:
                        pass

        self.spec_dict["molecules"].remove(_name)

    def density_append(self, _dens: AdhesionFlexMoleculeDensity) -> None:
        """
        Appends a molecule density; molecules are automatically appended if necessary

        :param _dens: adhesion molecule density spec
        :return: None
        """
        if _dens.molecule not in self.molecules:
            self.molecule_append(_dens.molecule)
        if _dens.molecule not in self.spec_dict["densities"].keys():
            self.spec_dict["densities"][_dens.molecule] = dict()
        x: dict = self.spec_dict["densities"][_dens.molecule]
        x[_dens.cell_type] = _dens

    def density_remove(self, molecule: str, cell_type: str) -> None:
        """
        Removes a molecule density

        :param molecule: name of molecule
        :param cell_type: name of cell type
        :return: None
        """
        if molecule not in self.molecules:
            raise SpecValueError(f"Molecule {molecule} not specified")
        if cell_type not in self.spec_dict["densities"][molecule].keys():
            raise SpecValueError(f"Molecule {molecule} density not specified for cell type {cell_type}")
        x: dict = self.spec_dict["densities"][molecule]
        x.pop(cell_type)

    def density_new(self, molecule: str, cell_type: str, density: float) -> AdhesionFlexMoleculeDensity:
        """
        Appends and returns a new adhesion molecule density spec

        :param molecule: name of molecule
        :param cell_type: name of cell type
        :param density: molecule density
        :return: new adhesion molecule density spec
        """
        p = AdhesionFlexMoleculeDensity(molecule=molecule, cell_type=cell_type, density=density)
        self.density_append(p)
        return p

    def formula_append(self, _formula: AdhesionFlexBindingFormula) -> None:
        """
        Append a binding formula spec

        :param _formula: binding formula spec
        :return: None
        """
        if _formula.formula_name in self.formulas:
            raise SpecValueError(f"Formula with name {_formula.formula_name} already specified")
        self.spec_dict["binding_formulas"][_formula.formula_name] = _formula

    def formula_remove(self, _formula_name: str) -> None:
        """
        Remove a new binding formula spec

        :param _formula_name: name of formula
        :return: None
        """
        if _formula_name not in self.formulas:
            raise SpecValueError(f"Formula with name {_formula_name} not specified")
        self.spec_dict["binding_formulas"].pop(_formula_name)

    def formula_new(self,
                    formula_name: str = "Binary",
                    formula: str = "min(Molecule1,Molecule2)") -> AdhesionFlexBindingFormula:
        """
        Append and return a new binding formula spec

        :param formula_name: name of forumla
        :param formula: formula
        :return: new binding formula spec
        """
        p = AdhesionFlexBindingFormula(formula_name=formula_name, formula=formula)
        self.formula_append(p)
        return p


class _AdhesionFlexMoleculeDensityAccessor(_PyCoreSpecsBase):
    """
    AdhesionFlex molecule density accessor

    Container with convenience containers for :class:`AdhesionFlexPlugin` to instances of
    :class:`AdhesionFlexMoleculeDensity`
    """

    def __init__(self, _plugin_spec: AdhesionFlexPlugin):
        super().__init__()

        self._plugin_spec = _plugin_spec
        self.spec_dict = {"densities": {}}
        for vv in _plugin_spec.spec_dict["densities"].values():
            for v in vv.values():
                if v.cell_type not in self.spec_dict["densities"].keys():
                    self.spec_dict["densities"][v.cell_type] = dict()
                self.spec_dict["densities"][v.cell_type][v.molecule] = v

    @property
    def xml(self) -> ElementCC3D:
        """

        :raises SpecValueError: when accessed
        """
        raise SpecValueError("Accessor has no xml")

    @property
    def cell_type(self) -> _PyCoreParamAccessor[Dict[str, AdhesionFlexMoleculeDensity]]:
        """

        :return: accessor to adnesion molecule densities with cell type names as keys
        """
        return _PyCoreParamAccessor(self, "densities")

    @property
    def molecule(self) -> _PyCoreParamAccessor[Dict[str, AdhesionFlexMoleculeDensity]]:
        """

        :return: accessor to adnesion molecule densities with molecule names as keys
        """
        return _PyCoreParamAccessor(self._plugin_spec, "densities")


# LengthConstraint Plugin


class LengthEnergyParameters(_PyCoreSpecsBase):
    """ Length Energy Parameters """

    def __init__(self,
                 _cell_type: str,
                 target_length: float,
                 lambda_length: float,
                 minor_target_length: float = None):
        """

        :param _cell_type: cell type name
        :param target_length: target length
        :param lambda_length: lambda length
        :param minor_target_length: minor target length, optional
        """
        super().__init__()

        self.spec_dict = {"cell_type": _cell_type,
                          "target_length": target_length,
                          "lambda_length": lambda_length,
                          "minor_target_length": minor_target_length}

    cell_type: str = SpecProperty(name="cell_type")
    """cell type name"""

    target_length: float = SpecProperty(name="target_length")
    """target length"""

    lambda_length: float = SpecProperty(name="lambda_length")
    """lambda value"""

    minor_target_length: Optional[float] = SpecProperty(name="minor_target_length")
    """minor target length, Optional, None if not set"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        attr_dict = {"CellType": self.cell_type,
                     "TargetLength": str(self.target_length),
                     "LambdaLength": str(self.lambda_length)}
        if self.minor_target_length is not None:
            attr_dict["MinorTargetLength"] = str(self.minor_target_length)
        return ElementCC3D("LengthEnergyParameters", attr_dict)

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
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


class LengthConstraintPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    LengthConstraint Plugin

    LengthEnergyParameters instances can be accessed as follows,

    .. code-block:: python

       spec: LengthConstraintPlugin
       cell_type_name: str
       params: LengthEnergyParameters = spec[cell_type_name]
       target_length: float = spec.target_length
       lambda_length: float = spec.lambda_length
       minor_target_length: float = spec.minor_target_length  # Optional, None if not set

    """

    name = "length_constraint"
    registered_name = "LengthConstraint"

    def __init__(self):
        super().__init__()

        self.spec_dict = {"param_specs": {}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"].values()]
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
        [param.validate(*specs) for param in self.spec_dict["param_specs"].values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: LengthConstraintPlugin
        """
        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("LengthEnergyParameters"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            kwargs = {"target_length": p_el.getAttributeAsDouble("TargetLength"),
                      "lambda_length": p_el.getAttributeAsDouble("LambdaLength")}
            if p_el.findAttribute("MinorTargetLength"):
                kwargs["minor_target_length"] = p_el.getAttributeAsDouble("MinorTargetLength")
            o.params_new(p_el.getAttribute("CellType"), **kwargs)

        return o

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return list(self.spec_dict["param_specs"].keys())

    def __getitem__(self, item):
        if item not in self.cell_types:
            raise SpecValueError(f"Cell type {item} not specified")
        return self.spec_dict["param_specs"][item]

    def params_append(self, param_spec: LengthEnergyParameters):
        """
        Appens a length energy parameters spec

        :param param_spec: length energy parameters spec
        :return: None
        """
        if param_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {param_spec.cell_type} already specified")
        self.spec_dict["param_specs"][param_spec.cell_type] = param_spec

    def params_remove(self, _cell_type: str):
        """
        Removes a length energy parameters spec

        :param _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Type name {_cell_type} not specified")
        self.spec_dict["param_specs"].pop(_cell_type)

    def params_new(self,
                   _cell_type: str,
                   target_length: float,
                   lambda_length: float,
                   minor_target_length: float = None) -> LengthEnergyParameters:
        """
        Appends and returns a new length energy parameters spec

        :param _cell_type: cell type name
        :param target_length: target length
        :param lambda_length: lambda length
        :param minor_target_length: minor target length, optional
        :return: new length energy parameters spec
        """
        param_spec = LengthEnergyParameters(_cell_type,
                                            target_length=target_length,
                                            lambda_length=lambda_length,
                                            minor_target_length=minor_target_length)
        self.params_append(param_spec=param_spec)
        return param_spec


class ConnectivityPlugin(_PyCorePluginSpecs):
    """ Connectivity Plugin """

    name = "connectivity"
    registered_name = "Connectivity"

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ConnectivityPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class ConnectivityGlobalPlugin(_PyCorePluginSpecs):
    """ ConnectivityGlobal Plugin """

    name = "connectivity_global"
    registered_name = "ConnectivityGlobal"

    def __init__(self, fast: bool = False, *_cell_types):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _ct in _cell_types:
                with err_ctx.ctx:
                    if not isinstance(_ct, str):
                        raise SpecValueError("Only cell type names as strings can be pass",
                                             names=[_ct.__name__])

        self.spec_dict = {"fast": fast,
                          "cell_types": []}
        [self.cell_type_append(x) for x in _cell_types]

    fast: bool = SpecProperty(name="fast")
    """flag whether to use fast algorithm"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.fast:
            self._el.ElementCC3D("FastAlgorithm")
        for _ct in self.cell_types:
            self._el.ElementCC3D("ConnectivityOn", {"Type": _ct})
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

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ConnectivityGlobalPlugin
        """
        el = cls.find_xml_by_attr(_xml)
        el_list = CC3DXMLListPy(el.getElements("ConnectivityOn"))
        o = cls(*[e.getAttribute("Type") for e in el_list])
        o.fast = el.findAttribute("FastAlgorithm")
        return o

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        """
        return [x for x in self.spec_dict["cell_types"]]

    def cell_type_append(self, _name: str) -> None:
        """
        Appends a cell type name

        :param _name: name of cell type
        :raises SpecValueError: when cell type name has already been specified
        :return: None
        """
        if _name in self.cell_types:
            raise SpecValueError(f"Type name {_name} already specified")
        self.spec_dict["cell_types"].append(_name)

    def cell_type_remove(self, _name: str) -> None:
        """
        Removes a cell type name

        :param _name: name of cell type
        :raises SpecValueError: when cell type name has not been specified
        :return: None
        """
        if _name not in self.cell_types:
            raise SpecValueError(f"Type name {_name} not specified")
        self.spec_dict["cell_types"].remove(_name)


# Secretion Plugin


class SecretionField(_PyCoreSpecsBase):
    """ Secretion Field Specs """

    check_dict = {"frequency": (lambda x: x < 1, "Frequency must be positive")}

    def __init__(self, field_name: str, frequency: int = 1, *_param_specs):
        """

        :param field_name: name of field
        :param frequency: frequency of calls per step
        :param _param_specs: variable number of SecretionParameters instances, optional
        """
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _ps in _param_specs:
                with err_ctx.ctx:
                    if not isinstance(_ps, SecretionParameters):
                        raise SpecValueError("Only SecretionParameters instances can be passed",
                                             names=[_ps.__name__])

        self.spec_dict = {"field_name": field_name,
                          "frequency": frequency,
                          "param_specs": []}
        [self.params_append(_ps) for _ps in _param_specs]

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    frequency: int = SpecProperty(name="frequency")
    """frequency of calls per step"""

    params: List[SecretionParameters] = SpecProperty(name="param_specs")
    """secretion field parameters specs"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        attr = {"Name": self.field_name}
        if self.frequency > 1:
            attr["ExtraTimesPerMC"] = str(self.frequency)
        self._el = ElementCC3D("Field", attr)
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

        # Validate field name against solvers
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        # Validate parameters
        [param.validate(*specs) for param in self.spec_dict["param_specs"]]

    def params_append(self, _ps: SecretionParameters) -> None:
        """
        Append a secretion parameters spec

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
        Remove a secretion parameters spec

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
        Append and return a new secretion parameters spec

        :param _cell_type: cell type name
        :param _val: value of parameter
        :param constant: flag for constant concentration, option
        :param contact_type: name of cell type for on-contact dependence, optional
        :return: new secretion spec
        """
        ps = SecretionParameters(_cell_type, _val, constant=constant, contact_type=contact_type)
        self.params_append(ps)
        return ps


class SecretionPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    Secretion Plugin

    SecretionParameters instances can be accessed by field name and cell type name as follows,

    .. code-block:: python

       spec: SecretionPlugin
       field_name: str
       cell_type_name: str
       field_specs: SecretionField = spec.fields[field_name]
       params: SecretionParameters = field_specs.specs[cell_type_name]

    """

    name = "secretion"
    registered_name = "Secretion"

    def __init__(self, pixel_tracker: bool = True, boundary_pixel_tracker: bool = True, *_field_spec):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _fs in _field_spec:
                with err_ctx.ctx:
                    if not isinstance(_fs, SecretionField):
                        raise SpecValueError("Only SecretionField instances can be passed",
                                             names=[_fs.__name__])

        self.spec_dict = {"pixel_tracker": pixel_tracker,
                          "boundary_pixel_tracker": boundary_pixel_tracker,
                          "field_specs": {}}
        [self.field_append(x) for x in _field_spec]

    pixel_tracker: bool = SpecProperty(name="pixel_tracker")
    """flag to use PixelTracker Plugin"""

    boundary_pixel_tracker: bool = SpecProperty(name="boundary_pixel_tracker")
    """flag to use BoundaryPixelTracker Plugin"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if not self.pixel_tracker:
            self._el.ElementCC3D("DisablePixelTracker")
        if not self.boundary_pixel_tracker:
            self._el.ElementCC3D("DisableBoundaryPixelTracker")
        [self._el.add_child(_fs.xml) for _fs in self.spec_dict["field_specs"].values()]
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        deps = super().depends_on
        if self.pixel_tracker:
            deps.append(PixelTrackerPlugin)
        if self.boundary_pixel_tracker:
            deps.append(BoundaryPixelTrackerPlugin)
        return deps

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
                with err_ctx.ctx:
                    # Validate PixelTracker
                    if isinstance(s, PixelTrackerPlugin) and not self.pixel_tracker:
                        raise SpecValueError("Could not validated disabled PixelTracker Plugin")

                    # Validate disabled BoundaryPixelTracker
                    if isinstance(s, BoundaryPixelTrackerPlugin) and not self.boundary_pixel_tracker:
                        raise SpecValueError("Could not validated disabled BoundaryPixelTracker Plugin")

            # Validate field name against solvers
            with err_ctx.ctx:
                CoreSpecsValidator.validate_field_names(*specs, field_names=self.field_names)

            for f in self.field_names:
                with err_ctx.ctx:
                    CoreSpecsValidator.validate_field_name_unique(*specs, field_name=f)

            # Validate fields
            for f in self.spec_dict["field_specs"].values():
                with err_ctx.ctx:
                    f.validate(*specs)

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: SecretionPlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls(pixel_tracker=not el.findElement("DisablePixelTracker"),
                boundary_pixel_tracker=not el.findElement("DisableBoundaryPixelTracker"))

        el_list = CC3DXMLListPy(el.getElements("Field"))

        for f_el in el_list:
            f_el: CC3DXMLElement
            f = SecretionField(field_name=f_el.getAttribute("Name"))

            if f_el.findAttribute("ExtraTimesPerMC"):
                f.frequency = f_el.getAttributeAsInt("ExtraTimesPerMC")

            f_el_list = CC3DXMLListPy(f_el.getElements("Secretion"))

            for p_el in f_el_list:
                p_el: CC3DXMLElement
                f.params_append(SecretionParameters(p_el.getAttribute("Type"), p_el.getDouble()))

            f_el_list = CC3DXMLListPy(f_el.getElements("ConstantConcentration"))

            for p_el in f_el_list:
                f.params_append(SecretionParameters(p_el.getAttribute("Type"), p_el.getDouble(),
                                                    constant=True))

            f_el_list = CC3DXMLListPy(f_el.getElements("SecretionOnContact"))

            for p_el in f_el_list:
                f.params_append(SecretionParameters(p_el.getAttribute("Type"), p_el.getDouble(),
                                                    contact_type=p_el.getAttribute("SecreteOnContactWith")))

            o.field_append(f)

        return o

    @property
    def fields(self) -> _PyCoreParamAccessor[SecretionField]:
        """

        :return: accessor to secretion field specs with field names as keys
        """
        return _PyCoreParamAccessor(self, "field_specs")

    @property
    def field_names(self) -> List[str]:
        """

        :return: list of registered field names
        """
        return list(self.spec_dict["field_specs"].keys())

    def field_append(self, _fs: SecretionField) -> None:
        """
        Append a secretion field spec

        :param _fs: secretion field spec
        :return: None
        """
        if _fs.field_name in self.field_names:
            raise SpecValueError(f"Field specs for field {_fs.field_name} already specified")
        self.spec_dict["field_specs"][_fs.field_name] = _fs

    def field_remove(self, _field_name: str) -> None:
        """
        Remove a secretion field spec

        :param _field_name: field name
        :return: None
        """
        if _field_name not in self.field_names:
            raise SpecValueError(f"Field specs for field {_field_name} not specified")
        self.spec_dict["field_specs"].pop(_field_name)

    def field_new(self, field_name: str, frequency: int = 1, *_param_specs) -> SecretionField:
        """
        Append and return a new secretion field spec

        :param field_name: name of field
        :param frequency: frequency of calls per step
        :param _param_specs: variable number of SecretionParameters instances, optional
        :return: new secretion field spec
        """
        fs = SecretionField(field_name=field_name, frequency=frequency, *_param_specs)
        self.field_append(fs)
        return fs


# FocalPointPlasticity Plugin


class LinkConstituentLaw(_PyCoreSpecsBase):
    """
    Link Constituent Law

    Usage is as follows

    .. code-block:: python
       lcl: LinkConstituentLaw
       lcl.variable["LambdaExtra"] = 1.0
       lcl.formua = "LambdaExtra*Lambda*(Length-TargetLength)^2"

    The variables Lambda, Length, TargetLength are defined by default according to cell properties
    """

    def __init__(self,
                 formula: str):
        super().__init__()

        self.spec_dict = {"formula": formula,
                          "variables": dict()}

    formula: str = SpecProperty(name="formula")
    """law formula"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        el = ElementCC3D("LinkConstituentLaw")
        el.ElementCC3D("Formula", {}, self.formula)
        for name, val in self.spec_dict["variables"].items():
            el.ElementCC3D("Variable", {"Name": name, "Value": str(val)})
        self._el = el
        return self._el

    @property
    def variable(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to variable values with variable names as keys
        """
        return _PyCoreParamAccessor(self, "variables")


class FocalPointPlasticityParameters(_PyCoreSpecsBase):
    """ FocalPointPlasticityParameters """

    check_dict = {
        "target_distance": (lambda x: x < 0, "Target distance must be non-negative"),
        "max_distance": (lambda x: x < 0, "Maximum distance must be non-negative"),
        "max_junctions": (lambda x: x < 1, "Maximum number of junctions must be positive"),
        "neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")
    }

    def __init__(self,
                 _type1: str,
                 _type2: str,
                 lambda_fpp: float,
                 activation_energy: float,
                 target_distance: float,
                 max_distance: float,
                 max_junctions: int = 1,
                 neighbor_order: int = 1,
                 internal: bool = False,
                 law: LinkConstituentLaw = None):
        """

        :param _type1: second type name
        :param _type2: first type name
        :param lambda_fpp: lambda value
        :param activation_energy: activation energy
        :param target_distance: target distance
        :param max_distance: max distance
        :param max_junctions: maximum number of junctions, defaults to 1
        :param neighbor_order: neighbor order, defaults to 1
        :param internal: flag for internal parameter, defaults to False
        :param law: link constitutive law, optional
        """
        super().__init__()

        self.check_inputs(target_distance=target_distance,
                          max_distance=max_distance,
                          max_junctions=max_junctions,
                          neighbor_order=neighbor_order)

        self.spec_dict = {"type1": _type1,
                          "type2": _type2,
                          "lambda_fpp": lambda_fpp,
                          "activation_energy": activation_energy,
                          "target_distance": target_distance,
                          "max_distance": max_distance,
                          "max_junctions": max_junctions,
                          "neighbor_order": neighbor_order,
                          "internal": internal,
                          "law": law}

    type1: str = SpecProperty(name="type1")
    """first cell type name"""

    type2: str = SpecProperty(name="type2")
    """second cell type name"""

    lambda_fpp: float = SpecProperty(name="lambda_fpp")
    """lambda value"""

    activation_energy: float = SpecProperty(name="activation_energy")
    """activation energy"""

    target_distance: float = SpecProperty(name="target_distance")
    """target distance"""

    max_distance: float = SpecProperty(name="max_distance")
    """maximum distance"""

    max_junctions: int = SpecProperty(name="max_junctions")
    """maximum number of junction"""

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    internal: bool = SpecProperty(name="internal")
    """internal parameter flag"""

    law: Optional[LinkConstituentLaw] = SpecProperty(name="law")
    """link constitute law, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        if self.internal:
            n = "InternalParameters"
        else:
            n = "Parameters"
        el = ElementCC3D(n, {"Type1": self.type1, "Type2": self.type2})
        el.ElementCC3D("Lambda", {}, str(self.lambda_fpp))
        el.ElementCC3D("ActivationEnergy", {}, str(self.activation_energy))
        el.ElementCC3D("TargetDistance", {}, str(self.target_distance))
        el.ElementCC3D("MaxDistance", {}, str(self.max_distance))
        el.ElementCC3D("MaxNumberOfJunctions", {"NeighborOrder": str(self.neighbor_order)}, str(self.max_junctions))
        if self.law is not None:
            el.add_child(self.law.xml)
        self._el = el
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
                CoreSpecsValidator.validate_cell_type_names(type_names=[self.type1, self.type2], cell_type_spec=s)


class FocalPointPlasticityPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    FocalPointPlasticity Plugin

    FocalPointPlasticityParameters instances can be accessed as follows,

    .. code-block:: python

       spec: FocalPointPlasticityPlugin
       cell_type_1: str
       cell_type_2: str
       param: FocalPointPlasticityParameters = spec.link[cell_type_1][cell_type_2]

    """

    name = "focal_point_plasticity"
    registered_name = "FocalPointPlasticity"

    check_dict = {"neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")}

    def __init__(self, neighbor_order: int = 1, *_params):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for _p in _params:
                with err_ctx.ctx:
                    if not isinstance(_p, FocalPointPlasticityParameters):
                        raise SpecValueError("Only FocalPointPlasticityParameters instances can be passed",
                                             names=[_p.__name__])

        self.spec_dict = {"neighbor_order": neighbor_order,
                          "param_specs": {}}
        [self.params_append(_ps) for _ps in _params]

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        for t1 in self.spec_dict["param_specs"].keys():
            for t2 in self.spec_dict["param_specs"][t1].keys():
                self._el.add_child(self.spec_dict["param_specs"][t1][t2].xml)
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
                for t1 in self.spec_dict["param_specs"].keys():
                    CoreSpecsValidator.validate_cell_type_names(
                        type_names=[t1] + list(self.spec_dict["param_specs"][t1].keys()), cell_type_spec=s)

        # Validate parameters
        for param_dict in self.spec_dict["param_specs"].values():
            [param.validate(*specs) for param in param_dict.values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: FocalPointPlasticityPlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        if el.findElement("NeighborOrder"):
            o.neighbor_order = el.getFirstElement("NeighborOrder").getInt()

        el_list = CC3DXMLListPy(el.getElements("Parameters"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            p = FocalPointPlasticityParameters(p_el.getAttribute("Type1"),
                                               p_el.getAttribute("Type2"),
                                               lambda_fpp=p_el.getFirstElement("Lambda").getDouble(),
                                               activation_energy=p_el.getFirstElement("ActivationEnergy").getDouble(),
                                               target_distance=p_el.getFirstElement("TargetDistance").getDouble(),
                                               max_distance=p_el.getFirstElement("MaxDistance").getDouble(),
                                               max_junctions=p_el.getFirstElement("MaxNumberOfJunctions").getInt())
            j_el: CC3DXMLElement = p_el.getFirstElement("MaxNumberOfJunctions")
            if j_el.findAttribute("NeighborOrder"):
                p.neighbor_order = j_el.getAttributeAsInt("NeighborOrder")
            if p_el.findElement("LinkConstituentLaw"):
                l_el: CC3DXMLElement = p_el.getFirstElement("LinkConstituentLaw")
                law = LinkConstituentLaw(formula=l_el.getFirstElement("Formula").getText())

                l_el_list = CC3DXMLListPy(l_el.getElements("Variable"))

                for v_el in l_el_list:
                    v_el: CC3DXMLElement
                    law.variable[v_el.getAttribute("Name")] = v_el.getAttributeAsDouble("Value")
                p.law = law

            o.params_append(p)

        el_list = CC3DXMLListPy(el.getElements("InternalParameters"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            p = FocalPointPlasticityParameters(p_el.getAttribute("Type1"),
                                               p_el.getAttribute("Type2"),
                                               lambda_fpp=p_el.getFirstElement("Lambda").getDouble(),
                                               activation_energy=p_el.getFirstElement("ActivationEnergy").getDouble(),
                                               target_distance=p_el.getFirstElement("TargetDistance").getDouble(),
                                               max_distance=p_el.getFirstElement("MaxDistance").getDouble(),
                                               max_junctions=p_el.getFirstElement("MaxNumberOfJunctions").getInt(),
                                               internal=True)
            j_el: CC3DXMLElement = p_el.getFirstElement("MaxNumberOfJunctions")
            if j_el.findAttribute("NeighborOrder"):
                p.neighbor_order = j_el.getAttributeAsInt("NeighborOrder")
            if p_el.findElement("LinkConstituentLaw"):
                l_el: CC3DXMLElement = p_el.getFirstElement("LinkConstituentLaw")
                law = LinkConstituentLaw(formula=l_el.getFirstElement("Formula").getText())

                l_el_list = CC3DXMLListPy(l_el.getElements("Variable"))

                for v_el in l_el_list:
                    v_el: CC3DXMLElement
                    law.variable[v_el.getAttribute("Name")] = v_el.getAttributeAsDouble("Value")
                p.law = law

            o.params_append(p)

        return o

    @property
    def links(self) -> Tuple[str, str]:
        """

        :return: next tuple of current fpp parameter cell type name combinations
        """
        for v in self.spec_dict["param_specs"].values():
            for vv in v.values():
                yield vv.type1, vv.type2

    @property
    def link(self) -> _PyCoreParamAccessor[Dict[str, FocalPointPlasticityParameters]]:
        """

        :return: accessor to focal point plasticity parameters with cell types as keys
        """
        return _PyCoreParamAccessor(self, "param_specs")

    def params_append(self, _ps: FocalPointPlasticityParameters) -> None:
        """
        Append a focal point plasticity parameter spec

        :param _ps: focal point plasticity parameter spec
        :return: None
        """
        if _ps.type1 in self.spec_dict["param_specs"].keys() \
                and _ps.type2 in self.spec_dict["param_specs"][_ps.type1].keys():
            raise SpecValueError(f"Parameter already specified for types ({_ps.type1}, {_ps.type2})")
        if _ps.type1 not in self.spec_dict["param_specs"].keys():
            self.spec_dict["param_specs"][_ps.type1] = dict()
        self.spec_dict["param_specs"][_ps.type1][_ps.type2] = _ps

    def params_remove(self, type_1: str, type_2: str) -> None:
        """
        Remove a focal point plasticity parameter spec

        :param type_1: name of first cell type
        :param type_2: name of second cell type
        :return: None
        """
        if type_1 not in self.spec_dict["param_specs"].keys() \
                or type_2 not in self.spec_dict["param_specs"][type_1].keys():
            raise SpecValueError(f"Parameter not specified for types ({type_1}, {type_2})")
        self.spec_dict["param_specs"][type_1].pop(type_2)

    def params_new(self,
                   _type1: str,
                   _type2: str,
                   lambda_fpp: float,
                   activation_energy: float,
                   target_distance: float,
                   max_distance: float,
                   max_junctions: int = 1,
                   neighbor_order: int = 1,
                   internal: bool = False,
                   law: LinkConstituentLaw = None) -> FocalPointPlasticityParameters:
        """
        Append and return a new focal point plasticity parameter spec

        :param _type1: second type name
        :param _type2: first type name
        :param lambda_fpp: lambda value
        :param activation_energy: activation energy
        :param target_distance: target distance
        :param max_distance: max distance
        :param max_junctions: maximum number of junctions, defaults to 1
        :param neighbor_order: neighbor order, defaults to 1
        :param internal: flag for internal parameter, defaults to False
        :param law: link constitutive law, optional
        :return: new focal point plasticity parameter spec
        """
        p = FocalPointPlasticityParameters(_type1,
                                           _type2,
                                           lambda_fpp=lambda_fpp,
                                           activation_energy=activation_energy,
                                           target_distance=target_distance,
                                           max_distance=max_distance,
                                           max_junctions=max_junctions,
                                           neighbor_order=neighbor_order,
                                           internal=internal,
                                           law=law)
        self.params_append(p)
        return p


# Curvature Plugin


class CurvatureInternalParameters(_PyCoreSpecsBase):
    """ Curvature Internal Parameter Spec """

    def __init__(self,
                 _type1: str,
                 _type2: str,
                 _lambda_curve: float,
                 _activation_energy: float):
        """

        :param _type1: name of first cell type
        :param _type2: name of second cell type
        :param _lambda_curve: lambda value
        :param _activation_energy: activation energy
        """
        super().__init__()

        self.spec_dict = {"type1": _type1,
                          "type2": _type2,
                          "lambda_curve": _lambda_curve,
                          "activation_energy": _activation_energy}

    type1: str = SpecProperty(name="type1")
    """name of first cell type"""

    type2: str = SpecProperty(name="type2")
    """name of second cell type"""

    lambda_curve: float = SpecProperty(name="lambda_curve")
    """lambda value"""

    activation_energy: float = SpecProperty(name="activation_energy")
    """activation energy"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = ElementCC3D("InternalParameters", {"Type1": self.type1, "Type2": self.type2})
        self._el.ElementCC3D("Lambda", {}, str(self.lambda_curve))
        self._el.ElementCC3D("ActivationEnergy", {}, str(self.activation_energy))
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
                CoreSpecsValidator.validate_cell_type_names(type_names=[self.type1, self.type2], cell_type_spec=s)


class CurvatureInternalTypeParameters(_PyCoreSpecsBase):
    """ CurvatureInternalTypeParameters """

    check_dict = {
        "max_junctions": (lambda x: x < 1, "Maximum number of junctions must be positive"),
        "neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")
    }

    def __init__(self,
                 _cell_type: str,
                 _max_junctions: int,
                 _neighbor_order: int):
        """

        :param _cell_type: name of cell type
        :param _max_junctions: maximum number of junctions
        :param _neighbor_order: neighbor order
        """
        super().__init__()

        self.check_inputs(max_junctions=_max_junctions,
                          neighbor_order=_neighbor_order)

        self.spec_dict = {"cell_type": _cell_type,
                          "max_junctions": _max_junctions,
                          "neighbor_order": _neighbor_order}

    cell_type: str = SpecProperty(name="cell_type")
    """name of cell type"""

    max_junctions: int = SpecProperty(name="max_junctions")
    """maximum number of junctions"""

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = ElementCC3D("Parameters",
                               {"TypeName": self.cell_type,
                                "MaxNumberOfJunctions": self.max_junctions,
                                "NeighborOrder": self.neighbor_order})
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


class CurvaturePlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface):
    """
    Curvature Plugin

    CurvatureInternalParameters instances can be accessed as follows,

    .. code-block:: python

       spec: CurvaturePlugin
       cell_type_1: str
       cell_type_2: str
       param: CurvatureInternalParameters = spec.internal_param[cell_type_1][cell_type_2]

    CurvatureInternalTypeParameters instances can be accessed as follows,

    .. code-block:: python

       spec: CurvaturePlugin
       cell_type_name: str
       param: CurvatureInternalTypeParameters = spec.type_param[cell_type_name]

    """

    name = "curvature"
    registered_name = "Curvature"

    def __init__(self):
        super().__init__()

        self.spec_dict = {"param_specs": dict(),
                          "type_spec": dict()}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        for v in self.spec_dict["param_specs"].values():
            [self._el.add_child(vv.xml) for vv in v.values()]
        if len(self.spec_dict["type_spec"].keys()) > 0:
            el = self._el.ElementCC3D("InternalTypeSpecificParameters")
            [el.add_child(x.xml) for x in self.spec_dict["type_spec"].values()]
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
                for t1, p_dict in self.spec_dict["param_specs"].items():
                    CoreSpecsValidator.validate_cell_type_names(type_names=[t1] + list(p_dict.keys()),
                                                                cell_type_spec=s)

                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["type_spec"].keys(),
                                                            cell_type_spec=s)

        # Validate parameters
        for param_dict in self.spec_dict["param_specs"].values():
            [param.validate(*specs) for param in param_dict.values()]
        [param.validate(*specs) for param in self.spec_dict["type_spec"].values()]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: CurvaturePlugin
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = CC3DXMLListPy(el.getElements("InternalParameters"))

        for p_el in el_list:
            p_el: CC3DXMLElement
            o.params_internal_new(p_el.getAttribute("Type1"),
                                  p_el.getAttribute("Type2"),
                                  p_el.getFirstElement("Lambda").getDouble(),
                                  p_el.getFirstElement("ActivationEnergy").getDouble())

        if el.findElement("InternalTypeSpecificParameters"):
            t_el: CC3DXMLElement = el.getFirstElement("InternalTypeSpecificParameters")

            t_el_list = CC3DXMLListPy(t_el.getElements("Parameters"))

            for pt_el in t_el_list:
                pt_el: CC3DXMLElement
                o.params_type_new(pt_el.getAttribute("TypeName"),
                                  pt_el.getAttributeAsInt("MaxNumberOfJunctions"),
                                  pt_el.getAttributeAsInt("NeighborOrder"))

        return o

    @property
    def internal_param(self) -> _PyCoreParamAccessor[Dict[str, CurvatureInternalParameters]]:
        """

        :return: accessor to curvature parameters with cell types as keys
        """
        return _PyCoreParamAccessor(self, "param_specs")

    @property
    def type_param(self) -> _PyCoreParamAccessor[CurvatureInternalTypeParameters]:
        """

        :return: accessor to type parameters with cell types as keys
        """
        return _PyCoreParamAccessor(self, "type_spec")

    @property
    def internal_params(self) -> Tuple[str, str]:
        """

        :return: next tuple of current internal parameter cell type name combinations
        """
        for v in self.spec_dict["param_specs"].values():
            for vv in v.values():
                yield vv.type1, vv.type2

    @property
    def type_params(self) -> str:
        """

        :return: next current type parameters cell type names
        """
        for x in self.spec_dict["type_spec"].values():
            yield x.cell_type

    def params_internal_append(self, _ps: CurvatureInternalParameters) -> None:
        """
        Append a curvature internal parameter spec

        :param _ps: curvature internal parameter spec
        :return: None
        """
        if _ps.type1 in self.spec_dict["param_specs"].keys() \
                and _ps.type2 in self.spec_dict["param_specs"][_ps.type1].keys():
            raise SpecValueError(f"Parameter already specified for types ({_ps.type1}, {_ps.type2})")
        if _ps.type1 not in self.spec_dict["param_specs"].keys():
            self.spec_dict["param_specs"][_ps.type1] = dict()
        self.spec_dict["param_specs"][_ps.type1][_ps.type2] = _ps

    def params_internal_remove(self, _type1: str, _type2: str) -> None:
        """
        Remove a curvature internal parameter spec

        :param _type1: name of fist cell type
        :param _type2: name of second cell type
        :return: None
        """
        if _type1 not in self.spec_dict["param_specs"].keys() \
                or _type2 not in self.spec_dict["param_specs"][_type1].keys():
            raise SpecValueError(f"Parameter not specified for types ({_type1}, {_type2})")
        self.spec_dict["param_specs"][_type1].pop(_type2)

    def params_internal_new(self,
                            _type1: str,
                            _type2: str,
                            _lambda_curve: float,
                            _activation_energy: float) -> CurvatureInternalParameters:
        """
        Append and return a new curvature internal parameter spec

        :param _type1: name of first cell type
        :param _type2: name of second cell type
        :param _lambda_curve: lambda value
        :param _activation_energy: activation energy
        :return: new curvature internal parameter spec
        """
        p = CurvatureInternalParameters(_type1, _type2, _lambda_curve, _activation_energy)
        self.params_internal_append(p)
        return p

    def params_type_append(self, _ps: CurvatureInternalTypeParameters) -> None:
        """
        Append a curvature internal type parameters

        :param _ps: curvature internal type parameters
        :return: None
        """
        if _ps.cell_type in self.spec_dict["type_spec"].keys():
            raise SpecValueError(f"Parameter already specified for type {_ps.cell_type}")
        self.spec_dict["type_spec"][_ps.cell_type] = _ps

    def params_type_remove(self, _cell_type: str) -> None:
        """
        Remove a curvature internal type parameters

        :param _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.spec_dict["type_spec"].keys():
            raise SpecValueError(f"Parameter not specified for type {_cell_type}")
        self.spec_dict["type_spec"].pop(_cell_type)

    def params_type_new(self,
                        _cell_type: str,
                        _max_junctions: int,
                        _neighbor_order: int) -> CurvatureInternalTypeParameters:
        """
        Append and return a new curvature internal type parameters

        :param _cell_type: name of cell type
        :param _max_junctions: maximum number of junctions
        :param _neighbor_order: neighbor order
        :return: new curvature internal type parameters
        """
        p = CurvatureInternalTypeParameters(_cell_type, _max_junctions, _neighbor_order)
        self.params_type_append(p)
        return p


class BoundaryPixelTrackerPlugin(_PyCorePluginSpecs):
    """ BoundaryPixelTracker Plugin """

    name = "boundary_pixel_tracker"
    registered_name = "BoundaryPixelTracker"

    check_dict = {"neighbor_order": (lambda x: x < 1, "Neighbor order must be positive")}

    def __init__(self, neighbor_order: int = 1):
        super().__init__()

        self.spec_dict = {"neighbor_order": neighbor_order}

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.neighbor_order > 1:
            self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: BoundaryPixelTrackerPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class PixelTrackerPlugin(_PyCorePluginSpecs):
    """ PixelTracker Plugin """

    name = "pixel_tracker"
    registered_name = "PixelTracker"

    def __init__(self, track_medium: bool = False):
        super().__init__()

        self.spec_dict = {"track_medium": track_medium}

    track_medium: bool = SpecProperty(name="track_medium")
    """flag to track the medium"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.track_medium:
            self._el.ElementCC3D("TrackMedium")
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: PixelTrackerPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class MomentOfInertiaPlugin(_PyCorePluginSpecs):
    """ MomentOfInertia Plugin """

    name = "moment_of_inertia"
    registered_name = "MomentOfInertia"

    def __init__(self):
        super().__init__()

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: MomentOfInertiaPlugin
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class BoxWatcherSteppable(_PyCoreSteppableSpecs):
    """ BoxWatcher Steppable """

    name = "box_watcher"
    registered_name = "BoxWatcher"

    def __init__(self):
        super().__init__()

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: BoxWatcherSteppable
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Blob Initializer


class BlobInitializerRegion(_PyCoreSpecsBase):
    """ BlobInitializer Region """

    check_dict = {
        "gap": (lambda x: x < 0, "Gap must be greater non-negative"),
        "width": (lambda x: x < 1, "Width must be positive"),
        "radius": (lambda x: x < 1, "Radius must be positive"),
        "center": (lambda x: x.x < 0 or x.y < 0 or x.z < 0, "Coordinates are non-negative")
    }

    def __init__(self,
                 gap: int = 0,
                 width: int = 0,
                 radius: int = 0,
                 center: Point3DLike = Point3D(0, 0, 0),
                 cell_types: List[str] = None):
        """

        :param gap: blob gap
        :param width: width of cells
        :param radius: blob radius
        :param center: blob center point
        :param cell_types: names of cell types in blob
        """
        super().__init__()

        if cell_types is None:
            cell_types = []

        self.spec_dict = {"gap": gap,
                          "width": width,
                          "radius": radius,
                          "center": _as_point3d(center),
                          "cell_types": cell_types}

    gap: int = SpecProperty(name="gap")
    """blob gap"""

    width: int = SpecProperty(name="width")
    """cell width"""

    radius: int = SpecProperty(name="radius")
    """blob radius"""

    center: Point3D = SpecProperty(name="center")
    """blob center point"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("Region")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("Gap", {}, str(self.gap))
        self._el.ElementCC3D("Width", {}, str(self.width))
        self._el.ElementCC3D("Radius", {}, str(self.radius))
        self._el.ElementCC3D("Center", {"x": str(self.center.x), "y": str(self.center.y), "z": str(self.center.z)})
        self._el.ElementCC3D("Types", {}, ",".join(self.spec_dict["cell_types"]))
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
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["cell_types"], cell_type_spec=s)

            # Validate BlobInitializerRegion shape against Potts
            if isinstance(s, PottsCore):
                for c, o in itertools.product(["x", "y", "z"], [-self.radius, self.radius]):
                    pt = Point3D(self.center)
                    setattr(pt, c, getattr(pt, c) + o)
                    CoreSpecsValidator.validate_point(pt=pt, potts_spec=s)


class BlobInitializer(_PyCoreSteppableSpecs):
    """ BlobInitializer """

    name = "blob_initializer"
    registered_name = "BlobInitializer"

    def __init__(self, *_regions):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for r in _regions:
                with err_ctx.ctx:
                    if not isinstance(r, BlobInitializerRegion):
                        raise SpecValueError("Only BlobInitializerRegion instances can be passed",
                                             names=[r.__name__])

        self.spec_dict = {"regions": list(_regions)}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(reg.xml) for reg in self.spec_dict["regions"]]
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

        # Validate regions
        [reg.validate(*specs) for reg in self.spec_dict["regions"]]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: BlobInitializer
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("Region"))

        for r_el in el_list:
            r_el: CC3DXMLElement

            kwargs = {"gap": 0,
                      "width": r_el.getFirstElement("Width").getUInt(),
                      "radius": r_el.getFirstElement("Radius").getUInt(),
                      "center": Point3D(0, 0, 0)}
            # Making Gap optional
            if r_el.findElement("Gap"):
                kwargs["gap"] = r_el.getFirstElement("Gap").getUInt()

            c_el: CC3DXMLElement = r_el.getFirstElement("Center")
            for comp in ["x", "y", "z"]:
                if c_el.findAttribute(comp):
                    setattr(kwargs["center"], comp, c_el.getAttributeAsShort(comp))

            if r_el.findElement("Types"):
                kwargs["cell_types"] = [x.replace(" ", "") for x in r_el.getFirstElement("Types").getText().split(",")]

            o.region_new(**kwargs)

        return o

    def region_append(self, _reg: BlobInitializerRegion) -> None:
        """
        Append a region

        :param _reg: a region
        :return: None
        """
        self.spec_dict["regions"].append(_reg)

    def region_pop(self, _idx: int) -> None:
        """
        Remove a region by index

        :param _idx: index of region to append
        :return: None
        """
        self.spec_dict["regions"].pop(_idx)

    def region_new(self,
                   gap: int = 0,
                   width: int = 0,
                   radius: int = 0,
                   center: Point3DLike = Point3D(0, 0, 0),
                   cell_types: List[str] = None) -> BlobInitializerRegion:
        """
        Appends and returns a blob region

        :param gap: blob gap
        :param width: width of cells
        :param radius: blob radius
        :param center: blob center point
        :param cell_types: names of cell types in blob
        :return: new blob region
        """
        reg = BlobInitializerRegion(gap=gap, width=width, radius=radius, center=center, cell_types=cell_types)
        self.region_append(reg)
        return reg


# Uniform Initializer


class UniformInitializerRegion(_PyCoreSpecsBase):
    """ Uniform Initializer Region Specs """

    check_dict = {
        "gap": (lambda x: x < 0, "Gap must be greater non-negative"),
        "width": (lambda x: x < 1, "Width must be positive")
    }

    def __init__(self,
                 pt_min: Point3DLike,
                 pt_max: Point3DLike,
                 gap: int = 0,
                 width: int = 0,
                 cell_types: List[str] = None):
        """

        :param pt_min: minimum box point
        :param pt_max: maximum box point
        :param gap: blob gap
        :param width: width of cells
        :param cell_types: names of cell types in region
        """
        super().__init__()

        if cell_types is None:
            cell_types = []

        self.spec_dict = {"pt_min": _as_point3d(pt_min),
                          "pt_max": _as_point3d(pt_max),
                          "gap": gap,
                          "width": width,
                          "cell_types": cell_types}

    pt_min: Point3D = SpecProperty(name="pt_min")
    """minimum box point"""

    pt_max: Point3D = SpecProperty(name="pt_max")
    """maximum box point"""

    gap: int = SpecProperty(name="gap")
    """blob gap"""

    width: int = SpecProperty(name="width")
    """cell width"""

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("Region")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("BoxMin", {c: str(getattr(self.pt_min, c)) for c in ["x", "y", "z"]})
        self._el.ElementCC3D("BoxMax", {c: str(getattr(self.pt_max, c)) for c in ["x", "y", "z"]})
        self._el.ElementCC3D("Gap", {}, str(self.gap))
        self._el.ElementCC3D("Width", {}, str(self.width))
        self._el.ElementCC3D("Types", {}, ",".join(self.spec_dict["cell_types"]))
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
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["cell_types"],
                                                            cell_type_spec=s)

            # Validate BlobInitializerRegion shape against Potts
            if isinstance(s, PottsCore):
                [CoreSpecsValidator.validate_point(pt=pt, potts_spec=s) for pt in [self.pt_min, self.pt_max]]


class UniformInitializer(_PyCoreSteppableSpecs):
    """ Uniform Initializer Specs """

    name = "uniform_initializer"
    registered_name = "UniformInitializer"

    def __init__(self, *_regions):
        super().__init__()

        with _SpecValueErrorContextBlock() as err_ctx:
            for r in _regions:
                with err_ctx.ctx:
                    if not isinstance(r, UniformInitializerRegion):
                        raise SpecValueError("Only UniformInitializerRegion instances can be passed",
                                             names=[r.__name__])

        self.spec_dict = {"regions": list(_regions)}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(reg.xml) for reg in self.spec_dict["regions"]]
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

        # Validate regions
        [reg.validate(*specs) for reg in self.spec_dict["regions"]]

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: UniformInitializer
        """

        o = cls()

        el_list = CC3DXMLListPy(cls.find_xml_by_attr(_xml).getElements("Region"))

        for r_el in el_list:
            r_el: CC3DXMLElement

            min_el: CC3DXMLElement = r_el.getFirstElement("BoxMin")
            max_el: CC3DXMLElement = r_el.getFirstElement("BoxMax")
            pt_min, pt_max = Point3D(), Point3D()
            for c in ["x", "y", "z"]:
                if min_el.findAttribute(c):
                    setattr(pt_min, c, min_el.getAttributeAsInt(c))
                if max_el.findAttribute(c):
                    setattr(pt_max, c, max_el.getAttributeAsInt(c))

            kwargs = {"pt_min": pt_min,
                      "pt_max": pt_max,
                      "gap": 0,
                      "width": r_el.getFirstElement("Width").getUInt()}
            # Making Gap optional
            if r_el.findElement("Gap"):
                kwargs["gap"] = r_el.getFirstElement("Gap").getUInt()

            if r_el.findElement("Types"):
                kwargs["cell_types"] = [x.replace(" ", "") for x in r_el.getFirstElement("Types").getText().split(",")]

            o.region_new(**kwargs)

        return o

    def region_append(self, _reg: UniformInitializerRegion) -> None:
        """
        Append a region

        :param _reg: a region
        :return: None
        """
        self.spec_dict["regions"].append(_reg)

    def region_pop(self, _idx: int) -> None:
        """
        Remove a region by index

        :param _idx: index of region to append
        :return: None
        """
        self.spec_dict["regions"].pop(_idx)

    def region_new(self,
                   pt_min: Point3DLike,
                   pt_max: Point3DLike,
                   gap: int = 0,
                   width: int = 0,
                   cell_types: List[str] = None) -> UniformInitializerRegion:
        """
        Appends and returns a uniform region

        :param pt_min: minimum box point
        :param pt_max: maximum box point
        :param gap: blob gap
        :param width: width of cells
        :param cell_types: names of cell types in region
        :return: new blob region
        """
        reg = UniformInitializerRegion(pt_min=pt_min, pt_max=pt_max, gap=gap, width=width, cell_types=cell_types)
        self.region_append(reg)
        return reg


class PIFInitializer(_PyCoreSteppableSpecs):
    """ PIF Initializer """

    name = "pif_initiazer"
    registered_name = "PIFInitializer"

    check_dict = {"pif_name": (lambda x: len(x) == 0, "PIF needs a name")}

    def __init__(self, pif_name: str):
        super().__init__()

        self.check_inputs(pif_name=pif_name)

        self.spec_dict = {"pif_name": pif_name}

    pif_name: str = SpecProperty(name="pif_name")
    """name of pif file"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("PIFName", {}, self.pif_name)
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

        # Validate file existence
        if not os.path.isfile(self.pif_name):
            raise SpecValueError("Could not validate pif file existence:", self.pif_name)

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: PIFInitializer
        """
        return cls(pif_name=cls.find_xml_by_attr(_xml).getFirstElement("PIFName").getText())


class PIFDumperSteppable(_PyCoreSteppableSpecs):
    """ PIFDumper Steppable """

    name = "pif_dumper"
    registered_name = "PIFDumper"

    check_dict = {
        "pif_name": (lambda x: len(x) == 0, "PIF needs a name"),
        "frequency": (lambda x: x < 1, "Frequency must be positive")
    }

    def __init__(self, pif_name: str, frequency: int = 1):
        super().__init__()

        self.check_inputs(pif_name=pif_name, frequency=frequency)

        self.spec_dict = {"pif_name": pif_name,
                          "frequency": frequency}

    pif_name: str = SpecProperty(name="pif_name")
    """name of pif file"""

    frequency: int = SpecProperty(name="frequency")
    """frequency of output"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("PIFName", {}, self.pif_name)
        return self._el

    def __getstate__(self) -> Tuple[str, tuple, dict]:
        return self.xml.getCC3DXMLElementString(), tuple(), {"pif_name": self.pif_name}

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: PIFDumperSteppable
        """
        el = cls.find_xml_by_attr(_xml)
        kwargs = {"pif_name": el.getFirstElement("PIFName").getText()}
        if el.findAttribute("Frequency"):
            kwargs["frequency"] = el.getAttributeAsInt("Frequency")
        return cls(**kwargs)


# DiffusionSolverFE


class DiffusionSolverFEDiffusionData(_PDEDiffusionDataSpecs):
    """ Diffusion Data Specs for DiffusionSolverFE """

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    init_expression: float = SpecProperty(name="init_expression")
    """expression of initial field distribution, Optional, None if not set"""

    init_filename: float = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("FieldName", {}, self.field_name)
        self._el.ElementCC3D("GlobalDiffusionConstant", {}, str(self.diff_global))
        self._el.ElementCC3D("GlobalDecayConstant", {}, str(self.decay_global))
        for type_name, val in self.spec_dict["diff_types"].items():
            self._el.ElementCC3D("DiffusionCoefficient", {"CellType": type_name}, str(val))
        for type_name, val in self.spec_dict["decay_types"].items():
            self._el.ElementCC3D("DecayCoefficient", {"CellType": type_name}, str(val))
        if self.init_expression:
            self._el.ElementCC3D("InitialConcentrationExpression", {}, self.init_expression)
        if self.init_filename:
            self._el.ElementCC3D("ConcentrationFileName", {}, self.init_filename)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [DiffusionSolverFE]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        solver = DiffusionSolverFE

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["diff_types"].keys(),
                                                            cell_type_spec=s)
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["decay_types"].keys(),
                                                            cell_type_spec=s)

            if isinstance(s, solver):
                # Validate field name against solver
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    @property
    def diff_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to diffusion coefficient values by cell type name
        """
        return _PyCoreParamAccessor(self, "diff_types")

    @property
    def decay_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to decay coefficient values by cell type name
        """
        return _PyCoreParamAccessor(self, "decay_types")


class DiffusionSolverFESecretionData(_PDESecretionDataSpecs):
    """ Secretion Data Specs for DiffusionSolverFE """

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [DiffusionSolverFE]


class DiffusionSolverFEField(_PDESolverFieldSpecs[DiffusionSolverFEDiffusionData, DiffusionSolverFESecretionData]):
    """ DiffusionSolverFE Field Specs """

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [DiffusionSolverFE]

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

        solver = DiffusionSolverFE

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate field name with solver (diffusion data tests for solver)
            if isinstance(s, solver):
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> SecretionParameters:
        """
        Specify and return a new secretion data spec

        :param _cell_type: name of cell type
        :param _val: value
        :param kwargs:
        :return: new secretion data spec
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        try:
            constant = kwargs["constant"]
        except KeyError:
            constant = False
        return _PDESolverFieldSpecs.secretion_data_new(
            self, _cell_type, _val, constant=constant, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class DiffusionSolverFE(_PDESolverSpecs[DiffusionSolverFEDiffusionData, DiffusionSolverFESecretionData]):
    """ DiffusionSolverFE """

    name = "diffusion_solver_fe"

    _field_spec = DiffusionSolverFEField
    _diff_data = DiffusionSolverFEDiffusionData
    _secr_data = DiffusionSolverFESecretionData

    def __init__(self):
        super().__init__()

        self.spec_dict["gpu"] = False

    fluc_comp: bool = SpecProperty(name="fluc_comp")
    """flag to use a FluctuationCompensator"""

    gpu: bool = SpecProperty(name="gpu")
    """flag to use GPU acceleration"""

    @property
    def registered_name(self) -> str:
        """

        :return: name according to core
        """
        if self.gpu:
            return "DiffusionSolverFE_OpenCL"
        return "DiffusionSolverFE"

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("Steppable", {"Type": self.registered_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.fluc_comp:
            self._el.ElementCC3D("FluctuationCompensator")
        [self._el.add_child(_fs.xml) for _fs in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: DiffusionSolverFE
        """
        o = cls()
        try:
            el = o.find_xml_by_attr(_xml, registered_name=o.registered_name)
        except SpecImportError:
            o.gpu = not o.gpu
            el = o.find_xml_by_attr(_xml, registered_name=o.registered_name)

        o.fluc_comp = el.findElement("FluctuationCompensator")

        el_list = CC3DXMLListPy(el.getElements("DiffusionField"))

        for f_el in el_list:
            f_el: CC3DXMLElement

            # Handling ambiguous field naming procedure
            # DiffusionField attribute Name takes precedence over DiffusionData element FieldName
            field_name = None
            if f_el.findAttribute("Name"):
                field_name = f_el.getAttribute("Name")

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if field_name is None:
                if not dd_el.findElement("FieldName"):
                    raise SpecImportError("Unknown field name")
                field_name = dd_el.getFirstElement("FieldName").getText()

            f = o.field_new(field_name)

            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()

            el_list = CC3DXMLListPy(dd_el.getElements("DiffusionCoefficient"))

            for t_el in el_list:
                f.diff_data.diff_types[t_el.getAttribute("CellType")] = t_el.getDouble()

            el_list = CC3DXMLListPy(dd_el.getElements("DecayCoefficient"))

            for t_el in el_list:
                f.diff_data.decay_types[t_el.getAttribute("CellType")] = t_el.getDouble()
            if dd_el.findElement("InitialConcentrationExpression"):
                f.diff_data.init_expression = dd_el.getFirstElement("InitialConcentrationExpression").getText()
            if dd_el.findElement("ConcentrationFileName"):
                f.diff_data.init_filename = dd_el.getFirstElement("ConcentrationFileName").getText()

            if el.findElement("SecretionData"):
                sd_el: CC3DXMLElement = el.getFirstElement("SecretionData")
                p_el: CC3DXMLElement

                sd_el_list = CC3DXMLListPy(sd_el.getElements("Secretion"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = CC3DXMLListPy(sd_el.getElements("ConstantConcentration"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = CC3DXMLListPy(sd_el.getElements("SecretionOnContact"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            if b_el is not None:
                b_el_list = CC3DXMLListPy(b_el.getElements("Plane"))
                for p_el in b_el_list:
                    p_el: CC3DXMLElement
                    axis: str = p_el.getAttribute("Axis").lower()
                    if p_el.findElement("Periodic"):
                        setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                    else:
                        c_el: CC3DXMLElement
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantValue"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYVALUE)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantDerivative"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYFLUX)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


# KernelDiffusionSolver


class KernelDiffusionSolverDiffusionData(_PDEDiffusionDataSpecs):
    """ KernelDiffusionSolver Diffusion Data Specs"""

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    init_expression: Optional[str] = SpecProperty(name="init_expression")
    """expression of initial field distribution, Optional, None if not set"""

    init_filename: Optional[str] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("FieldName", {}, self.field_name)
        self._el.ElementCC3D("GlobalDiffusionConstant", {}, str(self.diff_global))
        self._el.ElementCC3D("GlobalDecayConstant", {}, str(self.decay_global))
        if self.init_expression:
            self._el.ElementCC3D("InitialConcentrationExpression", {}, self.init_expression)
        if self.init_filename:
            self._el.ElementCC3D("ConcentrationFileName", {}, self.init_filename)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [KernelDiffusionSolver]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        solver = KernelDiffusionSolver

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            if isinstance(s, solver):
                # Validate field name against solver
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)


class KernelDiffusionSolverSecretionData(_PDESecretionDataSpecs):
    """ KernelDiffusionSolver Secretion Data Specs """

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [KernelDiffusionSolver]


class KernelDiffusionSolverBoundaryConditions(PDEBoundaryConditions):
    """
    KernelDiffusionSolver Boundary Condition Specs

    Simple derived class to enforce KernelDiffusionSolver constraints of periodic boundary conditions
    """

    check_dict = {
        "x_min_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions"),
        "x_max_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions"),
        "y_min_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions"),
        "y_max_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions"),
        "z_min_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions"),
        "z_max_type": (lambda x: x != BOUNDARYTYPESPDE[2],
                       "KernelDiffusionSolver only supports periodic boundary conditions")
    }

    def __init__(self):
        super().__init__(x_min_type=BOUNDARYTYPESPDE[2],
                         x_max_type=BOUNDARYTYPESPDE[2],
                         y_min_type=BOUNDARYTYPESPDE[2],
                         y_max_type=BOUNDARYTYPESPDE[2],
                         z_min_type=BOUNDARYTYPESPDE[2],
                         z_max_type=BOUNDARYTYPESPDE[2])

        self.spec_dict["x_min_val"] = None
        self.spec_dict["x_max_val"] = None
        self.spec_dict["y_min_val"] = None
        self.spec_dict["y_max_val"] = None
        self.spec_dict["z_min_val"] = None
        self.spec_dict["z_max_val"] = None

    x_min_type: str = SpecProperty(name="x_min_type", readonly=True)
    """boundary condition type along lower x-orthogonal boundary; read-only"""

    x_max_type: str = SpecProperty(name="x_max_type", readonly=True)
    """boundary condition type along upper x-orthogonal boundary; read-only"""

    y_min_type: str = SpecProperty(name="y_min_type", readonly=True)
    """boundary condition type along lower y-orthogonal boundary; read-only"""

    y_max_type: str = SpecProperty(name="y_max_type", readonly=True)
    """boundary condition type along upper y-orthogonal boundary; read-only"""

    z_min_type: str = SpecProperty(name="z_min_type", readonly=True)
    """boundary condition type along lower z-orthogonal boundary; read-only"""

    z_max_type: str = SpecProperty(name="z_max_type", readonly=True)
    """boundary condition type along upper z-orthogonal boundary; read-only"""


class KernelDiffusionSolverField(_PDESolverFieldSpecs[KernelDiffusionSolverDiffusionData,
                                                      KernelDiffusionSolverSecretionData]):
    """ KernelDiffusionSolver Field Specs """

    check_dict = {
        "kernel": (lambda x: x < 1, "Kernal must be positive"),
        "cgfactor": (lambda x: x < 1, "Coarse grain factor must be positive")
    }

    def __init__(self,
                 field_name: str,
                 diff_data=KernelDiffusionSolverDiffusionData,
                 secr_data=KernelDiffusionSolverSecretionData):
        super().__init__(field_name=field_name,
                         diff_data=diff_data,
                         secr_data=secr_data)

        self.spec_dict["bc_specs"] = KernelDiffusionSolverBoundaryConditions()
        self.spec_dict["kernel"] = 1
        self.spec_dict["cgfactor"] = 1

    kernel: int = SpecProperty(name="kernel")
    """kernel of diffusion solver"""

    cgfactor: int = SpecProperty(name="cgfactor")
    """coarse grain factor"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.kernel > 1:
            self._el.ElementCC3D("Kernel", {}, str(self.kernel))
        if self.cgfactor > 1:
            self._el.ElementCC3D("CoarseGrainFactor", {}, str(self.cgfactor))
        self._el.add_child(self.spec_dict["diff_data"].xml)
        self._el.add_child(self.spec_dict["secr_data"].xml)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [KernelDiffusionSolver]

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

        solver = KernelDiffusionSolver

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate field name with solver (diffusion data tests for solver)
            if isinstance(s, solver):
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> SecretionParameters:
        """
        Specify and return a new secretion data spec

        :param _cell_type: name of cell type
        :param _val: value
        :param kwargs:
        :return: new secretion data spec
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        try:
            constant = kwargs["constant"]
        except KeyError:
            constant = False
        return _PDESolverFieldSpecs.secretion_data_new(
            self, _cell_type, _val, constant=constant, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class KernelDiffusionSolver(_PDESolverSpecs[KernelDiffusionSolverDiffusionData, KernelDiffusionSolverSecretionData]):
    """ KernelDiffusionSolver """

    name = "kernel_diffusion_solver"
    registered_name = "KernelDiffusionSolver"

    _field_spec = KernelDiffusionSolverField
    _diff_data = KernelDiffusionSolverDiffusionData
    _secr_data = KernelDiffusionSolverSecretionData

    def __init__(self):
        super().__init__()

        self.steer = None  # Not currently steerable

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(f.xml) for f in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: KernelDiffusionSolver
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = CC3DXMLListPy(el.getElements("DiffusionField"))

        for f_el in el_list:
            f_el: CC3DXMLElement

            # Handling ambiguous field naming procedure
            # DiffusionField attribute Name takes precedence over DiffusionData element FieldName
            field_name = None
            if f_el.findAttribute("Name"):
                field_name = f_el.getAttribute("Name")

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if field_name is None:
                if not dd_el.findElement("FieldName"):
                    raise SpecImportError("Unknown field name")
                field_name = dd_el.getFirstElement("FieldName").getText()

            f = o.field_new(field_name)

            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()
            if dd_el.findElement("InitialConcentrationExpression"):
                f.diff_data.init_expression = dd_el.getFirstElement("InitialConcentrationExpression").getText()
            if dd_el.findElement("ConcentrationFileName"):
                f.diff_data.init_filename = dd_el.getFirstElement("ConcentrationFileName").getText()

            if el.findElement("SecretionData"):
                sd_el: CC3DXMLElement = el.getFirstElement("SecretionData")
                p_el: CC3DXMLElement

                sd_el_list = CC3DXMLListPy(sd_el.getElements("Secretion"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = CC3DXMLListPy(sd_el.getElements("ConstantConcentration"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = CC3DXMLListPy(sd_el.getElements("SecretionOnContact"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

        return o


# ReactionDiffusionSolverFE


class ReactionDiffusionSolverFEDiffusionData(_PDEDiffusionDataSpecs):
    """ ReactionDiffusionSolverFE Diffusion Data Specs"""

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    additional_term: str = SpecProperty(name="additional_term")
    """expression of additional term"""

    init_filename: Optional[str] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("FieldName", {}, self.field_name)
        self._el.ElementCC3D("GlobalDiffusionConstant", {}, str(self.diff_global))
        self._el.ElementCC3D("GlobalDecayConstant", {}, str(self.decay_global))
        for type_name, val in self.spec_dict["diff_types"].items():
            self._el.ElementCC3D("DiffusionCoefficient", {"CellType": type_name}, str(val))
        for type_name, val in self.spec_dict["decay_types"].items():
            self._el.ElementCC3D("DecayCoefficient", {"CellType": type_name}, str(val))
        if self.init_filename:
            self._el.ElementCC3D("ConcentrationFileName", {}, self.init_filename)
        if self.additional_term:
            self._el.ElementCC3D("AdditionalTerm", {}, self.additional_term)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [ReactionDiffusionSolverFE]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        solver = ReactionDiffusionSolverFE

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate against cell types defined in CellTypePlugin
            if isinstance(s, CellTypePlugin):
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["diff_types"].keys(),
                                                            cell_type_spec=s)
                CoreSpecsValidator.validate_cell_type_names(type_names=self.spec_dict["decay_types"].keys(),
                                                            cell_type_spec=s)

            if isinstance(s, solver):
                # Validate field name against solver
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    @property
    def diff_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to diffusion coefficient values by cell type name
        """
        return _PyCoreParamAccessor(self, "diff_types")

    @property
    def decay_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to decay coefficient values by cell type name
        """
        return _PyCoreParamAccessor(self, "decay_types")


class ReactionDiffusionSolverFESecretionData(_PDESecretionDataSpecs):
    """ ReactionDiffusionSolverFE Secretion Specs"""

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [ReactionDiffusionSolverFE]


class ReactionDiffusionSolverFEField(_PDESolverFieldSpecs[ReactionDiffusionSolverFEDiffusionData,
                                                          ReactionDiffusionSolverFESecretionData]):
    """ ReactionDiffusionSolverFE Field Specs"""

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [ReactionDiffusionSolverFE]

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

        solver = ReactionDiffusionSolverFE

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate field name with solver (diffusion data tests for solver)
            if isinstance(s, solver):
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> SecretionParameters:
        """
        Specify and return a new secretion data spec

        :param _cell_type: name of cell type
        :param _val: value
        :param kwargs:
        :raises SpecValueError: if setting constant concentration
        :return: new secretion data spec
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        if "constant" in kwargs.keys():
            raise SpecValueError("ReactionDiffusionSolverFE does not support constant concentrations")
        return _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class ReactionDiffusionSolverFE(_PDESolverSpecs[ReactionDiffusionSolverFEDiffusionData,
                                                ReactionDiffusionSolverFESecretionData]):
    """ ReactionDiffusionSolverFE """

    name = "reaction_diffusion_solver_fe"
    registered_name = "ReactionDiffusionSolverFE"

    _field_spec = ReactionDiffusionSolverFEField
    _diff_data = ReactionDiffusionSolverFEDiffusionData
    _secr_data = ReactionDiffusionSolverFESecretionData

    def __init__(self):
        super().__init__()

        self.spec_dict["autoscale"] = False

    fluc_comp: bool = SpecProperty(name="fluc_comp")
    """flag to use a FluctuationCompensator"""

    autoscale: bool = SpecProperty(name="autoscale")
    """flag to perform automatic scaling"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        if self.fluc_comp:
            self._el.ElementCC3D("FluctuationCompensator")
        if self.autoscale:
            self._el.ElementCC3D("AutoscaleDiffusion")
        [self._el.add_child(f.xml) for f in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: ReactionDiffusionSolverFE
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()
        o.autoscale = el.findElement("AutoscaleDiffusion")
        o.fluc_comp = el.findElement("FluctuationCompensator")

        el_list = CC3DXMLListPy(el.getElements("DiffusionField"))

        for f_el in el_list:
            f_el: CC3DXMLElement

            # Handling ambiguous field naming procedure
            # DiffusionField attribute Name takes precedence over DiffusionData element FieldName
            field_name = None
            if f_el.findAttribute("Name"):
                field_name = f_el.getAttribute("Name")

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if field_name is None:
                if not dd_el.findElement("FieldName"):
                    raise SpecImportError("Unknown field name")
                field_name = dd_el.getFirstElement("FieldName").getText()

            f = o.field_new(field_name)

            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()

            dd_el_list = CC3DXMLListPy(dd_el.getElements("DiffusionCoefficient"))

            for t_el in dd_el_list:
                f.diff_data.diff_types[t_el.getAttribute("CellType")] = t_el.getDouble()

            dd_el_list = CC3DXMLListPy(dd_el.getElements("DecayCoefficient"))

            for t_el in dd_el_list:
                f.diff_data.decay_types[t_el.getAttribute("CellType")] = t_el.getDouble()
            if dd_el.findElement("InitialConcentrationExpression"):
                f.diff_data.init_expression = dd_el.getFirstElement("InitialConcentrationExpression").getText()
            if dd_el.findElement("ConcentrationFileName"):
                f.diff_data.init_filename = dd_el.getFirstElement("ConcentrationFileName").getText()
            if dd_el.findElement("AdditionalTerm"):
                f.diff_data.additional_term = dd_el.getFirstElement("AdditionalTerm").getText()

            if el.findElement("SecretionData"):
                sd_el: CC3DXMLElement = el.getFirstElement("SecretionData")
                p_el: CC3DXMLElement

                sd_el_list = CC3DXMLListPy(sd_el.getElements("Secretion"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = CC3DXMLListPy(sd_el.getElements("ConstantConcentration"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = CC3DXMLListPy(sd_el.getElements("SecretionOnContact"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            if b_el is not None:
                b_el_list = CC3DXMLListPy(b_el.getElements("Plane"))
                for p_el in b_el_list:
                    p_el: CC3DXMLElement
                    axis: str = p_el.getAttribute("Axis").lower()
                    if p_el.findElement("Periodic"):
                        setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                    else:
                        c_el: CC3DXMLElement
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantValue"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYVALUE)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantDerivative"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYFLUX)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


# SteadyStateDiffusionSolver2D + SteadyStateDiffusionSolver

class SteadyStateDiffusionSolverDiffusionData(_PDEDiffusionDataSpecs):
    """ SteadyStateDiffusionSolver Diffusion Data Specs """

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    init_expression: Optional[str] = SpecProperty(name="init_expression")
    """expression of initial field distribution, Optional, None if not set"""

    init_filename: Optional[str] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("FieldName", {}, self.field_name)
        self._el.ElementCC3D("GlobalDiffusionConstant", {}, str(self.diff_global))
        self._el.ElementCC3D("GlobalDecayConstant", {}, str(self.decay_global))
        if self.init_expression:
            self._el.ElementCC3D("InitialConcentrationExpression", {}, self.init_expression)
        if self.init_filename:
            self._el.ElementCC3D("ConcentrationFileName", {}, self.init_filename)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [SteadyStateDiffusionSolver]

    def validate(self, *specs) -> None:
        """
        Validates specs against a variable number of _PyCoreSpecsBase-derived class instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :type specs: _PyCoreSpecsBase
        :raises SpecValueError: when something could not be validated
        :return: None
        """
        super().validate(*specs)

        solver = SteadyStateDiffusionSolver

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            if isinstance(s, solver):
                # Validate field name against solver
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)


class SteadyStateDiffusionSolverSecretionData(_PDESecretionDataSpecs):
    """ SteadyStateDiffusionSolver Secretion Data Specs """

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [SteadyStateDiffusionSolver]


class SteadyStateDiffusionSolverField(_PDESolverFieldSpecs[SteadyStateDiffusionSolverDiffusionData,
                                                           SteadyStateDiffusionSolverSecretionData]):
    """ SteadyStateDiffusionSolver Field Specs """

    def __init__(self,
                 field_name: str,
                 diff_data=SteadyStateDiffusionSolverDiffusionData,
                 secr_data=SteadyStateDiffusionSolverSecretionData):
        super().__init__(field_name=field_name,
                         diff_data=diff_data,
                         secr_data=secr_data)

        self.spec_dict["pymanage"] = False

    pymanage: bool = SpecProperty(name="pymanage")
    """flag to manage secretion in python"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        self._el.add_child(self.spec_dict["diff_data"].xml)
        if self.pymanage:
            self._el.ElementCC3D("ManageSecretionInPython")
        else:
            self._el.add_child(self.spec_dict["secr_data"].xml)
        self._el.add_child(self.spec_dict["bc_specs"].xml)
        return self._el

    @property
    def depends_on(self) -> List[Type]:
        """
        Returns a list of depencies, each of which must be instantiated exactly once to validate

        :return: list of dependencies
        """
        return super().depends_on + [SteadyStateDiffusionSolver]

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

        solver = SteadyStateDiffusionSolver

        # Validate field name uniqueness
        CoreSpecsValidator.validate_field_name_unique(*specs, field_name=self.field_name)

        for s in specs:
            # Validate field name with solver (diffusion data tests for solver)
            if isinstance(s, solver):
                CoreSpecsValidator.validate_field_name_unique(s, field_name=self.field_name)

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> SecretionParameters:
        """
        Specify and return a new secretion data spec

        :param _cell_type: name of cell type
        :param _val: value
        :param kwargs:
        :raises SpecValueError: if setting constant concentration or on contact with
        :return: new secretion data spec
        """
        if "contact_type" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support contact-based secretion")
        if "constant" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support constant concentrations")
        return _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        if "contact_type" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support contact-based secretion")
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type)


class SteadyStateDiffusionSolver(_PDESolverSpecs[SteadyStateDiffusionSolverDiffusionData,
                                                 SteadyStateDiffusionSolverSecretionData]):
    """ SteadyStateDiffusionSolver Specs"""

    name = "steady_state_diffusion_solver"

    _field_spec = SteadyStateDiffusionSolverField
    _diff_data = SteadyStateDiffusionSolverDiffusionData
    _secr_data = SteadyStateDiffusionSolverSecretionData

    def __init__(self):
        super().__init__()

        self.spec_dict["three_d"] = False

    three_d: bool = SpecProperty(name="three_d")
    """flag whether domain is three-dimensional"""

    @property
    def registered_name(self) -> str:
        """

        :return: name according to core
        """
        if self.three_d:
            return "SteadyStateDiffusionSolver"
        return "SteadyStateDiffusionSolver2D"

    def generate_header(self) -> Optional[ElementCC3D]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        """
        return ElementCC3D("Steppable", {"Type": self.registered_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        """
        self._el = self.generate_header()
        [self._el.add_child(f.xml) for f in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param _xml: parent xml
        :return: python class instace
        :rtype: SteadyStateDiffusionSolver
        """
        o = cls()

        try:
            el = o.find_xml_by_attr(_xml, registered_name=o.registered_name)
        except SpecImportError:
            o.three_d = not o.three_d
            el = o.find_xml_by_attr(_xml, registered_name=o.registered_name)

        el_list = CC3DXMLListPy(el.getElements("DiffusionField"))

        for f_el in el_list:
            f_el: CC3DXMLElement

            # Handling ambiguous field naming procedure
            # DiffusionField attribute Name takes precedence over DiffusionData element FieldName
            field_name = None
            if f_el.findAttribute("Name"):
                field_name = f_el.getAttribute("Name")

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if field_name is None:
                if not dd_el.findElement("FieldName"):
                    raise SpecImportError("Unknown field name")
                field_name = dd_el.getFirstElement("FieldName").getText()

            f = o.field_new(field_name)
            f.pymanage = f_el.findElement("ManageSecretionInPython")

            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()
            if dd_el.findElement("InitialConcentrationExpression"):
                f.diff_data.init_expression = dd_el.getFirstElement("InitialConcentrationExpression").getText()
            if dd_el.findElement("ConcentrationFileName"):
                f.diff_data.init_filename = dd_el.getFirstElement("ConcentrationFileName").getText()

            if el.findElement("SecretionData"):
                sd_el: CC3DXMLElement = el.getFirstElement("SecretionData")
                p_el: CC3DXMLElement

                sd_el_list = CC3DXMLListPy(sd_el.getElements("Secretion"))

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            if b_el is not None:
                b_el_list = CC3DXMLListPy(b_el.getElements("Plane"))
                for p_el in b_el_list:
                    p_el: CC3DXMLElement
                    axis: str = p_el.getAttribute("Axis").lower()
                    if p_el.findElement("Periodic"):
                        setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                    else:
                        c_el: CC3DXMLElement
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantValue"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYVALUE)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                        p_el_list = CC3DXMLListPy(p_el.getElements("ConstantDerivative"))
                        for c_el in p_el_list:
                            pos: str = c_el.getAttribute("PlanePosition").lower()
                            setattr(f.bcs, f"{axis}_{pos}_type", PDEBOUNDARYFLUX)
                            setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


PLUGINS = [
    AdhesionFlexPlugin,
    BoundaryPixelTrackerPlugin,
    CellTypePlugin,
    CenterOfMassPlugin,
    ChemotaxisPlugin,
    ConnectivityGlobalPlugin,
    ConnectivityPlugin,
    ContactPlugin,
    CurvaturePlugin,
    ExternalPotentialPlugin,
    FocalPointPlasticityPlugin,
    LengthConstraintPlugin,
    MomentOfInertiaPlugin,
    NeighborTrackerPlugin,
    PixelTrackerPlugin,
    SecretionPlugin,
    SurfacePlugin,
    VolumePlugin
]
"""list of plugins that can be registered with cc3d"""

STEPPABLES = [
    BoxWatcherSteppable,
    PIFDumperSteppable
]
"""list of steppables that can be registered with cc3d"""

INITIALIZERS = [
    BlobInitializer,
    PIFInitializer,
    UniformInitializer
]
"""list of initializers that can be registered with cc3d"""

PDESOLVERS = [
    DiffusionSolverFE,
    KernelDiffusionSolver,
    ReactionDiffusionSolverFE,
    SteadyStateDiffusionSolver
]
"""list of PDE solvers that can be registered with cc3d"""


class CoreSpecsValidator:
    """A class containing common validation methods"""

    @classmethod
    def validate_cell_type_names(cls, type_names: Iterable[str], cell_type_spec: CellTypePlugin) -> None:
        """
        Validate a list of cell type names against a CellTypePlugin instance

        :param type_names: cell type names
        :param cell_type_spec: CellTypePlugin instance
        :raises SpecValueError: when a name is not registered with a CellTypePlugin instance
        :return: None
        """
        names_not_found = [f for f in type_names if f not in cell_type_spec.cell_types]
        if names_not_found:
            raise SpecValueError("Could not validate the following names:", names_not_found)

    @classmethod
    def validate_field_names(cls, *specs: _PyCoreSpecsBase, field_names: Iterable[str]) -> None:
        """
        Validate a list of field names against a variable number of specs

        :param specs: spec
        :param field_names: names of fields
        :raises SpecValueError: when a name is not registered with a solver
        :return: None
        """
        solvers_found = {f: False for f in field_names}
        for s in specs:
            if isinstance(s, _PDESolverSpecs):
                for f in field_names:
                    if f in s.field_names:
                        solvers_found[f] = True

        fields_not_found = [k for k, v in solvers_found.items() if not v]
        if fields_not_found:
            raise SpecValueError("Could not validate the following fields:", fields_not_found)

    @classmethod
    def validate_field_name_unique(cls, *specs: _PyCoreSpecsBase, field_name: str) -> None:
        """
        Validate uniqueness of a field name against a variable number of specs

        :param specs: spec
        :param field_name: names of fields
        :raises SpecValueError: when a name is not registered with a solver, or with multiple solvers
        :return: None
        """
        solvers_found = list()
        for s in specs:
            if isinstance(s, _PDESolverSpecs):
                if field_name in s.field_names:
                    solvers_found.append(s.registered_name)

        if len(solvers_found) == 0:
            raise SpecValueError("Could not validate field registration:", field_name)
        elif len(solvers_found) > 1:
            raise SpecValueError("Could not validate uniqueness of field:", field_name, solvers_found)

    @classmethod
    def validate_point(cls, pt: Point3DLike, potts_spec: PottsCore) -> None:
        """
        Validate a point against a PottsCore instance

        :param pt: a point
        :param potts_spec: potts spec
        :raises SpecValueError: when a point is outside of the domain defined in potts_spec
        :return: None
        """
        pt = _as_point3d(pt)
        with _SpecValueErrorContextBlock() as err_ctx:
            for c in ["x", "y", "z"]:
                with err_ctx.ctx:
                    c_val = getattr(pt, c)
                    if c_val < 0:
                        raise SpecValueError(f"Could not validate uniform region {c}-min",
                                             names=[f"{c}-min"])
                    elif c_val > getattr(potts_spec, f"dim_{c}"):
                        raise SpecValueError(f"Could not validate uniform region {c}-max",
                                             names=[f"{c}-max"])

    @classmethod
    def validate_single_instance(cls,
                                 *specs: _PyCoreSpecsBase,
                                 cls_oi: type,
                                 caller_name: str = None) -> None:
        """
        Validates a single instance of a _PyCoreSpecsBase-derived class in a variable number of
        _PyCoreSpecsBase-derived instances

        :param specs: variable number of _PyCoreSpecsBase-derived class instances
        :param cls_oi: class for which to search
        :param caller_name: name of calling instance, for reporting only, Optional
        :raises SpecValueError: when not exactly one instance of cls_oi is found in specs
        :return: None
        """
        inst_found = len([s for s in specs if isinstance(s, cls_oi)])

        if not issubclass(cls_oi, _PyCoreSpecsBase):
            name = cls_oi.__name__
        else:
            name = cls_oi.registered_name

        if inst_found != 1:
            msg = f"Could not validate single instance of {name}"
            if caller_name:
                msg += f" ({caller_name})"
            raise SpecValueError(msg)


def from_xml(_xml: CC3DXMLElement) -> List[_PyCoreXMLInterface]:
    """
    Returns a list of spec instances instantiated from specification in a :class:`CC3DXMLElement` parent instance

    :param _xml: cc3dml specification parent instance
    :return: list of instantiated specs
    """
    o = []

    for s in [Metadata] + PLUGINS + STEPPABLES + INITIALIZERS + PDESOLVERS:
        try:
            s_inst = s.from_xml(_xml)
            o.append(s_inst)
        except SpecImportError:
            pass

    return o


def from_file(_fn: str) -> List[_PyCoreXMLInterface]:
    """
    Returns a list of spec instances instantiated from specification in a .cc3d or .xml file

    :param _fn: absolute path to file
    :raises SpecImportError: when the file does not exist
    :raises SpecValueError: when the file type is not supported
    :return: list of instantiated specs
    """

    if not os.path.isfile(_fn):
        raise SpecImportError("File not found:", _fn)

    filename, filetype = os.path.splitext(_fn)

    if filetype == ".cc3d":
        from cc3d.CompuCellSetup import readCC3DFile
        sdh = readCC3DFile(_fn)
        xml_filename = sdh.cc3dSimulationData.xmlScript

    elif filetype == ".xml":
        xml_filename = _fn
    else:
        raise SpecValueError("Unsupported file type. Supported types are .cc3d and .xml")

    xml2_obj_converter = Xml2Obj()
    xml_el = xml2_obj_converter.Parse(xml_filename)
    return from_xml(xml_el)


def build_xml(*specs: _PyCoreSpecsBase) -> ElementCC3D:
    """
    Returns a complete CC3DML model specification from a variable number of specs

    :param specs: variable number of _PyCoreSpecsBase-derived class instances
    :return: CC3DML model specification
    """
    el = PyCoreSpecsRoot().xml
    [el.add_child(s.xml) for s in specs]
    return el
