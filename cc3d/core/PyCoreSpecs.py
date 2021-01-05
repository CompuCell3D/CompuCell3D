"""
Defines python-based core specification classes
"""

# todo: add validation interface; e.g., chemotaxis checks field specification in pde solver
# todo: add automatic type checks of spec parameter sets
# todo: add to source code doc autogen
# todo: add to python reference manual

from abc import ABC
from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar, Union

import cc3d
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLElement
from cc3d.cpp.CompuCell import Point3D


_Type = TypeVar('_Type')
"""A generic type"""


class SpecValueError(Exception):
    """ Base class for specification value errors """


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
            :raises SpecValueError: when check_dict criterion is True, if any
            :return: None
            :rtype: _Type
            """
            try:
                fcn, msg = _self.check_dict[name]
                if fcn(val):
                    raise SpecValueError(msg)
            except KeyError:
                pass
            _self.spec_dict[name] = val

        def _fset_err(_self, val: _Type) -> None:
            """
            Read-only item setter in :attr:`spec_dict` for :class:`SpecProperty` instances

            :param _self: parent
            :type _self: _PyCoreSpecsBase
            :param val: value
            :raises SpecValueError: when setting
            :return: None
            :rtype: _Type
            """
            raise SpecValueError(f"Setting attribute is illegal.")

        if readonly:
            fset = _fset_err
        else:
            fset = _fset

        super().__init__(fget=_fget, fset=fset)


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
       the first element is a function that takes an input and raise a SpecValueError for an invalid input
       the second element is a message string to accompany the raised SpecValueError
    inputs without a check are not validated, and can be omitted
    
    As a simple example, to enforce positive values for :attr:`steps`, write the following
    
    .. code-block:: python
       
       check_dict = {"steps": (lambda x: x < 1, "Steps must be positive")}

    """

    def __init__(self):
        self.spec_dict: Dict[str, Any] = {}
        """specification dictionary"""

        self._el = None
        """(:class:`ElementCC3D` or None), for keeping a reference alive"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generates base element according to name and type

        :raises SpecValueError: spec type not recognized
        :return: Base element
        :rtype: CC3DXMLElement or None
        """
        if not self.name or not self.type:
            return None
        if self.type.lower() == "plugin":
            return ElementCC3D("Plugin", {"Name": self.registered_name})
        elif self.type.lower() == "steppable":
            return ElementCC3D("Steppable", {"Type": self.registered_name})
        raise SpecValueError(f"Spec type {self.type} not recognized")

    def check_inputs(self, **kwargs) -> None:
        """
        Validates inputs against all registered checks
        Checks are defined in implementation :attr:`check_dict`

        :param kwargs: key-value pair by registered input
        :raises SpecValueError: when check criterion is True
        :return: None
        """
        for k, v in kwargs.items():
            try:
                fcn, msg = self.check_dict[k]
                if fcn(v):
                    raise SpecValueError(msg)
            except KeyError:
                pass

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary. Implementations should override this

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        raise NotImplementedError

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
        raise SpecValueError("Core object cannot be set")


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
    """

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: _PyCoreSpecsBase
        """
        raise NotImplementedError

    @classmethod
    def find_xml_by_attr(cls, _xml: CC3DXMLElement, attr_name: str = None) -> CC3DXMLElement:
        """
        Returns first found xml element in parent by attribute value

        :param CC3DXMLElement _xml: parent element
        :param str attr_name: attribute name, default read from class type
        :raises SpecImportError: When xml element not found
        :return: first found element
        :rtype: CC3DXMLElement
        """
        if attr_name is None:
            if cls.type == "Plugin":
                attr_name = "Name"
            elif cls.type == "Steppable":
                attr_name = "Type"
            else:
                raise SpecImportError("Attribute name not known")

        el = None
        el_list = _xml.getElements(cls.type)
        for els in el_list:
            els: CC3DXMLElement
            if els.findElement(attr_name) and els.getAttribute(attr_name) == cls.registered_name:
                el = els
                break

        if el is None:
            raise SpecImportError(f"{cls.registered_name} not found")

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


class _PyCorePluginSpecs(_PyCoreSpecsBase, ABC):
    """
    Base class for plugins
    """

    type = "Plugin"

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
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
        :rtype: str
        """
        accessor_name = "get" + self.registered_name + self.type
        from cc3d.cpp import CompuCell
        if hasattr(CompuCell, accessor_name):
            return accessor_name
        else:
            return ""


class _PyCoreSteppableSpecs(_PyCoreSpecsBase, ABC):
    """
    Base class for steppables
    """

    type = "Steppable"

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
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
        :rtype: ElementCC3D
        """
        return ElementCC3D("DiffusionData")


class SecretionSpecs(_PyCoreSpecsBase):
    """ Secretion Specs """

    def __init__(self,
                 _cell_type: str,
                 _val: float,
                 constant: bool = False,
                 contact_type: str = None):
        """

        :param str _cell_type: cell type name
        :param float _val: value of parameter
        :param bool constant: flag for constant concentration, optional
        :param str contact_type: name of cell type for on-contact dependence, optional
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
        :rtype: ElementCC3D
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
    def contact_type(self) -> Union[str, None]:
        """

        :return: name of cell type if using secretion on contact with
        :rtype: str or None
        """
        return self.spec_dict["contact_type"]

    @contact_type.setter
    def contact_type(self, _name: Union[str, None]) -> None:
        if _name is not None:
            self.constant = False
        self.spec_dict["contact_type"] = _name

    @property
    def constant(self) -> bool:
        """

        :return: flag whether using constant concentration
        :rtype: bool
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

        :param _param_specs: variable number of SecretionSpecs instances, optional
        """
        super().__init__()

        for _ps in _param_specs:
            if not isinstance(_ps, SecretionSpecs):
                raise SpecValueError("Only SecretionSpecs instances can be passed")

        self.spec_dict = {"param_specs": []}
        [self.param_append(_ps) for _ps in _param_specs]

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("SecretionData")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"]]
        return self._el

    def param_append(self, _ps: SecretionSpecs) -> None:
        """
        Append a secretion spec

        :param SecretionSpecs _ps: secretion spec
        :return: None
        """
        if _ps.contact_type is None:
            for x in self.spec_dict["param_specs"]:
                x: SecretionSpecs
                if x.contact_type is None and _ps.cell_type == x.cell_type:
                    raise SpecValueError(f"Duplicate specification for cell type {_ps.cell_type}")
        self.spec_dict["param_specs"].append(_ps)

    def param_remove(self, _cell_type: str, contact_type: str = None):
        """
        Remove a secretion spec

        :param str _cell_type: name of cell type
        :param contact_type: name of on-contact dependent cell type, optional
        :return: None
        """
        for _ps in self.spec_dict["param_specs"]:
            _ps: SecretionSpecs
            if _ps.cell_type == _cell_type and contact_type == _ps.contact_type:
                self.spec_dict["param_specs"].remove(_ps)
                return
        raise SpecValueError("SecretionSpecs not specified")

    def param_new(self,
                  _cell_type: str,
                  _val: float,
                  constant: bool = False,
                  contact_type: str = None) -> SecretionSpecs:
        """
        Append and return a new secretion spec

        :param str _cell_type: cell type name
        :param float _val: value of parameter
        :param bool constant: flag for constant concentration, optional
        :param str contact_type: name of cell type for on-contact dependence, optional
        :return: new secretion spec
        :rtype: SecretionSpecs
        """
        ps = SecretionSpecs(_cell_type, _val, constant=constant, contact_type=contact_type)
        self.param_append(ps)
        return ps


BOUNDARYTYPESPDE = ["Value", "Flux", "Periodic"]


class PDEBoundaryConditionsSpec(_PyCoreSpecsBase):
    """ PDE Solver Boundary Conditions Spec """

    def __init__(self,
                 x_min_type: str = BOUNDARYTYPESPDE[0], x_min_val: float = 0.0,
                 x_max_type: str = BOUNDARYTYPESPDE[0], x_max_val: float = 0.0,
                 y_min_type: str = BOUNDARYTYPESPDE[0], y_min_val: float = 0.0,
                 y_max_type: str = BOUNDARYTYPESPDE[0], y_max_val: float = 0.0,
                 z_min_type: str = BOUNDARYTYPESPDE[0], z_min_val: float = 0.0,
                 z_max_type: str = BOUNDARYTYPESPDE[0], z_max_val: float = 0.0):
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

        self._xml_type_labels = {BOUNDARYTYPESPDE[0]: "ConstantValue",
                                 BOUNDARYTYPESPDE[1]: "ConstantDerivative",
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
        :rtype: str
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
        :rtype: str
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
        :rtype: str
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
        :rtype: str
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
        :rtype: str
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
        :rtype: str
        """
        return self.spec_dict["z_max_type"]

    @z_max_type.setter
    def z_max_type(self, _val: str) -> None:
        if _val not in BOUNDARYTYPESPDE:
            raise SpecValueError(f"Valid boundary types are {BOUNDARYTYPESPDE}")
        if BOUNDARYTYPESPDE[2] in [_val, self.spec_dict["z_min_type"]]:
            self.spec_dict["z_min_type"] = _val
        self.spec_dict["z_max_type"] = _val

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(self._xml_plane(x)) for x in ["X", "Y", "Z"]]
        return self._el


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
                          "bc_specs": PDEBoundaryConditionsSpec()}

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    diff_data: _DD = SpecProperty(name="diff_data")
    """diffusion data"""

    bcs: PDEBoundaryConditionsSpec = SpecProperty(name="bc_specs")
    """boundary conditions"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("DiffusionField", {"Name": self.field_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        self._el.add_child(self.spec_dict["diff_data"].xml)
        self._el.add_child(self.spec_dict["secr_data"].xml)
        self._el.add_child(self.spec_dict["bc_specs"].xml)
        return self._el

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> None:
        """
        Specify a new secretion data spec

        :param str _cell_type: name of cell type
        :param float _val: value
        :param kwargs:
        :return: None
        """
        self.spec_dict["secr_data"].param_new(_cell_type, _val, **kwargs)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param str _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        self.spec_dict["secr_data"].param_remove(_cell_type, **kwargs)


class _PDESolverSpecs(_PyCoreSteppableSpecs, _PyCoreSteerableInterface, _PyCoreXMLInterface, ABC, Generic[_DD, _SD]):

    _field_spec = _PDESolverFieldSpecs
    _diff_data: _DD = _PDEDiffusionDataSpecs
    _secr_data: _SD = _PDESecretionDataSpecs

    def __init__(self):
        super().__init__()

        self.spec_dict = {"fluc_comp": False,
                          "fields": dict()}

    @property
    def fields(self) -> _PyCoreParamAccessor[_PDESolverFieldSpecs[_DD, _SD]]:
        """

        :return: accessor to field parameters with field names as keys
        :rtype: dict of [str: _field_spec]
        """
        return _PyCoreParamAccessor(self, "fields")

    @property
    def field_names(self) -> List[str]:
        """

        :return: list of registered field names
        :rtype: list of str
        """
        return [x for x in self.spec_dict["fields"].keys()]

    def field_new(self, _field_name: str):
        """
        Append and return a new field spec

        :param str _field_name: name of field
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

        :param str _field_name: name of field
        :return: None
        """
        if _field_name not in self.field_names:
            raise SpecValueError(f"Field with name {_field_name} not specified")
        self.spec_dict["fields"].pop(_field_name)


class PyCoreSpecsMaster:
    """
    Master simulation specs
    """
    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        return ElementCC3D("CompuCell3D", {"Version": cc3d.__version__,
                                           "Revision": cc3d.__revision__})

    @staticmethod
    def get_simulator() -> cc3d.cpp.CompuCell.Simulator:
        """

        :return: simulator
        :rtype: cc3d.cpp.CompuCell.Simulator
        """
        from cc3d.CompuCellSetup import persistent_globals
        return persistent_globals.simulator


#: Supported lattice types
LATTICETYPES = ["Cartesian", "Hexagonal"]
#: Supported boundary types
BOUNDARYTYPESPOTTS = ["NoFlux", "Periodic"]
#: Supported fluctuation amplitude function names
FLUCAMPFCNS = ["Min", "Max", "ArithmeticAverage"]


class PottsCoreSpecs(_PyCoreSteerableInterface, _PyCoreXMLInterface):
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
        "neighbor_order": (lambda x: not 0 < x < 4, "Invalid neighbor order. Must be in [1, 3]"),
        "debug_output_frequency": (lambda x: x < 0,
                                   "Invalid debug output frequency. Must be non-negative"),
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
                 fluctuation_amplitude_function: str = "Min",
                 boundary_x: str = "NoFlux",
                 boundary_y: str = "NoFlux",
                 boundary_z: str = "NoFlux",
                 neighbor_order: int = 1,
                 debug_output_frequency: int = 10,
                 random_seed: int = None,
                 lattice_type: str = "Cartesian",
                 offset: float = 0):
        """

        :param int dim_x: x-dimension of simulation domain, defaults to 1
        :param int dim_y: y-dimension of simulation domain, defaults to 1
        :param int dim_z: z-dimension of simulation domain, defaults to 1
        :param int steps: number of simulation steps, defaults to 0
        :param int anneal: number of annealing steps, defaults to 0
        :param float fluctuation_amplitude: constant fluctuation amplitude, defaults to 10
        :param str fluctuation_amplitude_function:
            fluctuation amplitude function for heterotypic fluctuation amplitudes, defaults to "Min"
        :param str boundary_x: boundary conditions orthogonal to x-direction, defaults to "NoFlux"
        :param str boundary_y: boundary conditions orthogonal to y-direction, defaults to "NoFlux"
        :param str boundary_z: boundary conditions orthogonal to z-direction, defaults to "NoFlux"
        :param int neighbor_order: neighbor order of flip attempts, defaults to 1
        :param int debug_output_frequency: debug output frequency, defaults to 10
        :param int random_seed: random seed, optional
        :param str lattice_type: type of lattice, defaults to "Cartesian"
        :param float offset: offset in Boltzmann acceptance function
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
                          debug_output_frequency=debug_output_frequency,
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
                          "debug_output_frequency": debug_output_frequency,
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

    debug_output_frequency: int = SpecProperty(name="debug_output_frequency")
    """debug output frequency"""

    random_seed: Union[int, None] = SpecProperty(name="random_seed")
    """random seed"""

    lattice_type: str = SpecProperty(name="lattice_type")
    """type of lattice"""

    offset: float = SpecProperty(name="offset")
    """offset in Boltzmann acceptance function"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D(self.registered_name)

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("Dimensions", {"x": str(self.dim_x), "y": str(self.dim_y), "z": str(self.dim_z)})

        self._el.ElementCC3D("Steps", {}, str(self.steps))

        if self.anneal > 0:
            self._el.ElementCC3D("Anneal", {}, str(self.anneal))

        if isinstance(self.spec_dict["fluctuation_amplitude"], float):
            self._el.ElementCC3D("FluctuationAmplitude", {}, str(self.spec_dict["fluctuation_amplitude"]))
        else:
            fa_el = self._el.ElementCC3D("FluctuationAmplitude")
            for _type, _fa in self.spec_dict["fluctuation_amplitude"].items():
                fa_el.ElementCC3D("FluctuationAmplitudeParameters", {"CellType": _type,
                                                                     "FluctuationAmplitude": str(_fa)})

        if self.fluctuation_amplitude_function != FLUCAMPFCNS[0]:
            self._el.ElementCC3D("FluctuationAmplitudeFunctionName",
                                 {},
                                 self.fluctuation_amplitude_function)

        for c in ["x", "y", "z"]:
            if getattr(self, f"boundary_{c}") != BOUNDARYTYPESPOTTS[0]:
                self._el.ElementCC3D(f"Boundary_{c}", {}, getattr(self, f"boundary_{c}"))

        if self.neighbor_order > 1:
            self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))

        self._el.ElementCC3D("DebugOutputFrequency", {}, str(self.debug_output_frequency))

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

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: PottsCoreSpecs
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
                amp_el_list = amp_el.getElements("FluctuationAmplitudeParameters")
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
            o.debug_output_frequency = el.getFirstElement("DebugOutputFrequency").getUInt()

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

        :param float _fa: fluctuation amplitude
        :return: None
        """
        self.spec_dict["fluctuation_amplitude"] = float(_fa)

    def fluctuation_amplitude_type(self, _type: str, _fa: float) -> None:
        """
        Set fluctuation amplitude by cell type

        :param str _type: cell type
        :param float _fa: fluctuation amplitude
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

        :param str output_filename:
            name of the file to which CC3D will write average changes in energies returned by each plugin and
            corresponding standard deviations for those steps whose values are divisible by :attr:`frequency`.
        :param int frequency: frequency at which to output calculations
        :param int flip_frequency: frequency at which to output flip attempt data, optional
        :param bool gather_results: ensures one file written, defaults to False
        :param bool output_accepted: write data on accepted flip attempts, defaults to False
        :param bool output_rejected: write data on rejected flip attempts, defaults to False
        :param bool output_total: write data on all flip attempts, defaults to False
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
        :rtype: cc3d.cpp.CompuCell.Potts3D
        """
        from cc3d.CompuCellSetup import persistent_globals
        try:
            return persistent_globals.simulator.getPotts()
        except AttributeError:
            raise SpecValueError("Potts unavailable")


class CellTypePluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
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

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: CellTypePluginSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("CellType")

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
        :rtype: list of str
        """
        return [x[0] for x in self.spec_dict["cell_types"]]

    @property
    def cell_type_ids(self) -> List[int]:
        """

        :return: list of cell type ids
        :rtype: list of int
        """
        return [x[1] for x in self.spec_dict["cell_types"]]

    def is_frozen(self, _name: str) -> Union[bool, None]:
        """

        :param _name: name of cell type
        :type _name: str
        :return: True if frozen, False if not frozen, None if cell type not found
        :rtype: bool or None
        """
        for x in self.spec_dict["cell_types"]:
            if x[0] == _name:
                return x[2]
        return None

    def cell_type_append(self, _name: str, type_id: int = None, frozen: bool = False) -> None:
        """
        Add a cell type

        :param str _name: name of cell type
        :param int type_id: id of cell type, optional
        :param bool frozen: freeze cell type if True, defaults to False
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

        :param str _name: name of cell type
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

        :param str old_name: old cell type name
        :param str new_name: new cell type name
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

        :param int old_id: old id
        :param int new_id: new id
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

        :param str _type_name: cell type name
        :param bool freeze: frozen state
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

        :param str _cell_type: cell type name
        :param float target_volume: target volume
        :param float lambda_volume: lambda value
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
        :rtype: ElementCC3D
        """
        self._el = ElementCC3D("VolumeEnergyParameters", {"CellType": self.cell_type,
                                                          "TargetVolume": str(self.target_volume),
                                                          "LambdaVolume": str(self.lambda_volume)})
        return self._el


class VolumePluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """
    Volume Plugin

    VolumeEnergyParameter instances can be accessed as follows,

    .. code-block:: python

       spec: VolumePluginSpecs
       cell_type_name: str
       params: VolumeEnergyParameter = spec[cell_type_name]
       target_volume: float = params.target_volume
       lambda_volume: float = params.lambda_volume

    """

    name = "volume"
    registered_name = "Volume"

    def __init__(self, *_params):
        super().__init__()

        for _p in _params:
            if not isinstance(_p, VolumeEnergyParameter):
                raise SpecValueError("Only VolumeEnergyParameter instances can be passed")

        self.spec_dict = {"params": {_p.cell_type: _p for _p in _params}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        for _p in self.spec_dict["params"].values():
            self._el.add_child(_p.xml)
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: VolumePluginSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("VolumeEnergyParameters")

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
        :rtype: list of str
        """
        return list(self.spec_dict["params"].keys())

    def param_append(self, _p: VolumeEnergyParameter) -> None:
        """
        Appends a volume energy parameter

        :param VolumeEnergyParameter _p: volume energy parameter
        :return: None
        """
        if _p.cell_type in self.cell_types:
            raise SpecValueError(f"Cell type {_p.cell_type} already specified")
        self.spec_dict["params"][_p.cell_type] = _p

    def param_remove(self, _cell_type: str) -> None:
        """
        Remove a parameter by cell type

        :param str _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Cell type {_cell_type} not specified")
        self.spec_dict["params"].pop(_cell_type)

    def param_new(self, _cell_type: str, target_volume: float, lambda_volume: float) -> VolumeEnergyParameter:
        """
        Appends and returns a new volume energy parameter

        :param str _cell_type: cell type name
        :param float target_volume: target volume
        :param float lambda_volume: lambda value
        :return: new volume energy parameter
        :rtype: VolumeEnergyParameter
        """
        p = VolumeEnergyParameter(_cell_type, target_volume=target_volume, lambda_volume=lambda_volume)
        self.param_append(p)
        return p


# Surface Plugin


class SurfaceEnergyParameterSpecs(_PyCoreSpecsBase):
    """ Surface Energy Parameter """

    check_dict = {"target_surface": (lambda x: x < 0, "Target surface must be non-negative")}

    def __init__(self, _cell_type: str, target_surface: float, lambda_surface: float):
        """

        :param str _cell_type: cell type name
        :param float target_surface: target surface
        :param float lambda_surface: lambda value
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
        :rtype: ElementCC3D
        """
        self._el = ElementCC3D("SurfaceEnergyParameters", {"CellType": self.cell_type,
                                                           "TargetSurface": str(self.target_surface),
                                                           "LambdaSurface": str(self.lambda_surface)})
        return self._el


class SurfacePluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    Surface Plugin

    SurfaceEnergyParameterSpecs instances can be accessed as follows,

    .. code-block:: python

       spec: SurfacePluginSpecs
       cell_type_name: str
       params: SurfaceEnergyParameterSpecs = spec[cell_type_name]
       target_surface: float = params.target_surface
       lambda_surface: float = params.lambda_surface

    """

    name = "surface"
    registered_name = "Surface"

    def __init__(self, *_params):
        super().__init__()

        for _p in _params:
            if not isinstance(_p, SurfaceEnergyParameterSpecs):
                raise SpecValueError("Only SurfaceEnergyParameter instances can be passed")

        self.spec_dict = {"params": {_p.cell_type: _p for _p in _params}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        for _p in self.spec_dict["params"].values():
            self._el.add_child(_p.xml)
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: SurfacePluginSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("SurfaceEnergyParameters")

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
        :rtype: list of str
        """
        return list(self.spec_dict["params"].keys())

    def param_append(self, _p: SurfaceEnergyParameterSpecs):
        """
        Appends a surface energy parameter
        
        :param SurfaceEnergyParameterSpecs _p: surface energy parameter 
        :return: None
        """
        if _p.cell_type in self.cell_types:
            raise SpecValueError(f"Cell type {_p.cell_type} already specified")
        self.spec_dict["params"][_p.cell_type] = _p

    def param_remove(self, _cell_type: str):
        """
        Remove a parameter by cell type

        :param str _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Cell type {_cell_type} not specified")
        self.spec_dict["params"].pop(_cell_type)

    def param_new(self, cell_type: str, target_surface: float, lambda_surface: float) -> SurfaceEnergyParameterSpecs:
        """
        Appends and returns a new surface energy parameter

        :param str cell_type: cell type name
        :param float target_surface: target surface
        :param float lambda_surface: lambda value
        """
        p = SurfaceEnergyParameterSpecs(cell_type,
                                        target_surface=target_surface,
                                        lambda_surface=lambda_surface)
        self.param_append(p)
        return p


class NeighborTrackerPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: NeighborTrackerPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Chemotaxis Plugin


class ChemotaxisTypeSpecs(_PyCoreSpecsBase):
    """ Chemotaxis cell type specs """

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

        :param str _cell_type: name of cell type
        :param float lambda_chemo: lambda value
        :param float sat_cf: saturation coefficient
        :param float linear_sat_cf: linear saturation coefficient
        :param str towards: cell type name to chemotax towards, optional
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

    sat_cf: Union[float, None] = SpecProperty(name="sat_cf")
    """saturation coefficient, Optional, None if not set"""

    linear_sat_cf: Union[float, None] = SpecProperty(name="linear_sat_cf")
    """linear saturation coefficient, Optional, None if not set"""

    towards: Union[str, None] = SpecProperty(name="towards")
    """name of cell type if chemotaxing towards, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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


class ChemotaxisParams(_PyCoreSpecsBase):
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

        :param str field_name: name of field
        :param str solver_name: name of solver
        :param _type_specs: variable number of ChemotaxisTypeSpecs instances
        """
        super().__init__()

        for _ts in _type_specs:
            if not isinstance(_ts, ChemotaxisTypeSpecs):
                raise SpecValueError("Only ChemotaxisTypeSpecs instances can specify chemotaxis type data")

        self.spec_dict = {"field_name": field_name,
                          "solver_name": solver_name,
                          "type_specs": {_ts.cell_type: _ts for _ts in _type_specs}}

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(_ts.xml) for _ts in self.spec_dict["type_specs"].values()]
        return self._el

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    solver_name: str = SpecProperty(name="solver_name")
    """name of PDE solver of field"""

    def __getitem__(self, item):
        if item not in self.spec_dict["type_specs"].keys():
            raise SpecValueError(f"Type name {item} not specified")
        return self.spec_dict["type_specs"][item]

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        :rtype: list of str
        """
        return list(self.spec_dict["type_specs"].keys())

    def param_append(self, type_spec: ChemotaxisTypeSpecs) -> None:
        """
        Appends a chemotaxis type spec

        :param ChemotaxisTypeSpecs type_spec: chemotaxis type spec
        :return: None
        """
        if type_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {type_spec.cell_type} already specified")
        self.spec_dict["type_specs"][type_spec.cell_type] = type_spec

    def param_remove(self, _cell_type: str) -> None:
        """
        Removes a chemotaxis type spec

        :param str _cell_type: cell type name
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Type name {_cell_type} not specified")
        self.spec_dict["type_specs"].pop(_cell_type)

    def param_new(self,
                  _cell_type: str,
                  lambda_chemo: float = 0.0,
                  sat_cf: float = None,
                  linear_sat_cf: float = None,
                  towards: str = None) -> ChemotaxisTypeSpecs:
        """
        Appends and returns a new chemotaxis type spec

        :param str _cell_type: name of cell type
        :param float lambda_chemo: lambda value
        :param float sat_cf: saturation coefficient
        :param float linear_sat_cf: linear saturation coefficient
        :param str towards: cell type name to chemotax towards, optional
        :return: new chemotaxis type spec
        :rtype: ChemotaxisTypeSpecs
        """
        p = ChemotaxisTypeSpecs(_cell_type,
                                lambda_chemo=lambda_chemo,
                                sat_cf=sat_cf,
                                linear_sat_cf=linear_sat_cf,
                                towards=towards)
        self.param_append(p)
        return p


class ChemotaxisPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    Chemotaxis Plugin

    ChemotaxisTypeSpecs instances can be accessed by field name as follows,

    .. code-block:: python

       spec: ChemotaxisPluginSpecs
       field_name: str
       cell_type_name: str
       field_params: ChemotaxisParams = spec[field_name]
       type_params: ChemotaxisTypeSpecs = field_params[cell_type_name]
       lambda_chemo: float = type_params.lambda_chemo
       sat_cf: float = type_params.sat_cf
       linear_sat_cf: float = type_params.linear_sat_cf
       towards: str = type_params.towards

    """

    name = "chemotaxis"
    registered_name = "Chemotaxis"

    def __init__(self, *_field_specs):
        super().__init__()

        for _fs in _field_specs:
            if not isinstance(_fs, ChemotaxisParams):
                raise SpecValueError("Can only pass ChemotaxisParams instances")

        self.spec_dict = {"field_specs": {_fs.field_name: _fs for _fs in _field_specs}}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(_fs.xml) for _fs in self.spec_dict["field_specs"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ChemotaxisPluginSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("ChemicalField")

        for f_el in el_list:
            field_name = f_el.getAttribute("Name")
            # Allowing unspecified Source
            solver_name = ""
            if f_el.findAttribute("Source"):
                solver_name=f_el.getAttribute("Source")
            f = ChemotaxisParams(field_name=field_name, solver_name=solver_name)
            f_el_list = f_el.getElements("ChemotaxisByType")

            for p_el in f_el_list:
                cell_type = p_el.getAttribute("Type")
                kwargs = {"lambda_chemo": p_el.getAttributeAsDouble("Lambda")}
                if p_el.findAttribute("SaturationCoef"):
                    kwargs["sat_cf"] = p_el.getAttributeAsDouble("SaturationCoef")
                elif p_el.findAttribute("SaturationLinearCoef"):
                    kwargs["linear_sat_cf"] = p_el.getAttributeAsDouble("SaturationLinearCoef")
                if p_el.findAttribute("ChemotactTowards"):
                    kwargs["towards"] = p_el.getAttribute("ChemotactTowards")
                p = ChemotaxisTypeSpecs(cell_type, **kwargs)
                f.param_append(p)

            o.param_append(f)

        return o

    @property
    def fields(self) -> List[str]:
        """

        :return: list of registered field names
        :rtype: list of str
        """
        return [_fs.field_name for _fs in self.spec_dict["field_specs"].values()]

    def __getitem__(self, item) -> ChemotaxisParams:
        if item not in self.fields:
            raise SpecValueError(f"Field name {item} not specified")
        return self.spec_dict["field_specs"][item]

    def param_append(self, _field_specs: ChemotaxisParams) -> None:
        """
        Appends a chemotaxis parameter

        :param ChemotaxisParams _field_specs: chemotaxis parameter
        :return: None
        """
        if _field_specs.field_name in self.fields:
            raise SpecValueError(f"Field name {_field_specs.field_name} already specified")
        self.spec_dict["field_specs"][_field_specs.field_name] = _field_specs

    def param_remove(self, _field_name: str) -> None:
        """
        Removes a new chemotaxis parameter

        :param str _field_name: name of field
        :return: None
        """
        if _field_name not in self.fields:
            raise SpecValueError(f"Field name {_field_name} not specified")
        self.spec_dict["field_specs"].pop(_field_name)

    def param_new(self,
                  field_name: str,
                  solver_name: str,
                  *_type_specs) -> ChemotaxisParams:
        """
        Appends and returns a new chemotaxis parameter

        :param str field_name: name of field
        :param str solver_name: name of solver
        :param _type_specs: variable number of ChemotaxisTypeSpecs instances
        :return: new chemotaxis parameter
        :rtype: ChemotaxisParams
        """
        p = ChemotaxisParams(field_name=field_name, solver_name=solver_name, *_type_specs)
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

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("ExternalPotentialParameters", {"CellType": self.cell_type,
                                                           "x": self.x, "y": self.y, "z": self.z})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    cell_type: str = SpecProperty(name="cell_type")
    """cell type name"""

    x: float = SpecProperty(name="x")
    """x-component of external potential lambda vector"""

    y: float = SpecProperty(name="y")
    """y-component of external potential lambda vector"""

    z: float = SpecProperty(name="z")
    """z-component of external potential lambda vector"""


class ExternalPotentialPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    ExternalPotential Plugin

    ExternalPotentialParameter instances can be accessed by cell type name as follows,

    .. code-block:: python

       spec: ExternalPotentialPluginSpecs
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

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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

    com_based: bool = SpecProperty(name="com_based")
    """center-of-mass-based flag"""

    lambda_x: float = SpecProperty(name="lambda_x")
    """global x-component of external potential lambda vector"""

    lambda_y: float = SpecProperty(name="lambda_y")
    """global y-component of external potential lambda vector"""

    lambda_z: float = SpecProperty(name="lambda_z")
    """global z-component of external potential lambda vector"""

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ExternalPotentialPluginSpecs
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

        el_list = el.getElements("ExternalPotentialParameters")

        for p_el in el_list:
            p_el: CC3DXMLElement
            kwargs = {comp: p_el.getAttributeAsDouble(comp) for comp in ["x", "y", "z"] if p_el.findAttribute(comp)}
            o.param_append(ExternalPotentialParameter(p_el.getAttribute("CellType"), **kwargs))

        return o

    @property
    def cell_type(self) -> _PyCoreParamAccessor[ExternalPotentialParameter]:
        """

        :return: accessor to external potential parameters with cell types as keys
        :rtype: dict of [str: ExternalPotentialParameter]
        """
        return _PyCoreParamAccessor(self, "param_specs")

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        :rtype: list of str
        """
        return list(self.spec_dict["param_specs"].keys())

    def param_append(self, param_spec: ExternalPotentialParameter = None) -> None:
        """
        Append external potential parameter

        :param ExternalPotentialParameter param_spec: external potential parameters
        :return: None
        """
        if param_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {param_spec.cell_type} already specified")
        self.spec_dict["param_specs"][param_spec.cell_type] = param_spec

    def param_remove(self, _cell_type: str) -> None:
        """
        Remove a :class:`ExternalPotentialParameter` instance

        :param str _cell_type: name of cell type
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

        :param str _cell_type: name of cell type
        :param float x: x-component value
        :param float y: y-component value
        :param float z: z-component value
        :return: new parameter
        :rtype: ExternalPotentialParameter
        """
        p = ExternalPotentialParameter(_cell_type, x=x, y=y, z=z)
        self.param_append(p)
        return p


class CenterOfMassPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """ CenterOfMass Plugin """

    name = "center_of_mass"
    registered_name = "CenterOfMass"

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: CenterOfMassPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Contact Plugin


class ContactEnergyParam(_PyCoreSpecsBase):
    """ Contact Energy Parameter """

    def __init__(self, type_1: str, type_2: str, energy: float):
        """

        :param str type_1: first cell type name
        :param str type_2: second cell type name
        :param float energy: parameter value
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
        :rtype: ElementCC3D
        """
        self._el = ElementCC3D("Energy", {"Type1": self.type_1, "Type2": self.type_2}, str(self.energy))
        return self._el


class ContactPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """
    Contact Plugin

    ContactEnergyParam instances can be accessed as follows,

    .. code-block:: python

       spec: ContactPluginSpecs
       cell_type_1: str
       cell_type_2: str
       params: ContactEnergyParam = spec[cell_type_1][cell_type_2]
       energy: float = params.energy

    """

    name = "contact"
    registered_name = "Contact"

    check_dict = {"neighbor_order": (lambda x: not (0 < x < 5), "Valid neighbor orders are in [1, 4]")}

    def __init__(self, neighbor_order: int = 1, *_params):
        """

        :param int neighbor_order: neighbor order
        :param _params: variable number of ContactEnergyParam instances
        """
        super().__init__()

        for _p in _params:
            if not isinstance(_p, ContactEnergyParam):
                raise SpecValueError("Only ContactEnergyParam instances can be passed")

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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        for _pdict in self.spec_dict["energies"].values():
            for _p in _pdict.values():
                self._el.add_child(_p.xml)
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ContactPluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = el.getElements("Energy")

        for p_el in el_list:
            p_el: CC3DXMLElement
            o.param_append(ContactEnergyParam(type_1=p_el.getAttribute("Type1"),
                                              type_2=p_el.getAttribute("Type2"),
                                              energy=p_el.getDouble()))

        if el.findElement("NeighborOrder"):
            o.neighbor_order = el.getFirstElement("NeighborOrder").getInt()

        return o

    def __getitem__(self, item: str) -> Dict[str, ContactEnergyParam]:
        return self.spec_dict["energies"][item]

    def types_specified(self, _type_name: str) -> List[str]:
        """

        :return: list of cell type names
        :rtype: list of str
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

    def param_append(self, _p: ContactEnergyParam) -> None:
        """
        Add a contact energy parameter

        :param ContactEnergyParam _p: the parameter
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
        :type type_1: str
        :param type_2: name of second cell type
        :type type_2: str
        :return: None
        """
        if type_2 not in self.types_specified(type_1):
            raise SpecValueError(f"Contact parameter not specified for ({type_1}, {type_2})")
        if type_1 in self.spec_dict["energies"].keys():
            key, val = type_1, type_2
        else:
            key, val = type_2, type_1
        self.spec_dict["energies"][key].pop(val)

    def param_new(self, type_1: str, type_2: str, energy: float) -> ContactEnergyParam:
        """
        Appends and returns a new contact energy parameter

        :param str type_1: first cell type name
        :param str type_2: second cell type name
        :param float energy: parameter value
        :return: new parameter
        :rtype: ContactEnergyParam
        """
        p = ContactEnergyParam(type_1=type_1, type_2=type_2, energy=energy)
        self.param_append(p)
        return p


class ContactLocalFlexPluginSpecs(ContactPluginSpecs, _PyCoreSteerableInterface):
    """
    ContactLocalFlex Plugin

    A steerable version of :class:`ContactPluginSpecs`
    """

    name = "contact_local_flex"
    registered_name = "ContactLocalFlex"


class ContactInternalPluginSpecs(ContactLocalFlexPluginSpecs):
    """
    ContactInternal Plugin

    Like :class:`ContactLocalFlexPluginSpecs`, but for contact between compartments
    """

    name = "contact_internal"
    registered_name = "ContactInternal"


# AdhesionFlex Plugin


class AdhesionFlexBindingFormulaSpecs(_PyCoreSpecsBase):
    """
    AdhesionFlex Binding Formula

    Binding parameters can be set like a dictionary as follows

    .. code-block:: python

       spec: AdhesionFlexBindingFormulaSpecs
       molecule_1_name: str
       molecule_2_name: str
       density: float
       spec[molecule_1_name][molecule_2_name] = density

    """

    def __init__(self, formula_name: str, formula: str):
        """

        :param str formula_name: name of forumla
        :param str formula: formula
        """
        super().__init__()

        self.spec_dict = {"formula_name": formula_name,
                          "formula": formula,
                          "interactions": dict()}

    formula_name: str = SpecProperty(name="formula_name")
    """name of formula"""

    formula: str = SpecProperty(name="formula")
    """formula"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("BindingFormula", {"Name": self.formula_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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

        :param str item: molecule 1 name
        :return: dictionary of densities
        :rtype: dict
        """
        if item not in self.spec_dict["interactions"].keys():
            self.spec_dict["interactions"][item] = dict()
        return self.spec_dict["interactions"][item]

    def param_set(self, _mol1: str, _mol2: str, _val: float) -> None:
        """
        Sets an adhesion binding parameter

        :param _mol1: name of first molecule
        :type _mol1: str
        :param _mol2: name of second molecule
        :type _mol2: str
        :param _val: binding parameter value
        :type _val: str
        :return: None
        """
        if _mol1 not in self.spec_dict["interactions"].keys():
            self.spec_dict["interactions"][_mol1] = dict()
        self.spec_dict["interactions"][_mol1][_mol2] = _val

    def param_remove(self, _mol1: str, _mol2: str) -> None:
        """
        Removes an adhesion binding parameter

        :param _mol1: name of first molecule
        :type _mol1: str
        :param _mol2: name of second molecule
        :type _mol2: str
        :return: None
        """
        if _mol1 not in self.spec_dict["interactions"].keys() \
                or _mol2 not in self.spec_dict["interactions"][_mol1].keys():
            raise SpecValueError(f"Parameter not defined for types ({_mol1}, {_mol2})")
        self.spec_dict["interactions"].pop(_mol2)


class AdhesionMoleculeDensitySpecs(_PyCoreSpecsBase):
    """ AdhesionMoleculeDensitySpecs """

    check_dict = {"density": (lambda x: x < 0.0, "Molecule density must be non-negative")}

    def __init__(self, molecule: str, cell_type: str, density: float):
        """

        :param str molecule: name of molecule
        :param str cell_type: name of cell type
        :param float density: molecule density
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

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("AdhesionMoleculeDensity", {"Molecule": self.molecule,
                                                       "CellType": self.cell_type,
                                                       "Density": str(self.density)})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el


class AdhesionFlexPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    AdhesionFlex Plugin

    AdhesionMoleculeDensitySpecs instances can be accessed by molecule and cell type as follows,

    .. code-block:: python

       specs_adhesion_flex: AdhesionFlexPluginSpecs
       molecule_name: str
       cell_type_name: str
       x: dict = specs_adhesion_flex.density.cell_type[cell_type_name]  # keys are molecule names
       y: dict = specs_adhesion_flex.density.molecule[molecule_name]  # keys are cell type names
       a: AdhesionMoleculeDensitySpecs = x[molecule_name]
       b: AdhesionMoleculeDensitySpecs = y[cell_type_name]
       a is b  # Evaluates to True

    AdhesionFlexBindingFormulaSpecs instances can be accessed as follows,

    .. code-block:: python

       specs_adhesion_flex: AdhesionFlexPluginSpecs
       formula_name: str
       molecule1_name: str
       molecule2_name: str
       x: AdhesionFlexBindingFormulaSpecs = specs_adhesion_flex.formula[formula_name]
       y: dict = x[molecule1_name]  # Keys are molecule names
       binding_param: float = y[molecule2_name]

    """

    name = "adhesion_flex"
    registered_name = "AdhesionFlex"

    check_dict = {"neighbor_order": (lambda x: not (0 < x < 5), "Valid neighbor orders are in [1, 4]")}

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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.ElementCC3D("AdhesionMolecule", {"Molecule": m}) for m in self.molecules]
        [[self._el.add_child(e.xml) for e in e_dict.values()] for e_dict in self.spec_dict["densities"].values()]
        [self._el.add_child(e.xml) for e in self.spec_dict["binding_formulas"].values()]
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: AdhesionFlexPluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = el.getElements("AdhesionMolecule")

        for m_el in el_list:
            o.molecule_append(m_el.getAttribute("Molecule"))

        el_list = el.getElements("AdhesionMoleculeDensity")

        for d_el in el_list:
            d_el: CC3DXMLElement
            o.density_append(AdhesionMoleculeDensitySpecs(molecule=d_el.getAttribute("Molecule"),
                                                          cell_type=d_el.getAttribute("CellType"),
                                                          density=d_el.getAttributeAsDouble("Density")))

        el_list = el.getElements("BindingFormula")

        for f_el in el_list:
            f_el: CC3DXMLElement
            f = AdhesionFlexBindingFormulaSpecs(formula_name=f_el.getAttribute("Name"),
                                                formula=f_el.getFirstElement("Formula").getText())
            for p_mat_el in f_el.getFirstElement("Variables").getFirstElement("AdhesionInteractionMatrix"):

                p_el_list = p_mat_el.getElements("BindingParameter")

                for p_el in p_el_list:
                    p_el: CC3DXMLElement
                    f.param_set(p_el.getAttribute("Molecule1"), p_el.getAttribute("Molecule2"), p_el.getDouble())

            o.formula_append(f)

        return o

    @property
    def molecules(self) -> List[str]:
        """

        :return: list of registered molecule names
        :rtype: list of str
        """
        return [x for x in self.spec_dict["molecules"]]

    @property
    def formulas(self) -> List[str]:
        """

        :return: list of register formula names
        :rtype: list of str
        """
        return [x for x in self.spec_dict["binding_formulas"].keys()]

    @property
    def formula(self) -> _PyCoreParamAccessor[AdhesionFlexBindingFormulaSpecs]:
        """

        :return: accessor to binding formulas with formula names as keys
        :rtype: dict of [str: AdhesionFlexBindingFormulaSpecs]
        """
        return _PyCoreParamAccessor(self, "binding_formulas")

    @property
    def density(self):
        """

        :return: accessor to adhesion molecule density accessor
        :rtype: _AdhesionFlexDensityAccessor
        """
        return _AdhesionFlexDensityAccessor(self)

    def molecule_append(self, _name: str) -> None:
        """
        Append a molecule name

        :param str _name: name of molecule
        :return: None
        """
        if _name in self.molecules:
            raise SpecValueError(f"Molecule with name {_name} already specified")
        self.spec_dict["molecules"].append(_name)

    def molecule_remove(self, _name: str) -> None:
        """
        Remove molecule and all associated binding parameters and densities

        :param str _name: name of molecule
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

    def density_append(self, _dens: AdhesionMoleculeDensitySpecs) -> None:
        """
        Appends a molecule density; molecules are automatically appended if necessary

        :param AdhesionMoleculeDensitySpecs _dens: adhesion molecule density spec
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

        :param str molecule: name of molecule
        :param str cell_type: name of cell type
        :return: None
        """
        if molecule not in self.molecules:
            raise SpecValueError(f"Molecule {molecule} not specified")
        if cell_type not in self.spec_dict["densities"][molecule].keys():
            raise SpecValueError(f"Molecule {molecule} density not specified for cell type {cell_type}")
        x: dict = self.spec_dict["densities"][molecule]
        x.pop(cell_type)

    def density_new(self, molecule: str, cell_type: str, density: float) -> AdhesionMoleculeDensitySpecs:
        """
        Appends and returns a new adhesion molecule density spec

        :param str molecule: name of molecule
        :param str cell_type: name of cell type
        :param float density: molecule density
        :return: new adhesion molecule density spec
        :rtype: AdhesionMoleculeDensitySpecs
        """
        p = AdhesionMoleculeDensitySpecs(molecule=molecule, cell_type=cell_type, density=density)
        self.density_append(p)
        return p

    def formula_append(self, _formula: AdhesionFlexBindingFormulaSpecs) -> None:
        """
        Append a binding formula spec

        :param AdhesionFlexBindingFormulaSpecs _formula: binding formula spec
        :return: None
        """
        if _formula.formula_name in self.formulas:
            raise SpecValueError(f"Formula with name {_formula.formula_name} already specified")
        self.spec_dict["binding_formulas"][_formula.formula_name] = _formula

    def formula_remove(self, _formula_name: str) -> None:
        """
        Remove a new binding formula spec

        :param str _formula_name: name of formula
        :return: None
        """
        if _formula_name not in self.formulas:
            raise SpecValueError(f"Formula with name {_formula_name} not specified")
        self.spec_dict["binding_formulas"].pop(_formula_name)

    def formula_new(self, formula_name: str, formula: str) -> AdhesionFlexBindingFormulaSpecs:
        """
        Append and return a new binding formula spec

        :param str formula_name: name of forumla
        :param str formula: formula
        :return: new binding formula spec
        :rtype: AdhesionFlexBindingFormulaSpecs
        """
        p = AdhesionFlexBindingFormulaSpecs(formula_name=formula_name, formula=formula)
        self.formula_append(p)
        return p


class _AdhesionFlexDensityAccessor(_PyCoreSpecsBase):
    """
    AdhesionFlex density accessor

    Container with convenience containers for :class:`AdhesionFlexPluginSpecs` to instances of
    :class:`AdhesionMoleculeDensitySpecs`
    """

    def __init__(self, _plugin_spec: AdhesionFlexPluginSpecs):
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
    def cell_type(self) -> _PyCoreParamAccessor[Dict[str, AdhesionMoleculeDensitySpecs]]:
        """

        :return: accessor to adnesion molecule densities with cell type names as keys
        :rtype: dict of [str: dict of [str: AdhesionMoleculeDensitySpecs]]
        """
        return _PyCoreParamAccessor(self, "densities")

    @property
    def molecule(self) -> _PyCoreParamAccessor[Dict[str, AdhesionMoleculeDensitySpecs]]:
        """

        :return: accessor to adnesion molecule densities with molecule names as keys
        :rtype: dict of [str: dict of [str: AdhesionMoleculeDensitySpecs]]
        """
        return _PyCoreParamAccessor(self._plugin_spec, "densities")


# LengthConstraint Plugin


class LengthEnergyParametersSpecs(_PyCoreSpecsBase):
    """ Length Energy Parameters """

    def __init__(self,
                 _cell_type: str,
                 target_length: float,
                 lambda_length: float,
                 minor_target_length: float = None):
        """

        :param str _cell_type: cell type name
        :param float target_length: target length
        :param float lambda_length: lambda length
        :param float minor_target_length: minor target length, optional
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

    minor_target_length: Union[float, None] = SpecProperty(name="minor_target_length")
    """minor target length, Optional, None if not set"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el


class LengthConstraintPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    LengthConstraint Plugin

    LengthEnergyParametersSpecs instances can be accessed as follows,

    .. code-block:: python

       spec: LengthConstraintPluginSpecs
       cell_type_name: str
       params: LengthEnergyParametersSpecs = spec[cell_type_name]
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: LengthConstraintPluginSpecs
        """
        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("LengthEnergyParameters")

        for p_el in el_list:
            p_el: CC3DXMLElement
            kwargs = {"target_length": p_el.getAttributeAsDouble("TargetLength"),
                      "lambda_length": p_el.getAttributeAsDouble("LambdaLength")}
            if p_el.findAttribute("MinorTargetLength"):
                kwargs["minor_target_length"] = p_el.getAttributeAsDouble("MinorTargetLength")
            o.param_new(p_el.getAttribute("CellType"), **kwargs)

        return o

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        :rtype: list of str
        """
        return list(self.spec_dict["param_specs"].keys())

    def __getitem__(self, item):
        if item not in self.cell_types:
            raise SpecValueError(f"Cell type {item} not specified")
        return self.spec_dict["param_specs"][item]

    def param_append(self, param_spec: LengthEnergyParametersSpecs):
        """
        Appens a length energy parameters spec

        :param param_spec: length energy parameters spec
        :type param_spec: LengthEnergyParametersSpecs
        :return: None
        """
        if param_spec.cell_type in self.cell_types:
            raise SpecValueError(f"Type name {param_spec.cell_type} already specified")
        self.spec_dict["param_specs"][param_spec.cell_type] = param_spec

    def param_remove(self, _cell_type: str):
        """
        Removes a length energy parameters spec

        :param _cell_type: name of cell type
        :type _cell_type: str
        :return: None
        """
        if _cell_type not in self.cell_types:
            raise SpecValueError(f"Type name {_cell_type} not specified")
        self.spec_dict["param_specs"].pop(_cell_type)

    def param_new(self,
                  _cell_type: str,
                  target_length: float,
                  lambda_length: float,
                  minor_target_length: float = None) -> LengthEnergyParametersSpecs:
        """
        Appends and returns a new length energy parameters spec

        :param str _cell_type: cell type name
        :param float target_length: target length
        :param float lambda_length: lambda length
        :param float minor_target_length: minor target length, optional
        :return: new length energy parameters spec
        :rtype: LengthEnergyParametersSpecs
        """
        param_spec = LengthEnergyParametersSpecs(_cell_type,
                                                 target_length=target_length,
                                                 lambda_length=lambda_length,
                                                 minor_target_length=minor_target_length)
        self.param_append(param_spec=param_spec)
        return param_spec


class ConnectivityPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """ Connectivity Plugin """

    name = "connectivity"
    registered_name = "Connectivity"

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ConnectivityPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class ConnectivityGlobalPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """ ConnectivityGlobal Plugin """

    name = "connectivity_global"
    registered_name = "ConnectivityGlobal"

    def __init__(self, fast: bool = False, *_cell_types):
        super().__init__()

        for _ct in _cell_types:
            if not isinstance(_ct, str):
                raise SpecValueError("Only cell type names as strings can be pass")

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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        if self.fast:
            self._el.ElementCC3D("FastAlgorithm")
        for _ct in self.cell_types:
            self._el.ElementCC3D("ConnectivityOn", {"Type": _ct})
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ConnectivityGlobalPluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)
        el_list = el.getElements("ConnectivityOn")
        return cls(fast=el.findAttribute("FastAlgorithm"),
                   *[e.getAttribute("Type") for e in el_list])

    @property
    def cell_types(self) -> List[str]:
        """

        :return: list of cell type names
        :rtype: list of str
        """
        return [x for x in self.spec_dict["cell_types"]]

    def cell_type_append(self, _name: str) -> None:
        """
        Appends a cell type name

        :param _name: name of cell type
        :type _name: str
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
        :type _name: str
        :raises SpecValueError: when cell type name has not been specified
        :return: None
        """
        if _name not in self.cell_types:
            raise SpecValueError(f"Type name {_name} not specified")
        self.spec_dict["cell_types"].remove(_name)


# Secretion Plugin


class SecretionFieldSpecs(_PyCoreSpecsBase):
    """ Secretion Field Specs """

    check_dict = {"frequency": (lambda x: x < 1, "Frequency must be positive")}

    def __init__(self, field_name: str, frequency: int = 1, *_param_specs):
        """

        :param str field_name: name of field
        :param int frequency: frequency of calls per step
        :param _param_specs: variable number of SecretionSpecs instances, optional
        """
        super().__init__()

        for _ps in _param_specs:
            if not isinstance(_ps, SecretionSpecs):
                raise SpecValueError("Only SecretionSpecs instances can be passed")

        self.spec_dict = {"field_name": field_name,
                          "frequency": frequency,
                          "param_specs": []}
        [self.param_append(_ps) for _ps in _param_specs]

    field_name: str = SpecProperty(name="field_name")
    """name of field"""

    frequency: int = SpecProperty(name="frequency")
    """frequency of calls per step"""

    specs: List[SecretionSpecs] = SpecProperty(name="param_specs")
    """secretion field specs"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        attr = {"Name": self.field_name}
        if self.frequency > 1:
            attr["ExtraTimesPerMC"] = str(self.frequency)
        self._el = ElementCC3D("Field", attr)
        [self._el.add_child(_ps.xml) for _ps in self.spec_dict["param_specs"]]
        return self._el

    def param_append(self, _ps: SecretionSpecs) -> None:
        """
        Append a secretion spec

        :param SecretionSpecs _ps: secretion spec
        :return: None
        """
        if _ps.contact_type is None:
            for x in self.spec_dict["param_specs"]:
                x: SecretionSpecs
                if x.contact_type is None and _ps.cell_type == x.cell_type:
                    raise SpecValueError(f"Duplicate specification for cell type {_ps.cell_type}")
        self.spec_dict["param_specs"].append(_ps)

    def param_remove(self, _cell_type: str, contact_type: str = None):
        """
        Remove a secretion spec

        :param str _cell_type: name of cell type
        :param contact_type: name of on-contact dependent cell type, optional
        :return: None
        """
        for _ps in self.spec_dict["param_specs"]:
            _ps: SecretionSpecs
            if _ps.cell_type == _cell_type and contact_type == _ps.contact_type:
                self.spec_dict["param_specs"].remove(_ps)
                return
        raise SpecValueError("SecretionSpecs not specified")

    def param_new(self,
                  _cell_type: str,
                  _val: float,
                  constant: bool = False,
                  contact_type: str = None) -> SecretionSpecs:
        """
        Append and return a new secretion spec

        :param str _cell_type: cell type name
        :param float _val: value of parameter
        :param bool constant: flag for constant concentration, option
        :param str contact_type: name of cell type for on-contact dependence, optional
        :return: new secretion spec
        :rtype: SecretionSpecs
        """
        ps = SecretionSpecs(_cell_type, _val, constant=constant, contact_type=contact_type)
        self.param_append(ps)
        return ps


class SecretionPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    Secretion Plugin

    SecretionSpecs instances can be accessed by field name and cell type name as follows,

    .. code-block:: python

       spec: SecretionPluginSpecs
       field_name: str
       cell_type_name: str
       field_specs: SecretionFieldSpecs = spec.fields[field_name]
       params: SecretionSpecs = field_specs.specs[cell_type_name]

    """

    name = "secretion"
    registered_name = "Secretion"

    def __init__(self, pixel_tracker: bool = True, boundary_pixel_tracker: bool = True, *_field_spec):
        super().__init__()

        for _fs in _field_spec:
            if not isinstance(_fs, SecretionFieldSpecs):
                raise SpecValueError("Only SecretionFieldSpecs instances can be passed")

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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        if not self.pixel_tracker:
            self._el.ElementCC3D("DisablePixelTracker")
        if not self.boundary_pixel_tracker:
            self._el.ElementCC3D("DisableBoundaryPixelTracker")
        [self._el.add_child(_fs.xml) for _fs in self.spec_dict["field_specs"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: SecretionPluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls(pixel_tracker=not el.findElement("DisablePixelTracker"),
                boundary_pixel_tracker=not el.findElement("DisableBoundaryPixelTracker"))

        el_list = el.getElements("Field")

        for f_el in el_list:
            f_el: CC3DXMLElement
            f = SecretionFieldSpecs(field_name=f_el.getAttribute("Name"))

            if f_el.findAttribute("ExtraTimesPerMC"):
                f.frequency = f_el.getAttributeAsInt("ExtraTimesPerMC")

            f_el_list = f_el.getElements("Secretion")

            for p_el in f_el_list:
                p_el: CC3DXMLElement
                f.param_append(SecretionSpecs(p_el.getAttribute("Type"), p_el.getDouble()))

            f_el_list = f_el.getElements("ConstantConcentration")

            for p_el in f_el_list:
                f.param_append(SecretionSpecs(p_el.getAttribute("Type"), p_el.getDouble(),
                                              constant=True))

            f_el_list = f_el.getElements("SecretionOnContact")

            for p_el in f_el_list:
                f.param_append(SecretionSpecs(p_el.getAttribute("Type"), p_el.getDouble(),
                                              contact_type=p_el.getAttribute("SecreteOnContactWith")))

            o.field_append(f)

        return o

    @property
    def fields(self) -> _PyCoreParamAccessor[SecretionFieldSpecs]:
        """

        :return: accessor to secretion field specs with field names as keys
        :rtype: dict of [str: SecretionFieldSpecs]
        """
        return _PyCoreParamAccessor(self, "field_specs")

    @property
    def field_names(self) -> List[str]:
        """

        :return: list of registered field names
        :rtype: list of str
        """
        return list(self.spec_dict["field_specs"].keys())

    def field_append(self, _fs: SecretionFieldSpecs) -> None:
        """
        Append a secretion field spec

        :param SecretionFieldSpecs _fs: secretion field spec
        :return: None
        """
        if _fs.field_name in self.field_names:
            raise SpecValueError(f"Field specs for field {_fs.field_name} already specified")
        self.spec_dict["field_specs"][_fs.field_name] = _fs

    def field_remove(self, _field_name: str) -> None:
        """
        Remove a secretion field spec

        :param str _field_name: field name
        :return: None
        """
        if _field_name not in self.field_names:
            raise SpecValueError(f"Field specs for field {_field_name} not specified")
        self.spec_dict["field_specs"].pop(_field_name)

    def field_new(self, field_name: str, frequency: int = 1, *_param_specs) -> SecretionFieldSpecs:
        """
        Append and return a new secretion field spec

        :param str field_name: name of field
        :param int frequency: frequency of calls per step
        :param _param_specs: variable number of SecretionSpecs instances, optional
        :return: new secretion field spec
        :rtype: SecretionFieldSpecs
        """
        fs = SecretionFieldSpecs(field_name=field_name, frequency=frequency, *_param_specs)
        self.field_append(fs)
        return fs


# FocalPointPlasticity Plugin


class LinkConstituentLawSpecs(_PyCoreSpecsBase):
    """
    Link Constituent Law

    Usage is as follows

    .. code-block:: python
       lcl: LinkConstituentLawSpecs
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
        :rtype: ElementCC3D
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
        :rtype: dict of [str: float]
        """
        return _PyCoreParamAccessor(self, "variables")


class FocalPointPlasticityParamSpec(_PyCoreSpecsBase):
    """ FocalPointPlasticityParamSpec """

    check_dict = {
        "target_distance": (lambda x: x < 0, "Target distance must be non-negative"),
        "max_distance": (lambda x: x < 0, "Maximum distance must be non-negative"),
        "max_junctions": (lambda x: x < 1, "Maximum number of junctions must be positive"),
        "neighbor_order": (lambda x: not 0 < x < 5, "Invalid neighbor order. Must be in [1, 5]")
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
                 law: LinkConstituentLawSpecs = None):
        """

        :param str _type1: second type name
        :param str _type2: first type name
        :param float lambda_fpp: lambda value
        :param float activation_energy: activation energy
        :param float target_distance: target distance
        :param float max_distance: max distance
        :param int max_junctions: maximum number of junctions, defaults to 1
        :param int neighbor_order: neighbor order, defaults to 1
        :param bool internal: flag for internal parameter, defaults to False
        :param LinkConstituentLawSpecs law: link constitutive law, optional
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

    law: Union[LinkConstituentLawSpecs, None] = SpecProperty(name="law")
    """link constitute law, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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


class FocalPointPlasticityPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    FocalPointPlasticity Plugin

    FocalPointPlasticityParamSpec instances can be accessed as follows,

    .. code-block:: python

       spec: FocalPointPlasticityPluginSpecs
       cell_type_1: str
       cell_type_2: str
       param: FocalPointPlasticityParamSpec = spec.link[cell_type_1][cell_type_2]

    """

    name = "focal_point_plasticity"
    registered_name = "FocalPointPlasticity"

    check_dict = {"neighbor_order": (lambda x: not 0 < x < 5, "Invalid neighbor order. Must be in [1, 5]")}

    def __init__(self, neighbor_order: int = 1, *_params):
        super().__init__()

        for _p in _params:
            if not isinstance(_p, FocalPointPlasticityParamSpec):
                raise SpecValueError("Only FocalPointPlasticityParamSpec instances can be passed")

        self.spec_dict = {"neighbor_order": neighbor_order,
                          "param_specs": {}}
        [self.param_append(_ps) for _ps in _params]

    neighbor_order: int = SpecProperty(name="neighbor_order")
    """neighbor order"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        for t1 in self.spec_dict["param_specs"].keys():
            for t2 in self.spec_dict["param_specs"][t1].keys():
                self._el.add_child(self.spec_dict["param_specs"][t1][t2].xml)
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: FocalPointPlasticityPluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        if el.findElement("NeighborOrder"):
            o.neighbor_order = el.getFirstElement("NeighborOrder").getInt()

        el_list = el.getElements("Parameters")

        for p_el in el_list:
            p_el: CC3DXMLElement
            p = FocalPointPlasticityParamSpec(p_el.getAttribute("Type1"),
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
                law = LinkConstituentLawSpecs(formula=l_el.getFirstElement("Formula").getText())

                l_el_list = l_el.getElements("Variable")

                for v_el in l_el_list:
                    v_el: CC3DXMLElement
                    law.variable[v_el.getAttribute("Name")] = v_el.getAttributeAsDouble("Value")
                p.law = law

            o.param_append(p)

        el_list = el.getElements("InternalParameters")

        for p_el in el_list:
            p_el: CC3DXMLElement
            p = FocalPointPlasticityParamSpec(p_el.getAttribute("Type1"),
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
                law = LinkConstituentLawSpecs(formula=l_el.getFirstElement("Formula").getText())

                l_el_list = l_el.getElements("Variable")

                for v_el in l_el_list:
                    v_el: CC3DXMLElement
                    law.variable[v_el.getAttribute("Name")] = v_el.getAttributeAsDouble("Value")
                p.law = law

            o.param_append(p)

        return o

    @property
    def links(self) -> Tuple[str, str]:
        """

        :return: next tuple of current fpp parameter cell type name combinations
        :rtype: (str, str)
        """
        for v in self.spec_dict["param_specs"].values():
            for vv in v.values():
                yield vv.type1, vv.type2

    @property
    def link(self) -> _PyCoreParamAccessor[Dict[str, FocalPointPlasticityParamSpec]]:
        """

        :return: accessor to focal point plasticity parameters with cell types as keys
        :rtype: dict of [str: dict of [str: FocalPointPlasticityParamSpec]]
        """
        return _PyCoreParamAccessor(self, "param_specs")

    def param_append(self, _ps: FocalPointPlasticityParamSpec) -> None:
        """
        Append a focal point plasticity parameter spec

        :param FocalPointPlasticityParamSpec _ps: focal point plasticity parameter spec
        :return: None
        """
        if _ps.type1 in self.spec_dict["param_specs"].keys() \
                and _ps.type2 in self.spec_dict["param_specs"][_ps.type1].keys():
            raise SpecValueError(f"Parameter already specified for types ({_ps.type1}, {_ps.type2})")
        if _ps.type1 not in self.spec_dict["param_specs"].keys():
            self.spec_dict["param_specs"][_ps.type1] = dict()
        self.spec_dict["param_specs"][_ps.type1][_ps.type2] = _ps

    def param_remove(self, type_1: str, type_2: str) -> None:
        """
        Remove a focal point plasticity parameter spec

        :param str type_1: name of first cell type
        :param str type_2: name of second cell type
        :return: None
        """
        if type_1 not in self.spec_dict["param_specs"].keys() \
                or type_2 not in self.spec_dict["param_specs"][type_1].keys():
            raise SpecValueError(f"Parameter not specified for types ({type_1}, {type_2})")
        self.spec_dict["param_specs"][type_1].pop(type_2)

    def param_new(self,
                  _type1: str,
                  _type2: str,
                  lambda_fpp: float,
                  activation_energy: float,
                  target_distance: float,
                  max_distance: float,
                  max_junctions: int = 1,
                  neighbor_order: int = 1,
                  internal: bool = False,
                  law: LinkConstituentLawSpecs = None) -> FocalPointPlasticityParamSpec:
        """
        Append and return a new focal point plasticity parameter spec

        :param str _type1: second type name
        :param str _type2: first type name
        :param float lambda_fpp: lambda value
        :param float activation_energy: activation energy
        :param float target_distance: target distance
        :param float max_distance: max distance
        :param int max_junctions: maximum number of junctions, defaults to 1
        :param int neighbor_order: neighbor order, defaults to 1
        :param bool internal: flag for internal parameter, defaults to False
        :param LinkConstituentLawSpecs law: link constitutive law, optional
        :return: new focal point plasticity parameter spec
        :rtype: FocalPointPlasticityParamSpec
        """
        p = FocalPointPlasticityParamSpec(_type1,
                                          _type2,
                                          lambda_fpp=lambda_fpp,
                                          activation_energy=activation_energy,
                                          target_distance=target_distance,
                                          max_distance=max_distance,
                                          max_junctions=max_junctions,
                                          neighbor_order=neighbor_order,
                                          internal=internal,
                                          law=law)
        self.param_append(p)
        return p


# Curvature Plugin


class CurvatureInternalParamSpecs(_PyCoreSpecsBase):
    """ Curvature Internal Parameter Spec """

    def __init__(self,
                 _type1: str,
                 _type2: str,
                 _lambda_curve: float,
                 _activation_energy: float):
        """

        :param str _type1: name of first cell type
        :param str _type2: name of second cell type
        :param float _lambda_curve: lambda value
        :param float _activation_energy: activation energy
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
        :rtype: ElementCC3D
        """
        self._el = ElementCC3D("InternalParameters", {"Type1": self.type1, "Type2": self.type2})
        self._el.ElementCC3D("Lambda", {}, str(self.lambda_curve))
        self._el.ElementCC3D("ActivationEnergy", {}, str(self.activation_energy))
        return self._el


class CurvatureInternalTypeParameters(_PyCoreSpecsBase):
    """ CurvatureInternalTypeParameters """

    check_dict = {
        "max_junctions": (lambda x: x < 1, "Maximum number of junctions must be positive"),
        "neighbor_order": (lambda x: not 0 < x < 5, "Invalid neighbor order. Must be in [1, 5]")
    }

    def __init__(self,
                 _cell_type: str,
                 _max_junctions: int,
                 _neighbor_order: int):
        """

        :param str _cell_type: name of cell type
        :param int _max_junctions: maximum number of junctions
        :param int _neighbor_order: neighbor order
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
        :rtype: ElementCC3D
        """
        self._el = ElementCC3D("Parameters",
                               {"TypeName": self.cell_type,
                                "MaxNumberOfJunctions": self.max_junctions,
                                "NeighborOrder": self.neighbor_order})
        return self._el


class CurvaturePluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface, _PyCoreSteerableInterface):
    """
    Curvature Plugin

    CurvatureInternalParamSpecs instances can be accessed as follows,

    .. code-block:: python

       spec: CurvaturePluginSpecs
       cell_type_1: str
       cell_type_2: str
       param: CurvatureInternalParamSpecs = spec.internal_param[cell_type_1][cell_type_2]

    CurvatureInternalTypeParameters instances can be accessed as follows,

    .. code-block:: python

       spec: CurvaturePluginSpecs
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        for v in self.spec_dict["param_specs"].values():
            [self._el.add_child(vv.xml) for vv in v.values()]
        if len(self.spec_dict["type_spec"].keys()) > 0:
            el = self._el.ElementCC3D("InternalTypeSpecificParameters")
            [el.add_child(x.xml) for x in self.spec_dict["type_spec"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: CurvaturePluginSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = el.getElements("InternalParameters")

        for p_el in el_list:
            p_el: CC3DXMLElement
            o.param_internal_new(p_el.getAttribute("Type1"),
                                 p_el.getAttribute("Type2"),
                                 p_el.getFirstElement("Lambda").getDouble(),
                                 p_el.getFirstElement("ActivationEnergy").getDouble())

        if el.findElement("InternalTypeSpecificParameters"):
            t_el: CC3DXMLElement = el.getFirstElement("InternalTypeSpecificParameters")

            t_el_list = t_el.getElements("Parameters")

            for pt_el in t_el_list:
                pt_el: CC3DXMLElement
                o.param_type_new(pt_el.getAttribute("TypeName"),
                                 pt_el.getAttributeAsInt("MaxNumberOfJunctions"),
                                 pt_el.getAttributeAsInt("NeighborOrder"))

        return o

    @property
    def internal_param(self) -> _PyCoreParamAccessor[Dict[str, CurvatureInternalParamSpecs]]:
        """

        :return: accessor to curvature parameters with cell types as keys
        :rtype: dict of [str: dict of [str: CurvatureInternalParamSpecs]]
        """
        return _PyCoreParamAccessor(self, "param_specs")

    @property
    def type_param(self) -> _PyCoreParamAccessor[CurvatureInternalTypeParameters]:
        """

        :return: accessor to type parameters with cell types as keys
        :rtype: dict of [str: CurvatureInternalTypeParameters]
        """
        return _PyCoreParamAccessor(self, "type_spec")

    @property
    def internal_params(self) -> Tuple[str, str]:
        """

        :return: next tuple of current internal parameter cell type name combinations
        :rtype: (str, str)
        """
        for v in self.spec_dict["param_specs"].values():
            for vv in v.values():
                yield vv.type1, vv.type2

    @property
    def type_params(self) -> str:
        """

        :return: next current type parameters cell type names
        :rtype: list of str
        """
        for x in self.spec_dict["type_spec"].values():
            yield x.cell_type

    def param_internal_append(self, _ps: CurvatureInternalParamSpecs) -> None:
        """
        Append a curvature internal parameter spec

        :param CurvatureInternalParamSpecs _ps: curvature internal parameter spec
        :return: None
        """
        if _ps.type1 in self.spec_dict["param_specs"].keys() \
                and _ps.type2 in self.spec_dict["param_specs"][_ps.type1].keys():
            raise SpecValueError(f"Parameter already specified for types ({_ps.type1}, {_ps.type2})")
        if _ps.type1 not in self.spec_dict["param_specs"].keys():
            self.spec_dict["param_specs"][_ps.type1] = dict()
        self.spec_dict["param_specs"][_ps.type1][_ps.type2] = _ps

    def param_internal_remove(self, _type1: str, _type2: str) -> None:
        """
        Remove a curvature internal parameter spec

        :param str _type1: name of fist cell type
        :param str _type2: name of second cell type
        :return: None
        """
        if _type1 not in self.spec_dict["param_specs"].keys() \
                or _type2 not in self.spec_dict["param_specs"][_type1].keys():
            raise SpecValueError(f"Parameter not specified for types ({_type1}, {_type2})")
        self.spec_dict["param_specs"][_type1].pop(_type2)

    def param_internal_new(self,
                           _type1: str,
                           _type2: str,
                           _lambda_curve: float,
                           _activation_energy: float) -> CurvatureInternalParamSpecs:
        """
        Append and return a new curvature internal parameter spec

        :param str _type1: name of first cell type
        :param str _type2: name of second cell type
        :param float _lambda_curve: lambda value
        :param float _activation_energy: activation energy
        :return: new curvature internal parameter spec
        :rtype: CurvatureInternalParamSpecs
        """
        p = CurvatureInternalParamSpecs(_type1, _type2, _lambda_curve, _activation_energy)
        self.param_internal_append(p)
        return p

    def param_type_append(self, _ps: CurvatureInternalTypeParameters) -> None:
        """
        Append a curvature internal type parameters
        :param CurvatureInternalTypeParameters _ps: curvature internal type parameters
        :return: None
        """
        if _ps.cell_type in self.spec_dict["type_spec"].keys():
            raise SpecValueError(f"Parameter already specified for type {_ps.cell_type}")
        self.spec_dict["type_spec"][_ps.cell_type] = _ps

    def param_type_remove(self, _cell_type: str) -> None:
        """
        Remove a curvature internal type parameters

        :param str _cell_type: name of cell type
        :return: None
        """
        if _cell_type not in self.spec_dict["type_spec"].keys():
            raise SpecValueError(f"Parameter not specified for type {_cell_type}")
        self.spec_dict["type_spec"].pop(_cell_type)

    def param_type_new(self,
                       _cell_type: str,
                       _max_junctions: int,
                       _neighbor_order: int) -> CurvatureInternalTypeParameters:
        """
        Append and return a new curvature internal type parameters

        :param str _cell_type: name of cell type
        :param int _max_junctions: maximum number of junctions
        :param int _neighbor_order: neighbor order
        :return: new curvature internal type parameters
        :rtype: CurvatureInternalTypeParameters
        """
        p = CurvatureInternalTypeParameters(_cell_type, _max_junctions, _neighbor_order)
        self.param_type_append(p)
        return p


class BoundaryPixelTrackerPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
    """ BoundaryPixelTracker Plugin """

    name = "boundary_pixel_tracker"
    registered_name = "BoundaryPixelTracker"

    check_dict = {"neighbor_order": (lambda x: not 0 < x < 5, "Invalid neighbor order. Must be in [1, 5]")}

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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        if self.neighbor_order > 1:
            self._el.ElementCC3D("NeighborOrder", {}, str(self.neighbor_order))
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: BoundaryPixelTrackerPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class PixelTrackerPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        if self.track_medium:
            self._el.ElementCC3D("TrackMedium")
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: PixelTrackerPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class MomentOfInertiaPluginSpecs(_PyCorePluginSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: MomentOfInertiaPluginSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


class BoxWatcherSteppableSpecs(_PyCoreSteppableSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: BoxWatcherSteppableSpecs
        """
        cls.find_xml_by_attr(_xml)
        return cls()


# Blob Initializer


class BlobRegionSpecs(_PyCoreSpecsBase):
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
                 center: Point3D = Point3D(0, 0, 0),
                 cell_types: List[str] = None):
        """

        :param int gap: blob gap
        :param int width: width of cells
        :param int radius: blob radius
        :param Point3D center: blob center point
        :param List[str] cell_types: names of cell types in blob
        """
        super().__init__()

        if cell_types is None:
            cell_types = []

        self.spec_dict = {"gap": gap,
                          "width": width,
                          "radius": radius,
                          "center": center,
                          "cell_types": cell_types}

    gap: int = SpecProperty(name="gap")
    """blob gap"""

    width: int = SpecProperty(name="width")
    """cell width"""

    radius: int = SpecProperty(name="radius")
    """blob radius"""

    center: Point3D = SpecProperty(name="center")
    """blob center point"""

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("Region")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("Gap", {}, str(self.gap))
        self._el.ElementCC3D("Width", {}, str(self.width))
        self._el.ElementCC3D("Radius", {}, str(self.radius))
        self._el.ElementCC3D("Center", {"x": str(self.center.x), "y": str(self.center.y), "z": str(self.center.z)})
        self._el.ElementCC3D("Types", {}, ",".join(self.spec_dict["cell_types"]))
        return self._el


class BlobInitializerSpecs(_PyCoreSteppableSpecs, _PyCoreXMLInterface):
    """ BlobInitializer """

    name = "blob_initializer"
    registered_name = "BlobInitializer"

    def __init__(self, *_regions):
        super().__init__()

        for r in _regions:
            if not isinstance(r, BlobRegionSpecs):
                raise SpecValueError("Only BlobRegionSpecs instances can be passed")

        self.spec_dict = {"regions": list(_regions)}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(reg.xml) for reg in self.spec_dict["regions"]]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: BlobInitializerSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("Region")

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

    def region_append(self, _reg: BlobRegionSpecs) -> None:
        """
        Append a region

        :param BlobRegionSpecs _reg: a region
        :return: None
        """
        self.spec_dict["regions"].append(_reg)

    def region_pop(self, _idx: int) -> None:
        """
        Remove a region by index

        :param int _idx: index of region to append
        :return: None
        """
        self.spec_dict["regions"].pop(_idx)

    def region_new(self,
                   gap: int = 0,
                   width: int = 0,
                   radius: int = 0,
                   center: Point3D = Point3D(0, 0, 0),
                   cell_types: List[str] = None) -> BlobRegionSpecs:
        """
        Appends and returns a blob region

        :param int gap: blob gap
        :param int width: width of cells
        :param int radius: blob radius
        :param Point3D center: blob center point
        :param List[str] cell_types: names of cell types in blob
        :return: new blob region
        :rtype: BlobRegionSpecs
        """
        reg = BlobRegionSpecs(gap=gap, width=width, radius=radius, center=center, cell_types=cell_types)
        self.region_append(reg)
        return reg


# Uniform Initializer


class UniformRegionSpecs(_PyCoreSpecsBase):
    """ Uniform Initializer Region Specs """

    check_dict = {
        "gap": (lambda x: x < 0, "Gap must be greater non-negative"),
        "width": (lambda x: x < 1, "Width must be positive")
    }

    def __init__(self,
                 pt_min: Point3D,
                 pt_max: Point3D,
                 gap: int = 0,
                 width: int = 0,
                 cell_types: List[str] = None):
        """

        :param Point3D pt_min: minimum box point
        :param Point3D pt_max: maximum box point
        :param int gap: blob gap
        :param int width: width of cells
        :param List[str] cell_types: names of cell types in region
        """
        super().__init__()

        if cell_types is None:
            cell_types = []

        self.spec_dict = {"pt_min": pt_min,
                          "pt_max": pt_max,
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

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("Region")

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()

        self._el.ElementCC3D("BoxMin", {c: str(getattr(self.pt_min, c)) for c in ["x", "y", "z"]})
        self._el.ElementCC3D("BoxMax", {c: str(getattr(self.pt_max, c)) for c in ["x", "y", "z"]})
        self._el.ElementCC3D("Gap", {}, str(self.gap))
        self._el.ElementCC3D("Width", {}, str(self.width))
        self._el.ElementCC3D("Types", {}, ",".join(self.spec_dict["cell_types"]))
        return self._el


class UniformInitializerSpecs(_PyCoreSteppableSpecs, _PyCoreXMLInterface):
    """ Uniform Initializer Specs """

    name = "uniform_initializer"
    registered_name = "UniformInitializer"

    def __init__(self, *_regions):
        super().__init__()

        for r in _regions:
            if not isinstance(r, UniformRegionSpecs):
                raise SpecValueError("Only UniformRegionSpecs instances can be passed")

        self.spec_dict = {"regions": list(_regions)}

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(reg.xml) for reg in self.spec_dict["regions"]]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: UniformInitializerSpecs
        """

        o = cls()

        el_list = cls.find_xml_by_attr(_xml).getElements("Region")

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

    def region_append(self, _reg: UniformRegionSpecs) -> None:
        """
        Append a region

        :param UniformRegionSpecs _reg: a region
        :return: None
        """
        self.spec_dict["regions"].append(_reg)

    def region_pop(self, _idx: int) -> None:
        """
        Remove a region by index

        :param int _idx: index of region to append
        :return: None
        """
        self.spec_dict["regions"].pop(_idx)

    def region_new(self,
                   pt_min: Point3D,
                   pt_max: Point3D,
                   gap: int = 0,
                   width: int = 0,
                   cell_types: List[str] = None) -> UniformRegionSpecs:
        """
        Appends and returns a uniform region

        :param Point3D pt_min: minimum box point
        :param Point3D pt_max: maximum box point
        :param int gap: blob gap
        :param int width: width of cells
        :param List[str] cell_types: names of cell types in region
        :return: new blob region
        :rtype: UniformRegionSpecs
        """
        reg = UniformRegionSpecs(pt_min=pt_min, pt_max=pt_max, gap=gap, width=width, cell_types=cell_types)
        self.region_append(reg)
        return reg


class PIFInitializerSteppableSpecs(_PyCoreSteppableSpecs, _PyCoreXMLInterface):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("PIFName", {}, self.pif_name)
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: PIFInitializerSteppableSpecs
        """
        return cls(pif_name=cls.find_xml_by_attr(_xml).getFirstElement("PIFName").getText())


class PIFDumperSteppableSpecs(_PyCoreSteppableSpecs, _PyCoreXMLInterface):
    """ PIFDumper Steppable """

    name = "pif_dumper"
    registered_name = "PIFDumper"

    check_dict = {
        "pif_name": (lambda x: len(x) == 0, "PIF needs a name"),
        "frequency": (lambda x: x < 1, "Frequency must be positive")
    }

    def __init__(self, pif_name: str, frequency: int = 0):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        self._el.ElementCC3D("PIFName", {}, self.pif_name)
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: PIFDumperSteppableSpecs
        """
        el = cls.find_xml_by_attr(_xml)
        return cls(pif_name=el.getFirstElement("PIFName").getText(), frequency=el.getAttributeAsInt("Frequency"))


# DiffusionSolverFE


class DiffusionSolverFEDiffusionDataSpecs(_PDEDiffusionDataSpecs):
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
        :rtype: ElementCC3D
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
    def diff_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to diffusion coefficient values by cell type name
        :rtype: dict of [str: float]
        """
        return _PyCoreParamAccessor(self, "diff_types")

    @property
    def decay_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to decay coefficient values by cell type name
        :rtype: dict of [str: float]
        """
        return _PyCoreParamAccessor(self, "decay_types")


class DiffusionSolverFESecretionDataSpecs(_PDESecretionDataSpecs):
    """ Secretion Data Specs for DiffusionSolverFE """


class DiffusionSolverFEFieldSpecs(_PDESolverFieldSpecs[DiffusionSolverFEDiffusionDataSpecs,
                                                       DiffusionSolverFESecretionDataSpecs]):
    """ DiffusionSolverFE Field Specs """

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> None:
        """
        Specify a new secretion data spec

        :param str _cell_type: name of cell type
        :param float _val: value
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        try:
            constant = kwargs["constant"]
        except KeyError:
            constant = False
        _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val, constant=constant, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param str _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class DiffusionSolverFESpecs(_PDESolverSpecs[DiffusionSolverFEDiffusionDataSpecs, DiffusionSolverFESecretionDataSpecs],
                             _PyCoreXMLInterface):
    """ DiffusionSolverFE """

    name = "diffusion_solver_fe"

    _field_spec = DiffusionSolverFEFieldSpecs
    _diff_data = DiffusionSolverFEDiffusionDataSpecs
    _secr_data = DiffusionSolverFESecretionDataSpecs

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
        :rtype: str
        """
        if self.gpu:
            return "DiffusionSolverFE_OpenCL"
        return "DiffusionSolverFE"

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("Steppable", {"Type": self.registered_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: DiffusionSolverFESpecs
        """
        o = cls()
        try:
            el = o.find_xml_by_attr(_xml)
        except SpecImportError:
            o.gpu = not o.gpu
            el = o.find_xml_by_attr(_xml)

        o.fluc_comp = el.findElement("FluctuationCompensator")

        el_list = el.getElements("DiffusionField")

        for f_el in el_list:
            f_el: CC3DXMLElement

            f = o.field_new(f_el.getAttribute("Name"))

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()

            el_list = dd_el.getElements("DiffusionCoefficient")

            for t_el in el_list:
                f.diff_data.diff_types[t_el.getAttribute("CellType")] = t_el.getDouble()

            el_list = dd_el.getElements("DecayCoefficient")

            for t_el in el_list:
                f.diff_data.decay_types[t_el.getAttribute("CellType")] = t_el.getDouble()
            if dd_el.findElement("InitialConcentrationExpression"):
                f.diff_data.init_expression = dd_el.getFirstElement("InitialConcentrationExpression").getText()
            if dd_el.findElement("ConcentrationFileName"):
                f.diff_data.init_filename = dd_el.getFirstElement("ConcentrationFileName").getText()

            if el.findElement("SecretionData"):
                sd_el: CC3DXMLElement = el.getFirstElement("SecretionData")
                p_el: CC3DXMLElement

                sd_el_list = sd_el.getElements("Secretion")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = sd_el.getElements("ConstantConcentration")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = sd_el.getElements("SecretionOnContact")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            b_el_list = b_el.getElements("Plane")
            for p_el in b_el_list:
                p_el: CC3DXMLElement
                axis: str = p_el.getAttribute("Axis").lower()
                if p_el.findElement("Periodic"):
                    setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                else:
                    c_el: CC3DXMLElement
                    p_el_list = p_el.getElements("ConstantValue")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[0])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                    p_el_list = p_el.getElements("ConstantDerivative")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[1])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


# KernelDiffusionSolver


class KernelDiffusionSolverDiffusionDataSpecs(_PDEDiffusionDataSpecs):
    """ KernelDiffusionSolver Diffusion Data Specs"""

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    init_expression: Union[str, None] = SpecProperty(name="init_expression")
    """expression of initial field distribution, Optional, None if not set"""

    init_filename: Union[str, None] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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


class KernelDiffusionSolverSecretionDataSpecs(_PDESecretionDataSpecs):
    """ KernelDiffusionSolver Secretion Data Specs """


class KernelDiffusionSolverBoundaryConditionsSpecs(PDEBoundaryConditionsSpec):
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


class KernelDiffusionSolverFieldSpecs(_PDESolverFieldSpecs[KernelDiffusionSolverDiffusionDataSpecs,
                                                           KernelDiffusionSolverSecretionDataSpecs]):
    """ KernelDiffusionSolver Field Specs """

    check_dict = {
        "kernel": (lambda x: x < 1, "Kernal must be positive"),
        "cgfactor": (lambda x: x < 1, "Coarse grain factor must be positive")
    }

    def __init__(self,
                 field_name: str,
                 diff_data=KernelDiffusionSolverDiffusionDataSpecs,
                 secr_data=KernelDiffusionSolverSecretionDataSpecs):
        super().__init__(field_name=field_name,
                         diff_data=diff_data,
                         secr_data=secr_data)

        self.spec_dict["bc_specs"] = KernelDiffusionSolverBoundaryConditionsSpecs()
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        if self.kernel > 1:
            self._el.ElementCC3D("Kernel", {}, str(self.kernel))
        if self.cgfactor > 1:
            self._el.ElementCC3D("CoarseGrainFactor", {}, str(self.cgfactor))
        self._el.add_child(self.spec_dict["diff_data"].xml)
        self._el.add_child(self.spec_dict["secr_data"].xml)
        return self._el

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> None:
        """
        Specify a new secretion data spec

        :param str _cell_type: name of cell type
        :param float _val: value
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        try:
            constant = kwargs["constant"]
        except KeyError:
            constant = False
        _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val, constant=constant, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param str _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class KernelDiffusionSolverSpecs(_PDESolverSpecs[KernelDiffusionSolverDiffusionDataSpecs,
                                                 KernelDiffusionSolverSecretionDataSpecs]):
    """ KernelDiffusionSolver """

    name = "kernel_diffusion_solver"
    registered_name = "KernelDiffusionSolver"

    _field_spec = KernelDiffusionSolverFieldSpecs
    _diff_data = KernelDiffusionSolverDiffusionDataSpecs
    _secr_data = KernelDiffusionSolverSecretionDataSpecs

    def __init__(self):
        super().__init__()

        delattr(self, "bcs")  # Fixed boundary conditions
        delattr(self, "steer")  # Not steerable

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(f.xml) for f in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: KernelDiffusionSolverSpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()

        el_list = el.getElements("DiffusionField")

        for f_el in el_list:
            f_el: CC3DXMLElement

            f = o.field_new(f_el.getAttribute("Name"))

            if f_el.findElement("Kernel"):
                f.kernel = f_el.getFirstElement("Kernel").getInt()
            if f_el.findElement("CoarseGrainFactor"):
                f.cgfactor = f_el.getFirstElement("CoarseGrainFactor").getInt()

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
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

                sd_el_list = sd_el.getElements("Secretion")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = sd_el.getElements("ConstantConcentration")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = sd_el.getElements("SecretionOnContact")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

        return o


# ReactionDiffusionSolverFE


class ReactionDiffusionSolverFEDiffusionDataSpecs(_PDEDiffusionDataSpecs):
    """ ReactionDiffusionSolverFE Diffusion Data Specs"""

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    additional_term: str = SpecProperty(name="additional_term")
    """expression of additional term"""

    init_filename: Union[str, None] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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
    def diff_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to diffusion coefficient values by cell type name
        :rtype: dict of [str: float]
        """
        return _PyCoreParamAccessor(self, "diff_types")

    @property
    def decay_types(self) -> _PyCoreParamAccessor[float]:
        """

        :return: accessor to decay coefficient values by cell type name
        :rtype: dict of [str: float]
        """
        return _PyCoreParamAccessor(self, "decay_types")


class ReactionDiffusionSolverFESecretionSpecs(_PDESecretionDataSpecs):
    """ ReactionDiffusionSolverFE Secretion Specs"""


class ReactionDiffusionSolverFEFieldSpecs(_PDESolverFieldSpecs[ReactionDiffusionSolverFEDiffusionDataSpecs,
                                                               ReactionDiffusionSolverFESecretionSpecs]):
    """ ReactionDiffusionSolverFE Field Specs"""

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> None:
        """
        Specify a new secretion data spec

        :param str _cell_type: name of cell type
        :param float _val: value
        :param kwargs:
        :raises SpecValueError: if setting constant concentration
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        if "constant" in kwargs.keys():
            raise SpecValueError("ReactionDiffusionSolverFE does not support constant concentrations")
        _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val, contact_type=contact_type)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param str _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        try:
            contact_type = kwargs["contact_type"]
        except KeyError:
            contact_type = None
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type, contact_type=contact_type)


class ReactionDiffusionSolverFESpecs(_PDESolverSpecs[ReactionDiffusionSolverFEDiffusionDataSpecs,
                                                     ReactionDiffusionSolverFESecretionSpecs]):
    """ ReactionDiffusionSolverFE """

    name = "reaction_diffusion_solver_fe"
    registered_name = "ReactionDiffusionSolverFE"

    _field_spec = ReactionDiffusionSolverFEFieldSpecs
    _diff_data = ReactionDiffusionSolverFEDiffusionDataSpecs
    _secr_data = ReactionDiffusionSolverFESecretionSpecs

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
        :rtype: ElementCC3D
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

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: ReactionDiffusionSolverFESpecs
        """
        el = cls.find_xml_by_attr(_xml)

        o = cls()
        o.autoscale = el.findElement("AutoscaleDiffusion")
        o.fluc_comp = el.findElement("FluctuationCompensator")

        el_list = el.getElements("DiffusionField")

        for f_el in el_list:
            f_el: CC3DXMLElement

            f = o.field_new(f_el.getAttribute("Name"))

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
            if dd_el.findElement("DiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("DiffusionConstant").getDouble()
            if dd_el.findElement("GlobalDiffusionConstant"):
                f.diff_data.diff_global = dd_el.getFirstElement("GlobalDiffusionConstant").getDouble()
            if dd_el.findElement("DecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("DecayConstant").getDouble()
            if dd_el.findElement("GlobalDecayConstant"):
                f.diff_data.decay_global = dd_el.getFirstElement("GlobalDecayConstant").getDouble()

            dd_el_list = dd_el.getElements("DiffusionCoefficient")

            for t_el in dd_el_list:
                f.diff_data.diff_types[t_el.getAttribute("CellType")] = t_el.getDouble()

            dd_el_list = dd_el.getElements("DecayCoefficient")

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

                sd_el_list = sd_el.getElements("Secretion")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

                sd_el_list = sd_el.getElements("ConstantConcentration")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble(), constant=True)

                sd_el_list = sd_el.getElements("SecretionOnContact")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"),
                                         p_el.getDouble(),
                                         contact_type=p_el.getAttribute("SecreteOnContactWith"))

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            b_el_list = b_el.getElements("Plane")
            for p_el in b_el_list:
                p_el: CC3DXMLElement
                axis: str = p_el.getAttribute("Axis").lower()
                if p_el.findElement("Periodic"):
                    setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                else:
                    c_el: CC3DXMLElement
                    p_el_list = p_el.getElements("ConstantValue")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[0])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                    p_el_list = p_el.getElements("ConstantDerivative")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[1])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


# SteadyStateDiffusionSolver2D + SteadyStateDiffusionSolver

class SteadyStateDiffusionSolverDiffusionDataSpecs(_PDEDiffusionDataSpecs):
    """ SteadyStateDiffusionSolver Diffusion Data Specs """

    diff_global: float = SpecProperty(name="diff_global")
    """global diffusion coefficient"""

    decay_global: float = SpecProperty(name="decay_global")
    """global decay rate"""

    init_expression: Union[str, None] = SpecProperty(name="init_expression")
    """expression of initial field distribution, Optional, None if not set"""

    init_filename: Union[str, None] = SpecProperty(name="init_filename")
    """name of file containing initial field distribution, Optional, None if not set"""

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
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


class SteadyStateDiffusionSolverSecretionDataSpecs(_PDESecretionDataSpecs):
    """ SteadyStateDiffusionSolver Secretion Data Specs """


class SteadyStateDiffusionSolverFieldSpecs(_PDESolverFieldSpecs[SteadyStateDiffusionSolverDiffusionDataSpecs,
                                                                SteadyStateDiffusionSolverSecretionDataSpecs]):
    """ SteadyStateDiffusionSolver Field Specs """

    def __init__(self,
                 field_name: str,
                 diff_data=SteadyStateDiffusionSolverDiffusionDataSpecs,
                 secr_data=SteadyStateDiffusionSolverSecretionDataSpecs):
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
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        self._el.add_child(self.spec_dict["diff_data"].xml)
        if self.pymanage:
            self._el.ElementCC3D("ManageSecretionInPython")
        else:
            self._el.add_child(self.spec_dict["secr_data"].xml)
        self._el.add_child(self.spec_dict["bc_specs"].xml)
        return self._el

    def secretion_data_new(self, _cell_type: str, _val: float, **kwargs) -> None:
        """
        Specify a new secretion data spec

        :param str _cell_type: name of cell type
        :param float _val: value
        :param kwargs:
        :raises SpecValueError: if setting constant concentration or on contact with
        :return: None
        """
        if "contact_type" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support contact-based secretion")
        if "constant" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support constant concentrations")
        _PDESolverFieldSpecs.secretion_data_new(self, _cell_type, _val)

    def secretion_data_remove(self, _cell_type: str, **kwargs) -> None:
        """
        Remove a secretion data spec

        :param str _cell_type: name of cell type
        :param kwargs:
        :return: None
        """
        if "contact_type" in kwargs.keys():
            raise SpecValueError("SteadyStateDiffusionSolver does not support contact-based secretion")
        _PDESolverFieldSpecs.secretion_data_remove(self, _cell_type)


class SteadyStateDiffusionSolverSpecs(_PDESolverSpecs[SteadyStateDiffusionSolverDiffusionDataSpecs,
                                                      SteadyStateDiffusionSolverSecretionDataSpecs]):
    """ SteadyStateDiffusionSolver Specs"""

    name = "steady_state_diffusion_solver"

    _field_spec = SteadyStateDiffusionSolverFieldSpecs
    _diff_data = SteadyStateDiffusionSolverDiffusionDataSpecs
    _secr_data = SteadyStateDiffusionSolverSecretionDataSpecs

    def __init__(self):
        super().__init__()

        self.spec_dict["three_d"] = False

    three_d: bool = SpecProperty(name="three_d")
    """flag whether domain is three-dimensional"""

    @property
    def registered_name(self) -> str:
        """

        :return: name according to core
        :rtype: str
        """
        if self.three_d:
            return "SteadyStateDiffusionSolver"
        return "SteadyStateDiffusionSolver2D"

    def generate_header(self) -> Union[ElementCC3D, None]:
        """
        Generate and return the top :class:`ElementCC3D` instance

        :return: top ElementCC3D instance
        :rtype: ElementCC3D
        """
        return ElementCC3D("Steppable", {"Type": self.registered_name})

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from specification dictionary

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        self._el = self.generate_header()
        [self._el.add_child(f.xml) for f in self.spec_dict["fields"].values()]
        return self._el

    @classmethod
    def from_xml(cls, _xml: CC3DXMLElement):
        """
        Instantiate an instance from a CC3DXMLElement parent instance

        :param CC3DXMLElement _xml: parent xml
        :return: python class instace
        :rtype: SteadyStateDiffusionSolverSpecs
        """
        o = cls()

        try:
            el = o.find_xml_by_attr(_xml)
        except SpecImportError:
            o.three_d = not o.three_d
            el = o.find_xml_by_attr(_xml)

        el_list = el.getElements("DiffusionField")

        for f_el in el_list:
            f_el: CC3DXMLElement

            f = o.field_new(f_el.getAttribute("Name"))
            f.pymanage = f_el.findElement("ManageSecretionInPython")

            dd_el: CC3DXMLElement = f_el.getFirstElement("DiffusionData")
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

                sd_el_list = sd_el.getElements("Secretion")

                for p_el in sd_el_list:
                    f.secretion_data_new(p_el.getAttribute("Type"), p_el.getDouble())

            b_el: CC3DXMLElement = f_el.getFirstElement("BoundaryConditions")
            b_el_list = b_el.getElements("Plane")
            for p_el in b_el_list:
                p_el: CC3DXMLElement
                axis: str = p_el.getAttribute("Axis").lower()
                if p_el.findElement("Periodic"):
                    setattr(f.bcs, f"{axis}_min_type", BOUNDARYTYPESPDE[2])
                else:
                    c_el: CC3DXMLElement
                    p_el_list = p_el.getElements("ConstantValue")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[0])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))
                    p_el_list = p_el.getElements("ConstantDerivative")
                    for c_el in p_el_list:
                        pos: str = c_el.getAttribute("PlanePosition").lower()
                        setattr(f.bcs, f"{axis}_{pos}_type", BOUNDARYTYPESPDE[1])
                        setattr(f.bcs, f"{axis}_{pos}_val", c_el.getAttributeAsDouble("Value"))

        return o


PLUGINS = [
    AdhesionFlexPluginSpecs,
    BoundaryPixelTrackerPluginSpecs,
    CellTypePluginSpecs,
    CenterOfMassPluginSpecs,
    ChemotaxisPluginSpecs,
    ConnectivityGlobalPluginSpecs,
    ConnectivityPluginSpecs,
    ContactPluginSpecs,
    CurvaturePluginSpecs,
    ExternalPotentialPluginSpecs,
    FocalPointPlasticityPluginSpecs,
    LengthConstraintPluginSpecs,
    MomentOfInertiaPluginSpecs,
    NeighborTrackerPluginSpecs,
    PixelTrackerPluginSpecs,
    SecretionPluginSpecs,
    SurfacePluginSpecs,
    VolumePluginSpecs
]
"""list of plugins that can be registered with cc3d"""

STEPPABLES = [
    BoxWatcherSteppableSpecs,
    PIFDumperSteppableSpecs
]
"""list of steppables that can be registered with cc3d"""

INITIALIZERS = [
    BlobInitializerSpecs,
    PIFInitializerSteppableSpecs,
    UniformInitializerSpecs
]
"""list of initializers that can be registered with cc3d"""

PDESOLVERS = [
    DiffusionSolverFESpecs,
    KernelDiffusionSolverSpecs,
    ReactionDiffusionSolverFESpecs,
    SteadyStateDiffusionSolverSpecs
]
"""list of PDE solvers that can be registered with cc3d"""
