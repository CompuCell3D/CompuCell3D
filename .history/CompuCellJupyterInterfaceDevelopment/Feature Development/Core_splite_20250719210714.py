
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
