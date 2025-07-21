
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
