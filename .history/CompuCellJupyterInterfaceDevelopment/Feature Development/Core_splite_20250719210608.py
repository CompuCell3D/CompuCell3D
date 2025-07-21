
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

