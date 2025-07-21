
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