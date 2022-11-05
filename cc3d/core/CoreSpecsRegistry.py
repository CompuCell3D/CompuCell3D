"""
Defines a registry for python-based core specification
"""

from typing import Dict, List

from cc3d import CompuCellSetup
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLElement
from cc3d.core.PyCoreSpecs import PyCoreSpecsRoot, _PyCoreSpecsBase


class CoreSpecsRegistry:
    """Core specs registry"""

    MASTERELEMENT = PyCoreSpecsRoot
    """Root element of registry"""

    def __init__(self):

        self.core_specs: Dict[str, _PyCoreSpecsBase] = {}
        """spec registry by name: instance"""

        self._injected = False
        """a flag to remember whether or not we've injected ourselves into the core"""

    def inject(self) -> None:
        """
        Inject self into PersistantGlobals instance

        :return: None
        """
        if self._injected:
            return
        CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter = XML2ObjConverterAdapter(self)
        self._injected = True

    def register_spec(self, _spec) -> None:
        """
        Register a core spec

        :param _spec: _PyCoreSpecsBase-derived class instance
        :raises TypeError: when attempting to register an instance of an improper class
        :raises ValueError: when attempting to register specs with the name of a previously registered spec
        :return: None
        """
        if not issubclass(type(_spec), _PyCoreSpecsBase):
            raise TypeError
        if _spec.name in self.core_specs.keys():
            raise ValueError
        self.core_specs[_spec.name] = _spec

    @property
    def xml(self) -> ElementCC3D:
        """
        CC3DXML element; generated from constituent spec instances

        :return: CC3DML XML element
        :rtype: ElementCC3D
        """
        if "potts" not in self.core_specs.keys():
            raise AttributeError("No Potts specification")

        el = self.MASTERELEMENT().xml

        for spec in self.core_specs.values():
            spec_el = spec.xml
            el.add_child(spec_el)

        return el


class XML2ObjConverterAdapter:
    """Adapter for PersistentGlobals"""
    def __init__(self, _registry: CoreSpecsRegistry):
        self._registry = _registry
        """core specs registry"""

        self._xml_tree = None
        """simulation xml spec"""

    @property
    def xmlTree(self) -> ElementCC3D:
        """

        :return: simulation xml spec
        :rtype: ElementCC3D
        """
        if self._xml_tree is None:
            self._xml_tree = self._registry.xml
        return self._xml_tree

    @property
    def root(self) -> CC3DXMLElement:
        """

        :return: simulation xml spec
        :rtype: CC3DXMLElement
        """
        return self.xmlTree.CC3DXMLElement


class CoreSpecsAccessor(object):
    """
    For convenient retrieval of core specs in steppables
    All registered core specs are accessible by name as an attribute
    """

    @property
    def registry(self) -> CoreSpecsRegistry:
        """core specs registry"""
        return CompuCellSetup.persistent_globals.core_specs_registry

    @property
    def spec_names(self) -> List[str]:
        """

        :return: list of spec names
        :rtype: list of str
        """
        reg = self.registry
        return [x.name for x in reg.core_specs.values()]

    def __getattr__(self, name):
        if name == "_registry":
            return super().__getattribute__(self, name)

        reg = self.registry
        if not reg._injected:
            return None
        for x in reg.core_specs.values():
            if x.name == name:
                return x
        return None

    def __setattr__(self, name, value):
        if name == "_registry":
            super().__setattr__(name, value)
            return

        if name in self.spec_names:
            raise AttributeError(f"Setting attribute with name {name} is illegal.")
        super().__setattr__(self, name, value)
