# Cell Type related only


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

