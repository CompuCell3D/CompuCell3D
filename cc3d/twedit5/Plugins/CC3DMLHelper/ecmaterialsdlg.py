from copy import deepcopy
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.core import XMLUtils
from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_ecmaterialsdlg
from .ecmaterialssteppabledlg import ECMaterialsSteppableDlg

MAC = "qt_mac_set_native_menubar" in dir()


class ECMaterialsDlg(QDialog, ui_ecmaterialsdlg.Ui_ECMaterialsDlg):
    def __init__(self, cell_types: [], previous_info: {}):
        super(ECMaterialsDlg, self).__init__()
        self.setModal(True)

        self.setupUi(self)
        if not MAC:
            self.buttonBox.setFocusPolicy(Qt.NoFocus)

        self.setWindowFlags(Qt.Window)

        # To prevent accept on Return key; also nice to have a custom key filter implemented for future dev
        self.key_event_detector = KeyEventDetector(parent=self)
        self.installEventFilter(self.key_event_detector)
        self.was_a_key_press = False

        self.user_res = False

        self.cell_types = cell_types

        self.keys_set_here = self.get_keys()

        self.cb_emitters = None

        self.valid_adhesion = {}
        self.valid_remodeling = {}
        self.valid_durability = {}
        self.invalid_font_color = QColor("red")

        if previous_info:
            self.ec_materials_dict = None
            self.load_previous_info(previous_info=deepcopy(previous_info))
        else:
            self.ec_materials_dict = get_default_data()

        self.connect_all_signals()

        self.update_ui()

        self.resize_table_material_defs()

    @staticmethod
    def get_keys():
        return ["ECMaterials", "Adhesion", "Remodeling", "Advects", "Durability"]

    def resize_table_material_defs(self) -> None:
        hh = self.tableWidget_materialDefs.horizontalHeader()
        table_width = self.tableWidget_materialDefs.width()

        cb_width = 50
        self.tableWidget_materialDefs.setColumnWidth(1, cb_width)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)

        durability_width = 100
        self.tableWidget_materialDefs.setColumnWidth(2, durability_width)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)

        name_width = max(80, table_width - cb_width - durability_width - 18)
        self.tableWidget_materialDefs.setColumnWidth(0, name_width)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.resize_table_material_defs()

        QDialog.resizeEvent(self, event)

    def load_previous_info(self, previous_info: dict):
        self.ec_materials_dict = previous_info

        # Perform checks on imported data
        for key, val in get_default_data().items():
            if key not in self.ec_materials_dict.keys():
                self.ec_materials_dict[key] = val

        # Perform checks in case cell types changed
        adhesion_coefficients = self.ec_materials_dict["Adhesion"]
        remodeling_quantities = self.ec_materials_dict["Remodeling"]
        first_ec_material_set = False
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            for cell_type in self.cell_types - adhesion_coefficients[ec_material].keys():
                adhesion_coefficients[ec_material][cell_type] = 0

            adhesion_coefficients[ec_material] = {key: val for key, val in adhesion_coefficients[ec_material].items()
                                                  if key in self.cell_types}

            self.ec_materials_dict["Adhesion"][ec_material] = adhesion_coefficients[ec_material]

            for cell_type in self.cell_types - remodeling_quantities[ec_material].keys():
                if not first_ec_material_set:
                    remodeling_quantities[ec_material][cell_type] = 1
                else:
                    remodeling_quantities[ec_material][cell_type] = 0

            remodeling_quantities[ec_material] = {key: val for key, val in remodeling_quantities[ec_material].items()
                                                  if key in self.cell_types}

            self.ec_materials_dict["Remodeling"][ec_material] = remodeling_quantities[ec_material]

            first_ec_material_set = True

        # Initialize booleans for tracking valid entries
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            self.valid_adhesion[ec_material] = {key: True for key in self.cell_types}
            self.valid_remodeling[ec_material] = {key: True for key in self.cell_types}
            self.valid_durability[ec_material] = True

    def get_user_res(self):
        return self.user_res

    def extract_information(self):

        return self.ec_materials_dict

    def connect_all_signals(self):
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)
        self.tableWidget_materialDefs.itemChanged.connect(self.handle_material_defs_table_item_edit)
        self.tableWidget_materialDefs.cellChanged.connect(self.handle_material_defs_table_edit)
        self.tableWidget_adhesion.itemChanged.connect(self.handle_adhesion_table_edit)
        self.tableWidget_remodeling.itemChanged.connect(self.handle_remodeling_table_edit)
        self.pushButton_add.clicked.connect(self.handle_add_material)
        self.pushButton_delete.clicked.connect(self.handle_delete_material_button)
        self.pushButton_clear.clicked.connect(self.handle_clear_materials_button)

    def disconnect_all_signals(self):
        self.buttonBox.accepted.disconnect(self.on_accept)
        self.buttonBox.rejected.disconnect(self.on_reject)
        self.tableWidget_materialDefs.itemChanged.disconnect(self.handle_material_defs_table_item_edit)
        self.tableWidget_materialDefs.cellChanged.disconnect(self.handle_material_defs_table_edit)
        self.tableWidget_adhesion.itemChanged.disconnect(self.handle_adhesion_table_edit)
        self.tableWidget_remodeling.itemChanged.disconnect(self.handle_remodeling_table_edit)
        self.pushButton_add.clicked.disconnect(self.handle_add_material)
        self.pushButton_delete.clicked.disconnect(self.handle_delete_material_button)
        self.pushButton_clear.clicked.disconnect(self.handle_clear_materials_button)

    def update_ui(self):

        self.disconnect_all_signals()

        # Refresh LHS table
        self.tableWidget_materialDefs.setRowCount(0)
        self.cb_emitters = []
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            row_count = self.tableWidget_materialDefs.rowCount()
            self.tableWidget_materialDefs.insertRow(row_count)

            twi = QTableWidgetItem(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            twi.setText(ec_material)
            self.tableWidget_materialDefs.setItem(row_count, 0, twi)

            ccb = CustomCheckBox(parent=self, check_state=self.ec_materials_dict["Advects"][ec_material])
            self.tableWidget_materialDefs.setCellWidget(row_count, 1, ccb)
            self.cb_emitters.append(QCBCallbackEmitter(parent=self,
                                                       cb=ccb.cb,
                                                       cb_row=row_count,
                                                       cb_col=1))

            twi = QTableWidgetItem(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            twi.setText(str(self.ec_materials_dict["Durability"][ec_material]))
            if not self.valid_durability[ec_material]:
                twi.setData(Qt.TextColorRole, self.invalid_font_color)
            self.tableWidget_materialDefs.setItem(row_count, 2, twi)

        self.tableWidget_materialDefs.resizeRowsToContents()

        # Refresh RHS Adhesion table
        self.tableWidget_adhesion.setRowCount(0)
        self.tableWidget_adhesion.setColumnCount(len(self.cell_types))
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            row_count = self.tableWidget_adhesion.rowCount()
            self.tableWidget_adhesion.insertRow(row_count)
            for type_index in range(len(self.cell_types)):
                cell_type = self.cell_types[type_index]

                twi = QTableWidgetItem(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                twi.setText(str(self.ec_materials_dict["Adhesion"][ec_material][cell_type]))
                if not self.valid_adhesion[ec_material][cell_type]:
                    twi.setData(Qt.TextColorRole, self.invalid_font_color)
                self.tableWidget_adhesion.setItem(row_count, type_index, twi)

        self.tableWidget_adhesion.setVerticalHeaderLabels(self.ec_materials_dict["ECMaterials"])
        self.tableWidget_adhesion.setHorizontalHeaderLabels(self.cell_types)

        # Refresh RHS Remodeling table
        self.tableWidget_remodeling.setRowCount(0)
        self.tableWidget_remodeling.setColumnCount(len(self.cell_types))
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            row_count = self.tableWidget_remodeling.rowCount()
            self.tableWidget_remodeling.insertRow(row_count)
            for type_index in range(len(self.cell_types)):
                cell_type = self.cell_types[type_index]

                twi = QTableWidgetItem(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                twi.setText(str(self.ec_materials_dict["Remodeling"][ec_material][cell_type]))
                if not self.valid_remodeling[ec_material][cell_type]:
                    twi.setData(Qt.TextColorRole, self.invalid_font_color)
                self.tableWidget_remodeling.setItem(row_count, type_index, twi)

        self.tableWidget_remodeling.setVerticalHeaderLabels(self.ec_materials_dict["ECMaterials"])
        self.tableWidget_remodeling.setHorizontalHeaderLabels(self.cell_types)

        self.connect_all_signals()

        if self.tableWidget_materialDefs.rowCount() == 0:
            self.tableWidget_materialDefs.setEnabled(False)
            self.tableWidget_adhesion.setEnabled(False)
            self.tableWidget_remodeling.setEnabled(False)
        else:
            self.tableWidget_materialDefs.setEnabled(True)
            self.tableWidget_adhesion.setEnabled(True)
            self.tableWidget_remodeling.setEnabled(True)

    def handle_material_defs_table_item_edit(self, item):
        self.handle_material_defs_table_edit(row=item.row(), col=item.column())

    def handle_material_defs_table_edit(self, row, col):
        if col < 0:
            return

        ec_material = self.ec_materials_dict["ECMaterials"][row]
        if col == 0:  # Name change
            item: QTableWidget = self.tableWidget_materialDefs.item(row, col)
            new_name = item.text()
            if len(new_name) < 2 or new_name == ec_material:
                self.disconnect_all_signals()
                item.setText(ec_material)
                self.connect_all_signals()
                return
            for key in self.ec_materials_dict:
                if key == "ECMaterials":
                    self.ec_materials_dict[key][row] = new_name
                elif isinstance(self.ec_materials_dict[key], dict):
                    for key_key, val in self.ec_materials_dict[key].items():
                        if key_key == ec_material:
                            self.ec_materials_dict[key][new_name] = self.ec_materials_dict[key].pop(key_key)
                        elif val == ec_material:
                            self.ec_materials_dict[key][key_key] = new_name
                elif isinstance(self.ec_materials_dict[key], list):
                    for i in range(self.ec_materials_dict[key].__len__()):
                        for key_key in self.ec_materials_dict[key][i].keys():
                            if self.ec_materials_dict[key][i][key_key] == ec_material:
                                self.ec_materials_dict[key][i][key_key] = new_name
        elif col == 1:  # Advection toggle
            ccb: CustomCheckBox = self.tableWidget_materialDefs.cellWidget(row, col)
            self.ec_materials_dict["Advects"][ec_material] = ccb.is_checked()
        elif col == 2:  # Durability coefficient change
            item: QTableWidget = self.tableWidget_materialDefs.item(row, col)
            try:
                val = float(item.text())
                self.ec_materials_dict["Durability"][ec_material] = val
                self.valid_durability[ec_material] = True
            except ValueError:
                self.ec_materials_dict["Durability"][ec_material] = item.text()
                self.valid_durability[ec_material] = False

        self.update_ui()

    def handle_adhesion_table_edit(self, item: QTableWidgetItem):
        col = item.column()
        if col < 0:
            return
        row = item.row()
        ec_material = self.ec_materials_dict["ECMaterials"][row]
        cell_type = self.cell_types[col]
        try:
            new_value = float(item.text())
            self.ec_materials_dict["Adhesion"][ec_material][cell_type] = new_value
            self.valid_adhesion[ec_material][cell_type] = True
        except ValueError:
            self.ec_materials_dict["Adhesion"][ec_material][cell_type] = item.text()
            self.valid_adhesion[ec_material][cell_type] = False

        self.update_ui()

    def handle_remodeling_table_edit(self, item: QTableWidgetItem):
        col = item.column()
        if col < 0:
            return
        row = item.row()
        ec_material = self.ec_materials_dict["ECMaterials"][row]
        cell_type = self.cell_types[col]
        try:
            new_value = float(item.text())
            self.ec_materials_dict["Remodeling"][ec_material][cell_type] = new_value
            self.valid_remodeling[ec_material][cell_type] = True
        except ValueError:
            self.ec_materials_dict["Remodeling"][ec_material][cell_type] = item.text()
            self.valid_remodeling[ec_material][cell_type] = False

        self.update_ui()

    def handle_add_material(self):
        new_material = self.lineEdit.text()
        if len(new_material) < 2:
            return
        self.ec_materials_dict["ECMaterials"].append(new_material)
        self.ec_materials_dict["Advects"][new_material] = self.checkBox.isChecked()
        self.ec_materials_dict["Durability"][new_material] = 0.0
        self.ec_materials_dict["Adhesion"][new_material] = {cell_type: 0.0 for cell_type in self.cell_types}
        self.ec_materials_dict["Remodeling"][new_material] = {cell_type: 0.0 for cell_type in self.cell_types}

        self.disconnect_all_signals()
        self.lineEdit.clear()
        self.checkBox.setChecked(True)
        self.connect_all_signals()

        self.valid_adhesion[new_material] = {key: True for key in self.cell_types}
        self.valid_remodeling[new_material] = {key: True for key in self.cell_types}
        self.valid_durability[new_material] = True

        self.update_ui()

    def handle_delete_material(self, ec_material: str):
        for key in self.ec_materials_dict.keys():
            if key == "ECMaterials":
                self.ec_materials_dict[key].remove(ec_material)
            elif isinstance(self.ec_materials_dict[key], dict) and ec_material in self.ec_materials_dict[key].keys():
                self.ec_materials_dict[key].pop(ec_material)

        self.valid_adhesion.pop(ec_material)
        self.valid_remodeling.pop(ec_material)
        self.valid_durability.pop(ec_material)

    def handle_delete_material_button(self):
        row = self.tableWidget_materialDefs.currentRow()
        if row < 0:
            return
        ec_material = self.tableWidget_materialDefs.item(row, 0).text()
        self.handle_delete_material(ec_material=ec_material)
        self.update_ui()

    def handle_clear_materials_button(self):
        ec_material_list = deepcopy(self.ec_materials_dict["ECMaterials"])
        for ec_material in ec_material_list:
            self.handle_delete_material(ec_material=ec_material)

        self.valid_adhesion = {}
        self.valid_remodeling = {}
        self.valid_durability = {}

        self.update_ui()

    def on_accept(self):
        if self.was_a_key_press:
            return
        self.user_res = True

        for ec_material in self.ec_materials_dict["Adhesion"].keys():
            for cell_type in self.cell_types:
                if not self.valid_adhesion[ec_material][cell_type]:
                    self.ec_materials_dict["Adhesion"][ec_material][cell_type] = 0.0

        for ec_material in self.ec_materials_dict["Remodeling"].keys():
            for cell_type in self.cell_types:
                if not self.valid_remodeling[ec_material][cell_type]:
                    self.ec_materials_dict["Remodeling"][ec_material][cell_type] = 0.0

        for ec_material in self.ec_materials_dict["Durability"].keys():
            if not self.valid_durability[ec_material]:
                self.ec_materials_dict["Durability"][ec_material] = 0.0

        self.close()

    def on_reject(self):
        if self.was_a_key_press:
            return
        self.user_res = False
        self.close()

    def set_key_press_flag(self, key_press_flag):
        self.was_a_key_press = key_press_flag


class KeyEventDetector(QObject):
    def __init__(self, parent: ECMaterialsDlg):
        super(KeyEventDetector, self).__init__(parent)
        self.main_UI = parent

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:
        self.main_UI.set_key_press_flag(key_press_flag=a1.type() == a1.KeyPress)

        return super(KeyEventDetector, self).eventFilter(a0, a1)


class QCBCallbackEmitter(QObject):
    def __init__(self, parent: ECMaterialsDlg, cb: QCheckBox, cb_row: int, cb_col: int):
        super(QCBCallbackEmitter, self).__init__(parent)
        self.main_UI = parent
        self.cb = cb
        self.cb_row = cb_row
        self.cb_col = cb_col

        self.cb.stateChanged.connect(self.emit)

    def emit(self, state: int):
        self.main_UI.handle_material_defs_table_edit(row=self.cb_row, col=self.cb_col)


class CustomCheckBox(QWidget):
    def __init__(self, parent: ECMaterialsDlg, check_state: bool = True):
        super(CustomCheckBox, self).__init__(parent)

        self.cb = QCheckBox()
        self.cb.setCheckable(True)
        self.cb.setChecked(check_state)

        self.h_layout = QHBoxLayout(self)
        self.h_layout.addWidget(self.cb)
        self.h_layout.setAlignment(Qt.AlignCenter)

    def is_checked(self) -> bool:
        return self.cb.isChecked()


def get_default_data():
    return {"ECMaterials": [],
            "Adhesion": {},
            "Remodeling": {},
            "Advects": {},
            "Durability": {},
            "MaterialInteractions": [],
            "FieldInteractions": [],
            "CellInteractions": [],
            "MaterialDiffusion": {}}


# Parsing here; package somewhere else later
def ec_materials_xml_to_data(xml_data=None) -> {}:
    if xml_data is None:
        return ec_materials_plugin_xml_demo()

    ec_materials_data = get_default_data()

    # Import raw data (taken from C++)
    name_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("ECMaterial"))
    adhesion_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("ECAdhesion"))
    advection_bool_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("ECMaterialAdvects"))
    durability_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("ECMaterialDurability"))
    remodeling_quantity_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("RemodelingQuantity"))

    # Import ECMaterials
    for element in name_xml_list:
        ec_material = element.getAttribute('Material')
        ec_materials_data["ECMaterials"].append(ec_material)

    # Import all specified cell types in adhesion and remodeling
    cell_types = []
    [cell_types.append(element.getAttribute('CellType')) for element in adhesion_xml_list]
    [cell_types.append(element.getAttribute('CellType')) for element in remodeling_quantity_xml_list]
    cell_types = list(set(cell_types))
    cell_types.sort()

    # Generate default data to fill out from XML for plugin completion
    for ec_material in ec_materials_data["ECMaterials"]:
        ec_materials_data["Adhesion"][ec_material] = {}
        ec_materials_data["Remodeling"][ec_material] = {}
        ec_materials_data["Advects"][ec_material] = True
        ec_materials_data["Durability"][ec_material] = 0.0
        ec_materials_data["MaterialDiffusion"][ec_material] = {"Diffuses": False,
                                                               "Coefficient": 0}
        for cell_type in cell_types:
            ec_materials_data["Adhesion"][ec_material][cell_type] = 0.0
            ec_materials_data["Remodeling"][ec_material][cell_type] = 0.0

    # Import adhesion
    for element in adhesion_xml_list:
        ec_material = element.getAttribute('Material')
        cell_type = element.getAttribute('CellType')
        val = element.getDouble()
        try:
            ec_materials_data["Adhesion"][ec_material][cell_type] = val
        except KeyError:
            print('Could not import XML element attribute for adhesion:')
            print('   Material: ' + ec_material)
            print('   CellType: ' + cell_type)
            print('   Value: ' + str(val))

    # Import remodeling
    for element in remodeling_quantity_xml_list:
        ec_material = element.getAttribute('Material')
        cell_type = element.getAttribute('CellType')
        val = element.getDouble()
        try:
            ec_materials_data["Remodeling"][ec_material][cell_type] = val
        except KeyError:
            print('Could not import XML element attribute for remodeling quantity:')
            print('   Material: ' + ec_material)
            print('   CellType: ' + cell_type)
            print('   Value: ' + str(val))

    # Import advection
    for element in advection_bool_xml_list:
        ec_material = element.getAttribute('Material')
        val = element.getBool()
        try:
            ec_materials_data["Advects"][ec_material] = val
        except KeyError:
            print('Could not import XML element attribute for advection:')
            print('   Material: ' + ec_material)
            print('   Value: ' + str(val))

    # Import durability
    for element in durability_xml_list:
        ec_material = element.getAttribute('Material')
        val = element.getDouble()
        try:
            ec_materials_data["Durability"][ec_material] = val
        except KeyError:
            print('Could not import XML element attribute for durability:')
            print('   Material: ' + ec_material)
            print('   Value: ' + str(val))

    return ec_materials_data


def ec_materials_plugin_xml_demo() -> {}:
    ec_materials_data = get_default_data()
    # Generate example data
    ec_materials_ex = ['ECMaterial1', 'ECMaterial2']
    cell_types_ex = ['CellType1', 'CellType2']

    # Demo declaring ECMaterials
    ec_materials_data["ECMaterials"] = ec_materials_ex

    # Demo adhesion
    adhesion_coefficient_ex = 1.0
    for ec_material in ec_materials_ex:
        adhesion_dict = {}
        for cell_type in cell_types_ex:
            adhesion_dict[cell_type] = adhesion_coefficient_ex
            adhesion_coefficient_ex += 1

        ec_materials_data["Adhesion"][ec_material] = adhesion_dict

    # Demo remodeling
    remodeling_coefficients_ex = [0.25, 0.75]
    for ec_material in ec_materials_ex:
        ec_materials_data["Remodeling"][ec_material] = \
            {cell_types_ex[type_index]: remodeling_coefficients_ex[type_index]
             for type_index in range(cell_types_ex.__len__())}
        remodeling_coefficients_ex.reverse()

    # Demo advection
    advects_bool = False
    for ec_material in ec_materials_ex:
        ec_materials_data["Advects"][ec_material] = advects_bool
        advects_bool = not advects_bool

    # Demo durability
    durability_coefficient_ex = 1.0
    ec_materials_data["Durability"] = {ec_materials_ex[idx]: durability_coefficient_ex*(idx + 1)
                                       for idx in range(ec_materials_ex.__len__())}

    return ec_materials_data
