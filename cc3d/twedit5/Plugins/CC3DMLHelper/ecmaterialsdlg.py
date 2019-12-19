from copy import deepcopy
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_ecmaterialsdlg

MAC = "qt_mac_set_native_menubar" in dir()


class ECMaterialsDlg(QDialog, ui_ecmaterialsdlg.Ui_ECMaterialsDlg):
    def __init__(self, cell_types: [], previous_info: {}):
        super(ECMaterialsDlg, self).__init__()
        self.setModal(True)

        self.setupUi(self)
        if not MAC:
            self.buttonBox.setFocusPolicy(Qt.NoFocus)

        # To prevent accept on Return key; also nice to have a custom key filter implemented for future dev
        self.key_event_detector = KeyEventDetector(parent=self)
        self.installEventFilter(self.key_event_detector)
        self.was_a_key_press = False

        self.user_res = False

        self.cell_types = cell_types

        self.keys_set_here = self.get_keys()

        self.cb_emitters = None

        if previous_info:
            self.ec_materials_dict = None
            self.load_previous_info(previous_info=previous_info)
        else:
            self.ec_materials_dict = self.initialize_new_info()

        self.connect_all_signals()

        self.update_ui()

    @staticmethod
    def get_keys():
        return ["ECMaterials", "Adhesion", "Remodeling", "Advects", "Durability"]

    @staticmethod
    def initialize_new_info() -> {}:
        return {"ECMaterials": [],
                "Adhesion": {},
                "Remodeling": {},
                "Advects": {},
                "Durability": {}}

    def load_previous_info(self, previous_info: dict):
        self.ec_materials_dict = previous_info

        # Perform checks in case cell types changed
        adhesion_coefficients = self.ec_materials_dict["Adhesion"]
        remodeling_quantities = self.ec_materials_dict["Remodeling"]
        first_ec_material_set = False
        for ec_material in self.ec_materials_dict["ECMaterials"]:
            for cell_type in self.cell_types - adhesion_coefficients[ec_material].keys():
                adhesion_coefficients[ec_material][cell_type] = 0

            adhesion_coefficients[ec_material] = {key: val for key, val in adhesion_coefficients[ec_material].items()
                                                  if key in self.cell_types}

            self.ec_materials_dict["Adhesion"][ec_material].clear()
            self.ec_materials_dict["Adhesion"][ec_material] = adhesion_coefficients[ec_material]

            for cell_type in self.cell_types - remodeling_quantities[ec_material].keys():
                if not first_ec_material_set:
                    remodeling_quantities[ec_material][cell_type] = 1
                else:
                    remodeling_quantities[ec_material][cell_type] = 0

            remodeling_quantities[ec_material] = {key: val for key, val in remodeling_quantities[ec_material].items()
                                                  if key in self.cell_types}

            self.ec_materials_dict["Remodeling"][ec_material].clear()
            self.ec_materials_dict["Remodeling"][ec_material] = remodeling_quantities[ec_material]

            first_ec_material_set = True

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

            cb = QCheckBox()
            cb.setCheckable(True)
            cb.setChecked(self.ec_materials_dict["Advects"][ec_material])
            self.tableWidget_materialDefs.setCellWidget(row_count, 1, cb)
            self.cb_emitters.append(QCBCallbackEmitter(parent=self,
                                                       cb=cb,
                                                       cb_row=row_count,
                                                       cb_col=1))

            twi = QTableWidgetItem(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            twi.setText(str(self.ec_materials_dict["Durability"][ec_material]))
            self.tableWidget_materialDefs.setItem(row_count, 2, twi)

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
                else:
                    self.ec_materials_dict[key][new_name] = self.ec_materials_dict[key][ec_material]
                    self.ec_materials_dict[key].pop(ec_material)
        elif col == 1:  # Advection toggle
            cb: QCheckBox = self.tableWidget_materialDefs.cellWidget(row, col)
            self.ec_materials_dict["Advects"][ec_material] = cb.isChecked()
        elif col == 2:  # Durability coefficient change
            item: QTableWidget = self.tableWidget_materialDefs.item(row, col)
            try:
                self.ec_materials_dict["Durability"][ec_material] = float(item.text())
            except TypeError:
                self.disconnect_all_signals()
                item.setText(str(self.ec_materials_dict["Durability"][ec_material]))
                self.connect_all_signals()

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
        except TypeError:
            self.disconnect_all_signals()
            item.setText(str(self.ec_materials_dict["Adhesion"][ec_material][cell_type]))
            self.connect_all_signals()

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
        except TypeError:
            self.disconnect_all_signals()
            item.setText(str(self.ec_materials_dict["Remodeling"][ec_material][cell_type]))
            self.connect_all_signals()

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

        # Set empty dictionaries for keys not set here, in case info came back from steppable
        for key in self.ec_materials_dict.keys():
            if key not in self.keys_set_here:
                self.ec_materials_dict[key][new_material] = {}

        self.disconnect_all_signals()
        self.lineEdit.clear()
        self.checkBox.setChecked(True)
        self.connect_all_signals()

        self.update_ui()

    def handle_delete_material(self, ec_material: str):
        for key in self.ec_materials_dict.keys():
            if key == "ECMaterials":
                self.ec_materials_dict[key].remove(ec_material)
            else:
                self.ec_materials_dict[key].pop(ec_material)

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

        self.update_ui()

    def on_accept(self):
        if self.was_a_key_press:
            return
        self.user_res = True
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
