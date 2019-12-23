from copy import deepcopy
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.core import XMLUtils
from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_ecmaterialssteppable
from . import ecmaterialsdlg

MAC = "qt_mac_set_native_menubar" in dir()


class ECMaterialsSteppableDlg(QDialog, ui_ecmaterialssteppable.Ui_ECMaterialsSteppableDlg):
    def __init__(self, field_names: [], previous_info: {}):
        super(ECMaterialsSteppableDlg, self).__init__()
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

        self.cell_types = None

        self.field_list = field_names

        self.keys_set_here = self.get_keys()

        self.cb_emitters = None

        self.valid_diffusion = {}
        self.invalid_font_color = QColor("red")
        self.invalid_palette = QPalette()
        self.invalid_palette.setColor(QPalette.Text, self.invalid_font_color)
        self.lineEdit_mtl_palette = self.lineEdit_mtl.palette()
        self.lineEdit_fld_palette = self.lineEdit_fld.palette()
        self.lineEdit_cell_palette = self.lineEdit_cell.palette()

        self.ec_materials_dict = None
        self.load_previous_info(previous_info=deepcopy(previous_info))

        # Containers for interaction design
        self.mtl_interaction = None
        self.fld_interaction = None
        self.cell_interaction = None
        self.init_design_containers()
        self.cell_response_types = self.get_cell_response_types()
        self.responses_new_types = self.get_response_new_types()

        # Static connections
        self.lineEdit_mtl.textChanged.connect(self.on_line_edit_mtl_changed)
        self.lineEdit_fld.textChanged.connect(self.on_line_edit_fld_changed)
        self.lineEdit_cell.textChanged.connect(self.on_line_edit_cell_changed)
        self.comboBox_fldC.currentTextChanged.connect(self.on_fld_catalyst_select)
        self.comboBox_cellT.currentTextChanged.connect(self.update_combo_box_types_new)
        self.comboBox_cellR.currentTextChanged.connect(self.update_combo_box_types_new)
        self.tabWidget.currentChanged.connect(self.resize_tables)

        # Connect dynamic connections
        self.connect_all_signals()

        self.update_ui()

        self.tabWidget.setCurrentIndex(0)

    @staticmethod
    def get_keys():
        return ["MaterialInteractions", "FieldInteractions", "CellInteractions", "MaterialDiffusion"]

    @staticmethod
    def get_cell_response_types() -> []:
        return ['Proliferation', 'Death', 'Differentiation', 'Asymmetric Division']

    @staticmethod
    def get_response_new_types() -> []:
        return ['Differentiation', 'Asymmetric Division']

    def resize_tables(self) -> None:
        coefficient_width = 100

        table_dict = {self.tableWidget_mtl: 2, self.tableWidget_fld: 2, self.tableWidget_cell: 3}
        for table, coefficient_col in table_dict.items():
            hh = table.horizontalHeader()
            table_width = table.width()
            var_width = int((table_width - coefficient_width - 18) / (table.columnCount() - 1))
            for col in range(table.columnCount()):
                if col != coefficient_col:
                    table.setColumnWidth(col, var_width)
                    hh.setSectionResizeMode(col, QHeaderView.Fixed)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.resize_tables()

        QDialog.resizeEvent(self, event)

    def init_design_containers(self) -> None:
        self.reset_mtl_interaction()
        self.reset_fld_interaction()
        self.reset_cell_interaction()

    def reset_mtl_interaction(self) -> None:
        self.mtl_interaction = {"Catalyst": '',
                                "Reactant": '',
                                "Coefficient": 0}

    def reset_fld_interaction(self) -> None:
        self.fld_interaction = {"Catalyst": '',
                                "Reactant": '',
                                "Coefficient": 0}

    def reset_cell_interaction(self) -> None:
        self.cell_interaction = {"ECMaterial": '',
                                 "CellType": '',
                                 "ResponseType": '',
                                 "Coefficient": 0,
                                 "CellTypeNew": ''}

    def load_previous_info(self, previous_info: dict) -> None:
        self.ec_materials_dict = previous_info

        # Get cell types from adhesion coefficients defined in plugin
        if self.ec_materials_dict["ECMaterials"]:
            self.cell_types = list(self.ec_materials_dict["Adhesion"][self.ec_materials_dict["ECMaterials"][0]].keys())
        else:
            self.cell_types = []

        # Perform checks on imported data
        for key, val in ecmaterialsdlg.get_default_data().items():
            if key not in self.ec_materials_dict.keys():
                self.ec_materials_dict[key] = val

        # Perform checks in case materials changed
        self.ec_materials_dict["CellInteractions"] = \
            [val for val in self.ec_materials_dict["CellInteractions"]
             if val["ECMaterial"] in self.ec_materials_dict["ECMaterials"]]
        self.ec_materials_dict["MaterialInteractions"] = \
            [val for val in self.ec_materials_dict["MaterialInteractions"]
             if val["Catalyst"] in self.ec_materials_dict["ECMaterials"]
             and val["Reactant"] in self.ec_materials_dict["ECMaterials"]]
        self.ec_materials_dict["FieldInteractions"] = \
            [val for val in self.ec_materials_dict["FieldInteractions"]
             if len({val["Catalyst"], val["Reactant"]} & set(self.ec_materials_dict["ECMaterials"])) == 1]
        self.ec_materials_dict["MaterialDiffusion"] = \
            {key: val for key, val in self.ec_materials_dict["MaterialDiffusion"].items()
             if key in self.ec_materials_dict["ECMaterials"]}
        for ec_material in self.ec_materials_dict["ECMaterials"] - self.ec_materials_dict["MaterialDiffusion"].keys():
            self.ec_materials_dict["MaterialDiffusion"][ec_material] = {"Diffuses": False,
                                                                        "Coefficient": 0}

        # Perform checks in case cell types changed
        self.ec_materials_dict["CellInteractions"] = \
            [val for val in self.ec_materials_dict["CellInteractions"] if val["CellType"] in self.cell_types]
        self.ec_materials_dict["CellInteractions"] = \
            [val for val in self.ec_materials_dict["CellInteractions"]
             if val["CellTypeNew"] in self.cell_types or val["CellTypeNew"].__len__() == 0]

        # Perform checks in case fields changed
        self.ec_materials_dict["FieldInteractions"] = \
            [val for val in self.ec_materials_dict["FieldInteractions"]
             if len({val["Catalyst"], val["Reactant"]} & set(self.field_list)) == 1]

        # Initialize booleans for tracking valid diffusion table entries
        self.valid_diffusion = {key: True for key in self.ec_materials_dict["ECMaterials"]}

    def get_user_res(self):
        return self.user_res

    def extract_information(self) -> {}:

        return self.ec_materials_dict

    def connect_all_signals(self) -> None:
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

        # Tab: Material Interactions
        self.pushButton_mtlAdd.clicked.connect(self.on_mtl_add)
        self.pushButton__mtlDel.clicked.connect(self.on_mtl_del)
        self.pushButton__mtlClear.clicked.connect(self.on_mtl_clear)

        # Tab: Field Interactions
        self.pushButton_fldAdd.clicked.connect(self.on_fld_add)
        self.pushButton_fldDel.clicked.connect(self.on_fld_del)
        self.pushButton_fldClear.clicked.connect(self.on_fld_clear)

        # Tab: Cell Interactions
        self.pushButton_cellAdd.clicked.connect(self.on_cell_add)
        self.pushButton_cellDel.clicked.connect(self.on_cell_del)
        self.pushButton_cellClear.clicked.connect(self.on_cell_clear)

        # Tab: Material Diffusion
        self.tableWidget_diff.itemChanged.connect(self.on_diffusion_table_item_edit)
        self.tableWidget_diff.cellChanged.connect(self.on_diffusion_table_edit)

    def disconnect_all_signals(self) -> None:
        self.buttonBox.accepted.disconnect(self.on_accept)
        self.buttonBox.rejected.disconnect(self.on_reject)

        # Tab: Material Interactions
        self.pushButton_mtlAdd.clicked.disconnect(self.on_mtl_add)
        self.pushButton__mtlDel.clicked.disconnect(self.on_mtl_del)
        self.pushButton__mtlClear.clicked.disconnect(self.on_mtl_clear)

        # Tab: Field Interactions
        self.pushButton_fldAdd.clicked.disconnect(self.on_fld_add)
        self.pushButton_fldDel.clicked.disconnect(self.on_fld_del)
        self.pushButton_fldClear.clicked.disconnect(self.on_fld_clear)

        # Tab: Cell Interactions
        self.pushButton_cellAdd.clicked.disconnect(self.on_cell_add)
        self.pushButton_cellDel.clicked.disconnect(self.on_cell_del)
        self.pushButton_cellClear.clicked.disconnect(self.on_cell_clear)

        # Tab: Material Diffusion
        self.tableWidget_diff.itemChanged.disconnect(self.on_diffusion_table_item_edit)
        self.tableWidget_diff.cellChanged.disconnect(self.on_diffusion_table_edit)

    @staticmethod
    def template_table_item(text=None) -> QTableWidgetItem:
        twi = QTableWidgetItem()
        if text is None:
            text = ''
        twi.setText(str(text))
        twi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        return twi

    def update_combo_box_field_reactants(self) -> None:
        current_field = self.comboBox_fldR.currentText()
        if self.comboBox_fldC.currentText() in self.ec_materials_dict["ECMaterials"]:
            new_list = self.field_list
        elif self.comboBox_fldC.currentText() in self.field_list:
            new_list = self.ec_materials_dict["ECMaterials"]
        else:
            new_list = [self.ec_materials_dict["ECMaterials"], self.field_list]

        self.comboBox_fldR.clear()
        self.comboBox_fldR.addItems(new_list)
        if current_field in new_list:
            self.comboBox_fldR.setCurrentText(current_field)

    def update_combo_box_types_new(self) -> None:
        if self.comboBox_cellR.currentText() not in self.responses_new_types:
            self.comboBox_cellT_New.clear()
            self.comboBox_cellT_New.setEnabled(False)
        else:
            current_type = self.comboBox_cellT_New.currentText()
            new_list = [cell_type for cell_type in self.cell_types if cell_type != self.comboBox_cellT.currentText()]
            self.comboBox_cellT_New.clear()
            self.comboBox_cellT_New.addItems(new_list)
            if current_type in new_list:
                self.comboBox_cellT_New.setCurrentText(current_type)

            self.comboBox_cellT_New.setEnabled(True)

    def update_ui(self) -> None:

        self.disconnect_all_signals()

        # Tab: Material Interactions
        self.comboBox_mtlC.clear()
        self.comboBox_mtlC.addItems(self.ec_materials_dict["ECMaterials"])
        self.comboBox_mtlC.setEditable(False)
        if self.mtl_interaction["Catalyst"].__len__() > 0:
            self.comboBox_mtlC.setCurrentText(self.mtl_interaction["Catalyst"])
        self.comboBox_mtlR.clear()
        self.comboBox_mtlR.addItems(self.ec_materials_dict["ECMaterials"])
        self.comboBox_mtlR.setEditable(False)
        if self.mtl_interaction["Reactant"].__len__() > 0:
            self.comboBox_mtlR.setCurrentText(self.mtl_interaction["Reactant"])
        self.lineEdit_mtl.setText(str(self.mtl_interaction["Coefficient"]))

        self.tableWidget_mtl.setRowCount(0)
        for interaction in self.ec_materials_dict["MaterialInteractions"]:
            row = self.tableWidget_mtl.rowCount()
            self.tableWidget_mtl.insertRow(row)
            self.tableWidget_mtl.setItem(row, 0, self.template_table_item(text=interaction["Catalyst"]))
            self.tableWidget_mtl.setItem(row, 1, self.template_table_item(text=interaction["Reactant"]))
            self.tableWidget_mtl.setItem(row, 2, self.template_table_item(text=interaction["Coefficient"]))

        if self.tableWidget_mtl.rowCount() == 0:
            self.tableWidget_mtl.setEnabled(False)
            self.pushButton__mtlDel.setEnabled(False)
            self.pushButton__mtlClear.setEnabled(False)
        else:
            self.tableWidget_mtl.setEnabled(True)
            self.pushButton__mtlDel.setEnabled(True)
            self.pushButton__mtlClear.setEnabled(True)

        # Tab: Field Interactions
        # mtl_fld_list = list(set(self.ec_materials_dict["ECMaterials"]).union(set(self.field_list)))
        mtl_fld_list = self.ec_materials_dict["ECMaterials"] + self.field_list
        self.comboBox_fldC.clear()
        self.comboBox_fldC.addItems(mtl_fld_list)
        if self.fld_interaction["Catalyst"].__len__() > 0:
            self.comboBox_fldC.setCurrentText(self.fld_interaction["Catalyst"])
        self.on_fld_catalyst_select(self.comboBox_fldC.currentText())
        self.lineEdit_fld.setText(str(self.fld_interaction["Coefficient"]))

        self.tableWidget_fld.setRowCount(0)
        for interaction in self.ec_materials_dict["FieldInteractions"]:
            row = self.tableWidget_fld.rowCount()
            self.tableWidget_fld.insertRow(row)
            self.tableWidget_fld.setItem(row, 0, self.template_table_item(text=interaction["Catalyst"]))
            self.tableWidget_fld.setItem(row, 1, self.template_table_item(text=interaction["Reactant"]))
            self.tableWidget_fld.setItem(row, 2, self.template_table_item(text=interaction["Coefficient"]))

        if self.tableWidget_fld.rowCount() == 0:
            self.tableWidget_fld.setEnabled(False)
            self.pushButton_fldDel.setEnabled(False)
            self.pushButton_fldClear.setEnabled(False)
        else:
            self.tableWidget_fld.setEnabled(True)
            self.pushButton_fldDel.setEnabled(True)
            self.pushButton_fldClear.setEnabled(True)

        # Tab: Cell Interactions
        self.comboBox_cellM.clear()
        self.comboBox_cellM.addItems(self.ec_materials_dict["ECMaterials"])
        self.comboBox_cellM.setEditable(False)
        if self.cell_interaction["ECMaterial"].__len__() > 0:
            self.comboBox_cellM.setCurrentText(self.cell_interaction["ECMaterial"])
        self.comboBox_cellT.clear()
        self.comboBox_cellT.addItems(self.cell_types)
        self.comboBox_cellT.setEditable(False)
        if self.cell_interaction["CellType"].__len__() > 0:
            self.comboBox_cellT.setCurrentText(self.cell_interaction["CellType"])
        self.comboBox_cellR.clear()
        self.comboBox_cellR.addItems(self.cell_response_types)
        self.comboBox_cellR.setEditable(False)
        if self.cell_interaction["ResponseType"].__len__() > 0:
            self.comboBox_cellR.setCurrentText(self.cell_interaction["ResponseType"])
        self.lineEdit_cell.setText(str(self.cell_interaction["Coefficient"]))
        self.update_combo_box_types_new()

        self.tableWidget_cell.setRowCount(0)
        for interaction in self.ec_materials_dict["CellInteractions"]:
            row = self.tableWidget_cell.rowCount()
            self.tableWidget_cell.insertRow(row)
            self.tableWidget_cell.setItem(row, 0, self.template_table_item(text=interaction["ECMaterial"]))
            self.tableWidget_cell.setItem(row, 1, self.template_table_item(text=interaction["CellType"]))
            self.tableWidget_cell.setItem(row, 2, self.template_table_item(text=interaction["ResponseType"]))
            self.tableWidget_cell.setItem(row, 3, self.template_table_item(text=interaction["Coefficient"]))
            self.tableWidget_cell.setItem(row, 4, self.template_table_item(text=interaction["CellTypeNew"]))

        if self.tableWidget_cell.rowCount() == 0:
            self.tableWidget_cell.setEnabled(False)
            self.pushButton_cellDel.setEnabled(False)
            self.pushButton_cellClear.setEnabled(False)
        else:
            self.tableWidget_cell.setEnabled(True)
            self.pushButton_cellDel.setEnabled(True)
            self.pushButton_cellClear.setEnabled(True)

        # Tab: Material Diffusion
        self.tableWidget_diff.setRowCount(0)
        self.cb_emitters = []
        for key, diffusion in self.ec_materials_dict["MaterialDiffusion"].items():
            row = self.tableWidget_diff.rowCount()
            self.tableWidget_diff.insertRow(row)
            self.tableWidget_diff.setItem(row, 0, self.template_table_item(text=key))
            ccb = CustomCheckBox(parent=self, check_state=diffusion["Diffuses"])
            self.tableWidget_diff.setCellWidget(row, 1, ccb)
            self.cb_emitters.append(QCBCallbackEmitter(parent=self,
                                                       cb=ccb.cb,
                                                       cb_row=row,
                                                       cb_col=1))
            twi = self.template_table_item(text=diffusion["Coefficient"])
            twi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            if not self.valid_diffusion[key]:
                twi.setData(Qt.TextColorRole, self.invalid_font_color)
            self.tableWidget_diff.setItem(row, 2, twi)

        self.tableWidget_diff.resizeRowsToContents()

        self.resize_tables()

        self.connect_all_signals()

    # Callbacks
    def on_mtl_add(self) -> None:
        # Assemble candidate entry
        try:
            val = float(self.lineEdit_mtl.text())
        except ValueError:
            return
        self.mtl_interaction = {"Catalyst": self.comboBox_mtlC.currentText(),
                                "Reactant": self.comboBox_mtlR.currentText(),
                                "Coefficient": val}

        # Check for completion
        for val in self.mtl_interaction.values():
            if val == '':
                return

        # Check for duplicates
        for interaction in self.ec_materials_dict["MaterialInteractions"]:
            is_different = False
            for key in interaction.keys() - "Coefficient":
                if interaction[key] != self.mtl_interaction[key]:
                    is_different = True

            if not is_different:
                return

        self.ec_materials_dict["MaterialInteractions"].append(self.mtl_interaction)
        self.reset_mtl_interaction()
        self.update_ui()

    def on_mtl_del(self) -> None:
        row = self.tableWidget_mtl.currentRow()
        if row < 0:
            return
        self.ec_materials_dict["MaterialInteractions"].pop(row)
        self.update_ui()

    def on_mtl_clear(self) -> None:
        self.ec_materials_dict["MaterialInteractions"].clear()
        self.update_ui()

    def on_line_edit_mtl_changed(self, text: str):
        try:
            float(text)
            self.lineEdit_mtl.setPalette(self.lineEdit_mtl_palette)
        except ValueError:
            self.lineEdit_mtl.setPalette(self.invalid_palette)

    def on_line_edit_fld_changed(self, text: str):
        try:
            float(text)
            self.lineEdit_fld.setPalette(self.lineEdit_fld_palette)
        except ValueError:
            self.lineEdit_fld.setPalette(self.invalid_palette)

    def on_line_edit_cell_changed(self, text: str):
        try:
            float(text)
            self.lineEdit_cell.setPalette(self.lineEdit_cell_palette)
        except ValueError:
            self.lineEdit_cell.setPalette(self.invalid_palette)

    def on_fld_catalyst_select(self, text) -> None:
        current_fldR = None
        if text in self.ec_materials_dict["ECMaterials"]:
            if self.comboBox_fldR.currentText() in self.field_list:
                current_fldR = self.comboBox_fldR.currentText()
            new_list = self.field_list
        elif text in self.field_list:
            if self.comboBox_fldR.currentText() in self.ec_materials_dict["ECMaterials"]:
                current_fldR = self.comboBox_fldR.currentText()
            new_list = self.ec_materials_dict["ECMaterials"]
        else:
            new_list = self.ec_materials_dict["ECMaterials"] + self.field_list

        self.comboBox_fldR.clear()
        self.comboBox_fldR.addItems(new_list)
        if current_fldR is not None:
            self.comboBox_fldR.setCurrentText(current_fldR)

    def on_fld_add(self) -> None:
        # Assemble candidate entry
        try:
            val = float(self.lineEdit_fld.text())
        except ValueError:
            return
        self.fld_interaction = {"Catalyst": self.comboBox_fldC.currentText(),
                                "Reactant": self.comboBox_fldR.currentText(),
                                "Coefficient": val}

        # Check for completion
        for val in self.fld_interaction.values():
            if val == '':
                return

        # Check for duplicates
        for interaction in self.ec_materials_dict["FieldInteractions"]:
            is_different = False
            for key in interaction.keys() - "Coefficient":
                if interaction[key] != self.fld_interaction[key]:
                    is_different = True

            if not is_different:
                return

        self.ec_materials_dict["FieldInteractions"].append(self.fld_interaction)
        self.reset_fld_interaction()
        self.update_ui()

    def on_fld_del(self) -> None:
        row = self.tableWidget_fld.currentRow()
        if row < 0:
            return
        self.ec_materials_dict["FieldInteractions"].pop(row)
        self.update_ui()

    def on_fld_clear(self) -> None:
        self.ec_materials_dict["FieldInteractions"].clear()
        self.update_ui()

    def on_cell_add(self) -> None:
        # Assemble candidate entry
        try:
            val = float(self.lineEdit_cell.text())
        except ValueError:
            return
        self.cell_interaction = {"ECMaterial": self.comboBox_cellM.currentText(),
                                 "CellType": self.comboBox_cellT.currentText(),
                                 "ResponseType": self.comboBox_cellR.currentText(),
                                 "Coefficient": val,
                                 "CellTypeNew": self.comboBox_cellT_New.currentText()}

        # Check for completion
        for key in self.cell_interaction.keys():
            if self.cell_interaction[key] == '':
                if key == "CellTypeNew":
                    if self.cell_interaction["ResponseType"] in self.responses_new_types:
                        return
                else:
                    return

        # Check for duplicates
        for interaction in self.ec_materials_dict["CellInteractions"]:
            is_different = False
            for key in interaction.keys() - "Coefficient":
                if interaction[key] != self.cell_interaction[key]:
                    is_different = True

            if not is_different:
                return

        self.ec_materials_dict["CellInteractions"].append(self.cell_interaction)
        self.reset_cell_interaction()
        self.update_ui()

    def on_cell_del(self) -> None:
        row = self.tableWidget_cell.currentRow()
        if row < 0:
            return
        self.ec_materials_dict["CellInteractions"].pop(row)
        self.update_ui()

    def on_cell_clear(self) -> None:
        self.ec_materials_dict["CellInteractions"].clear()
        self.update_ui()

    def on_diffusion_table_item_edit(self, item) -> None:
        self.on_diffusion_table_edit(item.row(), item.column())

    def on_diffusion_table_edit(self, row, col) -> None:
        if row < 0 or col <= 0:
            return
        ec_material = self.tableWidget_diff.item(row, 0).text()
        ccb: CustomCheckBox = self.tableWidget_diff.cellWidget(row, 1)
        self.ec_materials_dict["MaterialDiffusion"][ec_material]["Diffuses"] = ccb.is_checked()
        twi = self.tableWidget_diff.item(row, 2)
        try:
            val = float(twi.text())
            self.ec_materials_dict["MaterialDiffusion"][ec_material]["Coefficient"] = val
            self.valid_diffusion[ec_material] = True
        except ValueError:
            self.ec_materials_dict["MaterialDiffusion"][ec_material]["Coefficient"] = twi.text()
            self.valid_diffusion[ec_material] = False
        self.update_ui()

    def on_accept(self):
        if self.was_a_key_press:
            return
        self.user_res = True

        for key in self.valid_diffusion.keys():
            if not self.valid_diffusion[key]:
                self.ec_materials_dict["MaterialDiffusion"][key] = {"Diffuses": False,
                                                                    "Coefficient": 0}
        self.close()

    def on_reject(self):
        if self.was_a_key_press:
            return
        self.user_res = False
        self.close()

    def set_key_press_flag(self, key_press_flag):
        self.was_a_key_press = key_press_flag


class KeyEventDetector(QObject):
    def __init__(self, parent: ECMaterialsSteppableDlg):
        super(KeyEventDetector, self).__init__(parent)
        self.main_UI = parent

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:
        self.main_UI.set_key_press_flag(key_press_flag=a1.type() == a1.KeyPress)

        return super(KeyEventDetector, self).eventFilter(a0, a1)


class QCBCallbackEmitter(QObject):
    def __init__(self, parent: ECMaterialsSteppableDlg, cb: QCheckBox, cb_row: int, cb_col: int):
        super(QCBCallbackEmitter, self).__init__(parent)
        self.main_UI = parent
        self.cb = cb
        self.cb_row = cb_row
        self.cb_col = cb_col

        self.cb.stateChanged.connect(self.emit)

    def emit(self, state: int):
        self.main_UI.on_diffusion_table_edit(row=self.cb_row, col=self.cb_col)


class CustomCheckBox(QWidget):
    def __init__(self, parent: ECMaterialsSteppableDlg, check_state: bool = True):
        super(CustomCheckBox, self).__init__(parent)

        self.cb = QCheckBox()
        self.cb.setCheckable(True)
        self.cb.setChecked(check_state)

        self.h_layout = QHBoxLayout(self)
        self.h_layout.addWidget(self.cb)
        self.h_layout.setAlignment(Qt.AlignCenter)

    def is_checked(self) -> bool:
        return self.cb.isChecked()


# Parsing here; package somewhere else later
def ec_materials_steppable_xml_to_data(xml_data=None, plugin_data=None):
    if xml_data is None:
        return ec_materials_steppable_xml_demo()

    ec_materials_data = deepcopy(plugin_data)

    # Import raw data (taken from C++)
    material_interaction_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("MaterialInteractions"))
    field_interaction_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("FieldInteractions"))
    material_diffusion_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("MaterialDiffusion"))
    cell_interaction_xml_list = XMLUtils.CC3DXMLListPy(xml_data.getElements("CellInteraction"))

    # Get all specified cell types from adhesion
    cell_types = list(ec_materials_data["Adhesion"][ec_materials_data["ECMaterials"][0]].keys())
    cell_types.sort()

    # Import all specified field names
    catalyst_names = [element.getAttribute('Catalyst') for element in field_interaction_xml_list]
    reactant_names = [element.getAttribute('Reactant') for element in field_interaction_xml_list]
    field_names = catalyst_names + reactant_names
    field_names = [field_name for field_name in field_names if field_name not in ec_materials_data["ECMaterials"]]
    field_names = list(set(field_names))
    field_names.sort()

    # Import material interactions
    ec_materials_data["MaterialInteractions"] = []
    for element in material_interaction_xml_list:
        ec_material = element.getAttribute('ECMaterial')
        catalyst_name = element.getFirstElement('Catalyst').getText()
        val = element.getFirstElement('ConstantCoefficient').getDouble()

        if ec_material not in ec_materials_data["ECMaterials"]:
            print('Undefined ECMaterial: ' + ec_material + '... rejecting')
            continue
        elif catalyst_name not in ec_materials_data["ECMaterials"]:
            print('Undefined catalyst: ' + catalyst_name + '... rejecting')
            continue

        try:
            ec_materials_data["MaterialInteractions"].append({"Catalyst": catalyst_name,
                                                              "Reactant": ec_material,
                                                              "Coefficient": val})
        except KeyError:
            print('Could not import XML element attribute for material interaction:')
            print('   ECMaterial: ' + ec_material)
            print('   Catalyst: ' + catalyst_name)
            print('   ConstantCoefficient: ' + str(val))

    # Import material diffusion
    for element in material_diffusion_xml_list:
        ec_material = element.getAttribute('ECMaterial')
        val = element.getDouble()

        if ec_material not in ec_materials_data["ECMaterials"]:
            print('Undefined ECMaterial: ' + ec_material + '... rejecting')
            continue

        try:
            ec_materials_data["MaterialDiffusion"][ec_material]["Diffuses"] = True
            ec_materials_data["MaterialDiffusion"][ec_material]["Coefficient"] = val
        except KeyError:
            print('Could not import XML element attribute for material diffusion:')
            print('   ECMaterial: ' + ec_material)
            print('   ConstantCoefficient: ' + str(val))

    # Import field interactions
    ec_materials_data["FieldInteractions"] = []
    for element in field_interaction_xml_list:
        reactant_name = element.getFirstElement('Reactant').getText()
        catalyst_name = element.getFirstElement('Catalyst').getText()
        val = element.getFirstElement('ConstantCoefficient').getDouble()

        if reactant_name in field_names and catalyst_name in field_names:
            print('Cannot define field-field interactions here: ' +
                  reactant_name + ', ' + catalyst_name + '... rejecting')
            continue
        elif not (catalyst_name in ec_materials_data["ECMaterials"] and reactant_name in field_names) and not \
                (catalyst_name in field_names and reactant_name in ec_materials_data["ECMaterials"]):
            print('Cannot imports as a ECMaterial-field interaction: ' +
                  catalyst_name + ', ' + reactant_name + '... rejecting')
            continue

        try:
            ec_materials_data["FieldInteractions"].append({"Catalyst": catalyst_name,
                                                           "Reactant": reactant_name,
                                                           "Coefficient": val})
        except KeyError:
            print('Could not import XML element attribute for field interaction:')
            print('   Reactant: ' + reactant_name)
            print('   Catalyst: ' + catalyst_name)
            print('   ConstantCoefficient: ' + str(val))

    # Import cell interactions
    ec_materials_data["CellInteractions"] = []
    for element in cell_interaction_xml_list:
        ec_material = element.getFirstElement('ECMaterial').getText()
        method_name = element.getAttribute('Method')
        val = element.getFirstElement('Probability').getDouble()
        cell_type = element.getFirstElement('CellType').getText()
        cell_type_new = ''
        if element.findElement('CellTypeNew'):
            cell_type_new = element.getFirstElement('CellTypeNew').getText()

        if ec_material not in ec_materials_data["ECMaterials"]:
            print('Undefined ECMaterial: ' + ec_material + '... rejecting')
            continue
        elif method_name not in ECMaterialsSteppableDlg.get_cell_response_types():
            print('Undefined method: ' + method_name + '... rejecting')
            continue
        elif cell_type not in cell_types:
            print('Undefined cell type: ' + cell_type + '... rejecting')
            continue
        elif cell_type_new.__len__() > 0 and cell_type_new not in cell_types:
            print('Undefined new cell type: ' + cell_type_new + '... rejecting')
            continue
        elif cell_type_new.__len__() > 0 and method_name not in ECMaterialsSteppableDlg.get_response_new_types():
            print('Undefined new cell type for requested method: ' +
                  cell_type_new + ', ' + method_name + '... rejecting new cell type')
            cell_type_new = ''

        try:
            ec_materials_data["CellInteractions"].append({"ECMaterial": ec_material,
                                                          "CellType": cell_type,
                                                          "ResponseType": method_name,
                                                          "Coefficient": val,
                                                          "CellTypeNew": cell_type_new})
        except KeyError:
            print('Could not import XML element attribute for material interaction:')
            print('   Method: ' + method_name)
            print('   ECMaterial: ' + ec_material)
            print('   Probability: ' + str(val))
            print('   CellType: ' + cell_type)
            print('   ConstantCoefficient: ' + str(val))
            if cell_type_new.__len__() > 0:
                print('   CellTypeNew: ' + cell_type_new)

    return ec_materials_data


def ec_materials_steppable_xml_demo():
    ec_materials_data = ecmaterialsdlg.ec_materials_xml_to_data()
    ec_materials_ex = ec_materials_data["ECMaterials"]
    cell_types_ex = list(ec_materials_data["Adhesion"][ec_materials_ex[0]].keys())
    field_names_ex = ['Field1', 'Field2']

    # Demo ECMaterial diffusion
    val = 0.1
    for ec_material in ec_materials_ex:
        ec_materials_data["MaterialDiffusion"][ec_material] = {"Diffuses": True,
                                                               "Coefficient": val}
        val += 0.1

    # Demo material interactions
    val = 0.1
    for reactant in ec_materials_ex:
        for catalyst in ec_materials_ex:
            ec_materials_data["MaterialInteractions"].append({"Catalyst": catalyst,
                                                              "Reactant": reactant,
                                                              "Coefficient": val})
            val *= -2

    # Demo field interactions
    reactant_list_ex = [ec_materials_ex[0], field_names_ex[1]]
    catalyst_list_ex = [field_names_ex[0], ec_materials_ex[1]]
    val_list = [0.1, -0.2]
    for int_index in range(catalyst_list_ex.__len__()):
        ec_materials_data["FieldInteractions"].append({"Catalyst": catalyst_list_ex[int_index],
                                                       "Reactant": reactant_list_ex[int_index],
                                                       "Coefficient": val_list[int_index]})

    # Demo cell interactions
    cell_response_types = ECMaterialsSteppableDlg.get_cell_response_types()
    alt_index = [1, 0]
    cell_types_index = 0
    val = 0.001
    for method_index in range(cell_response_types.__len__()):
        method_name = cell_response_types[method_index]
        if cell_types_index >= cell_types_ex.__len__():
            cell_types_index = 0

        if method_name in ECMaterialsSteppableDlg.get_response_new_types():
            cell_type_new = cell_types_ex[alt_index[cell_types_index]]
        else:
            cell_type_new = ''

        ec_materials_data["CellInteractions"].append({"ECMaterial": ec_materials_ex[alt_index[cell_types_index]],
                                                      "CellType": cell_types_ex[cell_types_index],
                                                      "ResponseType": method_name,
                                                      "Coefficient": val,
                                                      "CellTypeNew": cell_type_new})

        cell_types_index += 1
        val *= 2

    return ec_materials_data
