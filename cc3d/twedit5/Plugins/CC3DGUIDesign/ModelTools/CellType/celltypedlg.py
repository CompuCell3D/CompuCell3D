from copy import deepcopy
from itertools import product
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolGUIBase import CC3DModelToolGUIBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CellType.ui_celltypedlg import Ui_CellTypePluginGUI


class CellTypeGUI(CC3DModelToolGUIBase, Ui_CellTypePluginGUI):
    def __init__(self, parent=None, cell_types=None, is_frozen=None):
        super(CellTypeGUI, self).__init__(parent)
        self.setupUi(self)

        self.cell_types = deepcopy(cell_types)
        self.is_frozen = deepcopy(is_frozen)

        self.user_decision = None

        self.selected_row = None

        self.init_data()

        self.connect_all_signals()

        self.draw_ui()

        self.showNormal()

    def init_data(self):
        if self.cell_types is None or not self.cell_types:
            self.cell_types = ["Medium"]

        if self.is_frozen is None or not self.is_frozen:
            self.is_frozen = [False]

    def draw_ui(self):
        if self.cellTypeTable.currentRow() > 0:
            self.selected_row = self.cellTypeTable.currentRow()
        else:
            self.selected_row = None

        self.cellTypeTable.itemChanged.disconnect(self.on_table_item_change)

        self.cellTypeTable.setRowCount(0)
        for i in range(self.cell_types.__len__()):
            row = self.cellTypeTable.rowCount()
            self.cellTypeTable.insertRow(row)
            tti = TypeTableItem(text=self.cell_types[i])
            self.cellTypeTable.setItem(row, 0, tti)

            fcb = FreezeCB(parent=self, check_state=self.is_frozen[i], is_medium=row == 0)
            self.cellTypeTable.setCellWidget(row, 1, fcb)

        if self.selected_row is not None and self.selected_row > self.cellTypeTable.rowCount():
            self.selected_row = None

        if self.selected_row is not None:
            self.cellTypeTable.setCurrentCell(self.selected_row, 0)

        self.cellTypeTable.setVerticalHeaderLabels([str(row) for row in range(self.cellTypeTable.rowCount())])
        self.cellTypeTable.resizeColumnsToContents()
        self.cellTypeTable.resizeRowsToContents()
        self.cellTypeTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cellTypeTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)

        self.cellTypeTable.itemChanged.connect(self.on_table_item_change)

    def connect_all_signals(self):
        self.cellTypeTable.itemChanged.connect(self.on_table_item_change)
        self.cellTypeAddPB.clicked.connect(self.on_add_cell_type)
        self.deleteCellTypePB.clicked.connect(self.on_del_cell_type)
        self.clearCellTypeTablePB.clicked.connect(self.on_clear_table)
        self.okPB.clicked.connect(self.on_accept)
        self.cancelPB.clicked.connect(self.on_reject)

    def on_table_item_change(self, item: QTableWidgetItem):
        if item.row() == 0 and item.column() == 0:
            item.setText("Medium")
            return
        elif item.row() < 0:
            return
        if item.column() == 0 and item.text() != "Medium" and item.text().__len__() > 2:
            self.name_change(old_name=self.cell_types[item.row()], new_name=item.text())

    def on_add_cell_type(self):
        cell_name = self.cellTypeLE.text()
        is_freeze = self.freezeCHB.isChecked()

        if not self.validate_name(name=cell_name):
            return

        self.cell_types.append(cell_name)
        self.is_frozen.append(is_freeze)

        self.draw_ui()

    def on_del_cell_type(self):
        row = self.cellTypeTable.currentRow()
        col = self.cellTypeTable.currentColumn()
        if row < 0 or col < 0:
            return

        cell_name = self.cellTypeTable.item(row, 0).text()
        if cell_name == "Medium":
            return
        else:
            self.cell_types.pop(row)
            self.is_frozen.pop(row)

            self.draw_ui()

    def on_clear_table(self):
        self.cell_types = []
        self.is_frozen = []
        self.init_data()
        self.draw_ui()

    def on_accept(self):
        self.user_decision = True
        self.close()

    def on_reject(self):
        self.user_decision = False
        self.close()

    def name_change(self, old_name: str, new_name: str):
        if self.validate_name(name=new_name):
            for i in range(self.cell_types.__len__()):
                if self.cell_types[i] == old_name:
                    self.cell_types[i] = new_name
                    return

    def validate_name(self, name: str):
        return not (name in self.cell_types or name == "Medium" or name.__len__() < 2)


class TypeTableItem(QTableWidgetItem):
    def __init__(self, text: str):
        super(TypeTableItem, self).__init__()
        self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self.setText(text)


class FreezeCB(QWidget):
    def __init__(self, parent: CellTypeGUI, check_state: bool = False, is_medium: bool = False):
        super(FreezeCB, self).__init__(parent)

        self.cb = QCheckBox()
        self.cb.setCheckable(not is_medium)
        self.cb.setChecked(check_state and not is_medium)

        self.h_layout = QHBoxLayout(self)
        self.h_layout.addWidget(self.cb)
        self.h_layout.setAlignment(Qt.AlignCenter)
