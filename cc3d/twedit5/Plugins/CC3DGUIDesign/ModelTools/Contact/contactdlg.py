from copy import deepcopy
from itertools import product
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolGUIBase import CC3DModelToolGUIBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.Contact.ui_contactdlg import Ui_ContactPluginGUI


class ContactGUI(CC3DModelToolGUIBase, Ui_ContactPluginGUI):
    def __init__(self, parent=None, contact_matrix=None, neighbor_order=None):
        super(ContactGUI, self).__init__(parent)
        self.setupUi(self)

        self.contact_matrix = deepcopy(contact_matrix)
        self.neighbor_order = deepcopy(neighbor_order)
        self.neighbor_order_list = range(1, 5)
        self.cell_types = None

        self.user_decision = False

        self.valid_color = QColor("black")
        self.invalid_color = QColor("red")

        self.init_data()

        self.draw_ui()

        self.connect_all_signals()

    def init_data(self):
        if not self.contact_matrix:
            self.cell_types = []
        else:
            self.cell_types = list(self.contact_matrix.keys())

    def draw_ui(self):
        self.spinBox.setMinimum(min(self.neighbor_order_list))
        self.spinBox.setMaximum(max(self.neighbor_order_list))
        self.spinBox.setValue(self.neighbor_order)

        self.tableWidget.clear()
        self.tableWidget.setRowCount(self.cell_types.__len__())
        self.tableWidget.setColumnCount(self.cell_types.__len__())
        if self.tableWidget.rowCount() == 0:
            return
        self.tableWidget.setHorizontalHeaderLabels(self.cell_types)
        self.tableWidget.setVerticalHeaderLabels(self.cell_types)
        for id1, id2 in product(range(0, self.cell_types.__len__()), range(0, self.cell_types.__len__())):
            twi = QTableWidgetItem()
            twi.setText(str(self.contact_matrix[self.cell_types[id1]][self.cell_types[id2]]))
            twi.setData(Qt.TextColorRole, self.valid_color)
            self.tableWidget.setItem(id1, id2, twi)
            if id2 < id1:
                self.tableWidget.item(id1, id2).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            else:
                self.tableWidget.item(id1, id2).setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)

        self.resize_coefficient_table()
        self.tableWidget.show()

    def connect_all_signals(self):
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

        self.tableWidget.itemChanged.connect(self.on_table_edit)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.resize_coefficient_table()

        QWidget.resizeEvent(self, event)

    def resize_coefficient_table(self) -> None:
        if not self.cell_types:
            return

        width = self.tableWidget.width() - self.tableWidget.verticalHeader().width() - 2
        num_cols = self.tableWidget.columnCount()
        col_width = int(width/num_cols)
        [self.tableWidget.setColumnWidth(col, col_width) for col in range(num_cols)]

    def on_table_edit(self, item: QTableWidgetItem):
        try:
            self.tableWidget.itemChanged.disconnect(self.on_table_edit)
            is_connected = True
        except TypeError:
            is_connected = False

        row = item.row()
        col = item.column()
        if row < 0 or col < 0:
            return

        try:
            float(item.text())
            item.setData(Qt.TextColorRole, self.valid_color)
        except ValueError:
            item.setData(Qt.TextColorRole, self.invalid_color)

        self.tableWidget.item(col, row).setText(item.text())

        if is_connected:
            self.tableWidget.itemChanged.connect(self.on_table_edit)

    def on_accept(self) -> None:
        self.user_decision = True
        for id1, id2 in product(range(self.cell_types.__len__()), range(self.cell_types.__len__())):
            t1 = self.cell_types[id1]
            t2 = self.cell_types[id2]
            try:
                self.contact_matrix[t1][t2] = float(self.tableWidget.item(id1, id2).text())
            except ValueError:
                self.contact_matrix[t1][t2] = 0.0

        self.neighbor_order = self.spinBox.value()

        self.close()

    def on_reject(self) -> None:
        self.close()
