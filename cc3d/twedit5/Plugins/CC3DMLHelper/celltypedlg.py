from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_celltypedlg

MAC = "qt_mac_set_native_menubar" in dir()


class CellTypeDlg(QDialog, ui_celltypedlg.Ui_CellTypeDlg):

    def __init__(self, _currentEditor=None, parent=None):

        super(CellTypeDlg, self).__init__(parent)

        self.editorWindow = parent

        self.setupUi(self)

        if not MAC:
            self.cancelPB.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

    def keyPressEvent(self, event):

        cell_type = str(self.cellTypeLE.text()).strip()

        if event.key() == Qt.Key_Return:

            if cell_type != "":
                self.on_cellTypeAddPB_clicked()

                event.accept()

    @pyqtSlot()
    def on_cellTypeAddPB_clicked(self):

        cell_type = str(self.cellTypeLE.text()).strip()

        rows = self.cellTypeTable.rowCount()

        if cell_type == "":
            return

        # check if cell type with this name already exist

        cell_type_already_exists = False

        for rowId in range(rows):

            name = str(self.cellTypeTable.item(rowId, 0).text()).strip()

            print("CHECKING name=", name + "1", " type=", cell_type + "1")

            print("name==cellType ", name == cell_type)

            if name == cell_type:
                cell_type_already_exists = True

                break

        print("cellTypeAlreadyExists=", cell_type_already_exists)

        if cell_type_already_exists:
            print("WARNING")

            QMessageBox.warning(self, "Cell type name already exists",

                                "Cell type name already exist. Please choose different name", QMessageBox.Ok)

            return

        self.cellTypeTable.insertRow(rows)

        cell_type_item = QTableWidgetItem(cell_type)

        self.cellTypeTable.setItem(rows, 0, cell_type_item)

        cell_type_freeze_item = QTableWidgetItem()

        cell_type_freeze_item.data(Qt.CheckStateRole)

        if self.freezeCHB.isChecked():

            cell_type_freeze_item.setCheckState(Qt.Checked)

        else:

            cell_type_freeze_item.setCheckState(Qt.Unchecked)

        self.cellTypeTable.setItem(rows, 1, cell_type_freeze_item)

        # reset cell type entry line

        self.cellTypeLE.setText("")

        return

    @pyqtSlot()
    def on_clearCellTypeTablePB_clicked(self):

        rows = self.cellTypeTable.rowCount()

        for i in range(rows - 1, -1, -1):
            self.cellTypeTable.removeRow(i)

        # insert Medium

        self.cellTypeTable.insertRow(0)

        medium_item = QTableWidgetItem("Medium")

        self.cellTypeTable.setItem(0, 0, medium_item)

        medium_freeze_item = QTableWidgetItem()

        medium_freeze_item.data(Qt.CheckStateRole)

        medium_freeze_item.setCheckState(Qt.Unchecked)

        self.cellTypeTable.setItem(0, 1, medium_freeze_item)

    def extractInformation(self):

        cell_type_dict = {}

        for row in range(self.cellTypeTable.rowCount()):

            cell_type = str(self.cellTypeTable.item(row, 0).text())

            freeze = False

            if self.cellTypeTable.item(row, 1).checkState() == Qt.Checked:
                print("self.cellTypeTable.item(row,1).checkState()=", self.cellTypeTable.item(row, 1).checkState())

                freeze = True

            cell_type_dict[row] = [cell_type, freeze]

        return cell_type_dict

    def updateUi(self):

        self.cellTypeTable.insertRow(0)

        medium_item = QTableWidgetItem("Medium")

        self.cellTypeTable.setItem(0, 0, medium_item)

        medium_freeze_item = QTableWidgetItem()

        medium_freeze_item.data(Qt.CheckStateRole)

        medium_freeze_item.setCheckState(Qt.Unchecked)

        self.cellTypeTable.setItem(0, 1, medium_freeze_item)

        base_size = self.cellTypeTable.baseSize()

        self.cellTypeTable.setColumnWidth(0, base_size.width() / 2)

        self.cellTypeTable.setColumnWidth(1, base_size.width() / 2)

        self.cellTypeTable.horizontalHeader().setStretchLastSection(True)
