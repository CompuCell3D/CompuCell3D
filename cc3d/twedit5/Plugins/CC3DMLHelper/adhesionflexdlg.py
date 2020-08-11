from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_adhesionflexdlg

MAC = "qt_mac_set_native_menubar" in dir()


class AdhesionFlexDlg(QDialog, ui_adhesionflexdlg.Ui_AdhesionFlexDlg):

    def __init__(self, _currentEditor=None, parent=None):

        super(AdhesionFlexDlg, self).__init__(parent)

        self.editorWindow = parent

        self.setupUi(self)

        if not MAC:
            self.cancelPB.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

    def keyPressEvent(self, event):

        molecule = str(self.afMoleculeLE.text()).strip()

        if event.key() == Qt.Key_Return:

            if molecule != "":
                self.on_afMoleculeAddPB_clicked()

                event.accept()

    @pyqtSlot()
    def on_afMoleculeAddPB_clicked(self):

        molecule = str(self.afMoleculeLE.text()).strip()

        rows = self.afTable.rowCount()

        if molecule == "":
            return

        # check if molecule with this name already exist

        moleculeAlreadyExists = False

        for rowId in range(rows):

            name = str(self.afTable.item(rowId, 0).text()).strip()

            if name == molecule:
                moleculeAlreadyExists = True

                break

        if moleculeAlreadyExists:
            QMessageBox.warning(self, "Molecule Name Already Exists",

                                "Molecule name already exist. Please choose different name", QMessageBox.Ok)

            return

        self.afTable.insertRow(rows)

        moleculeItem = QTableWidgetItem(molecule)

        self.afTable.setItem(rows, 0, moleculeItem)

        # reset molecule entry line

        self.afMoleculeLE.setText("")

        return

    @pyqtSlot()
    def on_clearAFTablePB_clicked(self):

        rows = self.afTable.rowCount()

        for i in range(rows - 1, -1, -1):
            self.afTable.removeRow(i)

    def extractInformation(self):

        adhDict = {}

        for row in range(self.afTable.rowCount()):
            molecule = str(self.afTable.item(row, 0).text())

            adhDict[row] = molecule

        return adhDict, str(self.bindingFormulaLE.text())

    def updateUi(self):

        self.afTable.horizontalHeader().setStretchLastSection(True)
