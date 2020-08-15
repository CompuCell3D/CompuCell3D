import errno
import shutil
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import *
import os
import sys
from . import ui_newsimulationwizard
from collections import OrderedDict
import cc3d
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.core.Validation.sanity_checkers import validate_cc3d_entity_identifier
from cc3d.twedit5.Plugins.CC3DMLGenerator.CC3DMLGeneratorBase import CC3DMLGeneratorBase
from .CC3DPythonGenerator import CC3DPythonGenerator

MAC = "qt_mac_set_native_menubar" in dir()


class NewSimulationWizard(QWizard, ui_newsimulationwizard.Ui_NewSimulationWizard):
    def __init__(self, parent=None):
        super(NewSimulationWizard, self).__init__(parent)

        self.cc3dProjectTreeWidget = parent
        self.plugin = self.cc3dProjectTreeWidget.plugin
        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows
        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.mainProjDir = ""
        self.simulationFilesDir = ""
        self.projectPath = ""
        self.setupUi(self)

        # This dictionary holds references to certain pages e.g. plugin configuration pages are inserted on demand
        # and access to those pages is facilitated via self.pageDict

        self.pageDict = {}

        self.updateUi()

        self.typeTable = []
        self.diffusantDict = {}
        self.chemotaxisData = {}

        if sys.platform.startswith('win'):
            self.setWizardStyle(QWizard.ClassicStyle)

    def display_invalid_entity_label_message(self, error_message):
        """
        Displays warning about invalid identifier
        :param error_message:
        :return:
        """

        QMessageBox.warning(self, 'Invalid Identifier', error_message)


    def keyPressEvent(self, event):

        if self.currentPage() == self.pageDict["CellType"][0]:
            cell_type = str(self.cellTypeLE.text())
            cell_type = cell_type.strip()

            if event.key() == Qt.Key_Return:
                if cell_type != "":
                    self.on_cellTypeAddPB_clicked()
                    event.accept()

                else:
                    next_button = self.button(QWizard.NextButton)
                    next_button.clicked.emit(True)

        elif self.currentPage() == self.pageDict["Diffusants"][0]:

            field_name = str(self.fieldNameLE.text())
            field_name = field_name.strip()

            if event.key() == Qt.Key_Return:

                if field_name != "":
                    self.on_fieldAddPB_clicked()

                    event.accept()

                else:

                    next_button = self.button(QWizard.NextButton)
                    next_button.clicked.emit(True)

        elif self.currentPage() == self.pageDict["ContactMultiCad"][0]:

            cadherin = str(self.cmcMoleculeLE.text()).strip()

            if event.key() == Qt.Key_Return:
                if cadherin != "":

                    self.on_cmcMoleculeAddPB_clicked()

                    event.accept()

                else:

                    next_button = self.button(QWizard.NextButton)

                    next_button.clicked.emit(True)

        elif self.currentPage() == self.pageDict["AdhesionFlex"][0]:

            molecule = str(self.afMoleculeLE.text()).strip()

            if event.key() == Qt.Key_Return:

                if molecule != "":
                    self.on_afMoleculeAddPB_clicked()
                    event.accept()

                else:
                    next_button = self.button(QWizard.NextButton)
                    next_button.clicked.emit(True)

        # last page
        elif self.currentPage() == self.pageDict["FinalPage"][0]:

            if event.key() == Qt.Key_Return:
                finish_button = self.button(QWizard.FinishButton)
                finish_button.clicked.emit(True)

        else:

            if event.key() == Qt.Key_Return:
                # move to the next page
                next_button = self.button(QWizard.NextButton)
                next_button.clicked.emit(True)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_piffPB_clicked(self):

        file_name = QFileDialog.getOpenFileName(self, "Choose PIFF file...")

        file_name = str(file_name)

        # normalizing path
        file_name = os.path.abspath(file_name)

        self.piffLE.setText(file_name)

    def hideConstraintFlexOption(self):

        self.volumeFlexCHB.setChecked(False)

        self.volumeFlexCHB.setHidden(True)

        self.surfaceFlexCHB.setChecked(False)

        self.surfaceFlexCHB.setHidden(True)

    def showConstraintFlexOption(self):

        if not self.growthCHB.isChecked() and not self.mitosisCHB.isChecked() and not self.deathCHB.isChecked():
            self.volumeFlexCHB.setHidden(False)

            self.surfaceFlexCHB.setHidden(False)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_extPotCHB_toggled(self, _flag):

        if _flag:
            self.extPotLocalFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_extPotLocalFlexCHB_toggled(self, _flag):

        if _flag:
            self.extPotCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_volumeFlexCHB_toggled(self, _flag):

        if _flag:
            self.volumeLocalFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_volumeLocalFlexCHB_toggled(self, _flag):

        if _flag:
            self.volumeFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_surfaceFlexCHB_toggled(self, _flag):

        if _flag:
            self.surfaceLocalFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_surfaceLocalFlexCHB_toggled(self, _flag):

        if _flag:
            self.surfaceFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_connectGlobalCHB_toggled(self, _flag):

        if _flag:
            self.connect2DCHB.setChecked(not _flag)

            self.connectGlobalByIdCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_connect2DCHB_toggled(self, _flag):

        if _flag:
            self.connectGlobalCHB.setChecked(not _flag)

            self.connectGlobalByIdCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_connectGlobalByIdCHB_toggled(self, _flag):

        if _flag:
            self.connect2DCHB.setChecked(not _flag)

            self.connectGlobalCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_lengthConstraintCHB_toggled(self, _flag):

        if _flag:
            self.lengthConstraintLocalFlexCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_lengthConstraintLocalFlexCHB_toggled(self, _flag):

        if _flag:
            self.lengthConstraintCHB.setChecked(not _flag)

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_growthCHB_toggled(self, _flag):

        if _flag:

            self.hideConstraintFlexOption()

        else:

            self.showConstraintFlexOption()

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_mitosisCHB_toggled(self, _flag):

        if _flag:

            self.hideConstraintFlexOption()

        else:

            self.showConstraintFlexOption()

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_deathCHB_toggled(self, _flag):

        if _flag:

            self.hideConstraintFlexOption()

        else:

            self.showConstraintFlexOption()

    @pyqtSlot()  # signature of the signal emited by the button
    def on_cellTypeAddPB_clicked(self):

        cell_type = str(self.cellTypeLE.text()).strip()
        try:
            validate_cc3d_entity_identifier(cell_type, entity_type_label='cell type')
        except AttributeError as e:
            self.display_invalid_entity_label_message(error_message=str(e))
            return

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

    @pyqtSlot()  # signature of the signal emited by the button
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

    @pyqtSlot()  # signature of the signal emited by the button
    def on_fieldAddPB_clicked(self):

        field_name = str(self.fieldNameLE.text()).strip()

        try:
            validate_cc3d_entity_identifier(field_name, entity_type_label='field label')
        except AttributeError as e:
            self.display_invalid_entity_label_message(error_message=str(e))
            return

        rows = self.fieldTable.rowCount()

        if field_name == "":
            return

        # check if cell type with this name already exist

        field_already_exists = False

        for row_id in range(rows):
            name = str(self.fieldTable.item(row_id, 0).text()).strip()
            print("CHECKING name=", name + "1", " type=", field_name + "1")
            print("name==cellType ", name == field_name)

            if name == field_name:
                field_already_exists = True

                break

        print("fieldAlreadyExists=", field_already_exists)
        if field_already_exists:
            print("WARNING")

            QMessageBox.warning(self, "Field name name already exists",
                                "Field name name already exist. Please choose different name", QMessageBox.Ok)

            return

        self.fieldTable.insertRow(rows)

        field_name_item = QTableWidgetItem(field_name)
        self.fieldTable.setItem(rows, 0, field_name_item)

        # picking solver name
        solver_name = str(self.solverCB.currentText()).strip()

        solver_name_item = QTableWidgetItem(solver_name)

        self.fieldTable.setItem(rows, 1, solver_name_item)

        # reset cell type entry line
        self.fieldNameLE.setText("")

    @pyqtSlot()  # signature of the signal emited by the button
    def on_clearFieldTablePB_clicked(self):

        rows = self.fieldTable.rowCount()

        for i in range(rows - 1, -1, -1):
            self.fieldTable.removeRow(i)

    # SECRETION
    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_secrConstConcRB_toggled(self, _flag):

        if _flag:
            self.secrRateLB.setText("Const. Concentration")
        else:

            self.secrRateLB.setText("Secretion Rate")

    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_secrOnContactRB_toggled(self, _flag):
        if _flag:

            self.secrAddOnContactPB.setHidden(False)
            self.secrOnContactCellTypeCB.setHidden(False)
            self.secrOnContactLE.setHidden(False)

        else:

            self.secrAddOnContactPB.setHidden(True)
            self.secrOnContactCellTypeCB.setHidden(True)
            self.secrOnContactLE.setHidden(True)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_secrAddOnContactPB_clicked(self):

        cell_type = str(self.secrOnContactCellTypeCB.currentText())

        current_text = str(self.secrOnContactLE.text())

        current_types = current_text.split(',')

        if current_text != "":
            if cell_type not in current_types:
                self.secrOnContactLE.setText(current_text + "," + cell_type)
        else:
            self.secrOnContactLE.setText(cell_type)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_secrAddRowPB_clicked(self):

        field = str(self.secrFieldCB.currentText()).strip()
        cell_type = str(self.secrCellTypeCB.currentText()).strip()

        try:
            secr_rate = float(str(self.secrRateLE.text()))
        except Exception:
            secr_rate = 0.0

        secr_on_contact = str(self.secrOnContactLE.text())

        secr_type = "uniform"
        if self.secrOnContactRB.isChecked():
            secr_type = "on contact"

        elif self.secrConstConcRB.isChecked():
            secr_type = "constant concentration"

        rows = self.secretionTable.rowCount()

        self.secretionTable.insertRow(rows)

        self.secretionTable.setItem(rows, 0, QTableWidgetItem(field))
        self.secretionTable.setItem(rows, 1, QTableWidgetItem(cell_type))
        self.secretionTable.setItem(rows, 2, QTableWidgetItem(str(secr_rate)))
        self.secretionTable.setItem(rows, 3, QTableWidgetItem(secr_on_contact))
        self.secretionTable.setItem(rows, 4, QTableWidgetItem(str(secr_type)))

        # reset entry lines
        self.secrOnContactLE.setText('')

    @pyqtSlot()  # signature of the signal emited by the button
    def on_secrRemoveRowsPB_clicked(self):

        selected_items = self.secretionTable.selectedItems()

        row_dict = {}
        for item in selected_items:
            row_dict[item.row()] = 0

        rows = list(row_dict.keys())

        rows.sort()

        rows_size = len(rows)
        for idx in range(rows_size - 1, -1, -1):
            row = rows[idx]
            self.secretionTable.removeRow(row)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_secrClearTablePB_clicked(self):

        rows = self.secretionTable.rowCount()

        for idx in range(rows - 1, -1, -1):
            self.secretionTable.removeRow(idx)

    # CHEMOTAXIS
    @pyqtSlot(bool)  # signature of the signal emited by the button
    def on_chemSatRB_toggled(self, _flag):

        if _flag:

            self.satCoefLB.setText("Saturation Coef.")
            self.satCoefLB.setHidden(False)
            self.satChemLE.setHidden(False)

        else:

            self.satCoefLB.setHidden(True)
            self.satChemLE.setHidden(True)
            self.satChemLE.setText('')

    @pyqtSlot(bool)  # signature of the signal emited by the radio button
    def on_chemSatLinRB_toggled(self, _flag):

        if _flag:
            self.satCoefLB.setText("Saturation Coef. Linear")
            self.satCoefLB.setHidden(False)
            self.satChemLE.setHidden(False)

        else:

            self.satCoefLB.setHidden(True)
            self.satChemLE.setHidden(True)
            self.satChemLE.setText('')

    @pyqtSlot()  # signature of the signal emited by the button
    def on_chemotaxTowardsPB_clicked(self):

        cell_type = str(self.chemTowardsCellTypeCB.currentText())

        current_text = str(self.chemotaxTowardsLE.text())

        current_types = current_text.split(',')

        if current_text != "":
            if cell_type not in current_types:
                self.chemotaxTowardsLE.setText(current_text + "," + cell_type)
        else:
            self.chemotaxTowardsLE.setText(cell_type)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_chemotaxisAddRowPB_clicked(self):

        field = str(self.chemFieldCB.currentText()).strip()

        cell_type = str(self.chemCellTypeCB.currentText()).strip()

        try:
            lambda_ = float(str(self.lambdaChemLE.text()))
        except Exception:
            lambda_ = 0.0

        saturation_coef = 0.0

        if not self.chemRegRB.isChecked():

            try:
                saturation_coef = float(str(self.satChemLE.text()))
            except Exception:
                saturation_coef = 0.0

        chemotax_towards_types = str(self.chemotaxTowardsLE.text())

        chemotaxis_type = "regular"

        if self.chemSatRB.isChecked():
            chemotaxis_type = "saturation"

        elif self.chemSatLinRB.isChecked():

            chemotaxis_type = "saturation linear"

        rows = self.chamotaxisTable.rowCount()

        self.chamotaxisTable.insertRow(rows)

        self.chamotaxisTable.setItem(rows, 0, QTableWidgetItem(field))
        self.chamotaxisTable.setItem(rows, 1, QTableWidgetItem(cell_type))
        self.chamotaxisTable.setItem(rows, 2, QTableWidgetItem(str(lambda_)))
        self.chamotaxisTable.setItem(rows, 3, QTableWidgetItem(chemotax_towards_types))
        self.chamotaxisTable.setItem(rows, 4, QTableWidgetItem(str(saturation_coef)))
        self.chamotaxisTable.setItem(rows, 5, QTableWidgetItem(chemotaxis_type))

        # reset entry lines

        self.chemotaxTowardsLE.setText('')

    @pyqtSlot()  # signature of the signal emited by the button
    def on_chemotaxisRemoveRowsPB_clicked(self):

        selected_items = self.chamotaxisTable.selectedItems()

        row_dict = {}
        for item in selected_items:
            row_dict[item.row()] = 0

        rows = list(row_dict.keys())

        rows.sort()

        rows_size = len(rows)

        for idx in range(rows_size - 1, -1, -1):
            row = rows[idx]
            self.chamotaxisTable.removeRow(row)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_chemotaxisClearTablePB_clicked(self):

        rows = self.chamotaxisTable.rowCount()

        for idx in range(rows - 1, -1, -1):
            self.chamotaxisTable.removeRow(idx)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_afMoleculeAddPB_clicked(self):

        molecule = str(self.afMoleculeLE.text()).strip()

        rows = self.afTable.rowCount()

        if molecule == "":
            return

        # check if molecule with this name already exist

        molecule_already_exists = False
        for rowId in range(rows):
            name = str(self.afTable.item(rowId, 0).text()).strip()

            if name == molecule:
                molecule_already_exists = True
                break

        if molecule_already_exists:
            QMessageBox.warning(self, "Molecule Name Already Exists",

                                "Molecule name already exist. Please choose different name", QMessageBox.Ok)

            return

        self.afTable.insertRow(rows)

        molecule_item = QTableWidgetItem(molecule)

        self.afTable.setItem(rows, 0, molecule_item)

        # reset molecule entry line
        self.afMoleculeLE.setText("")

        return

    @pyqtSlot()  # signature of the signal emited by the button
    def on_clearAFTablePB_clicked(self):

        rows = self.afTable.rowCount()

        for i in range(rows - 1, -1, -1):
            self.afTable.removeRow(i)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_cmcMoleculeAddPB_clicked(self):

        cadherin = str(self.cmcMoleculeLE.text()).strip()

        rows = self.cmcTable.rowCount()

        if cadherin == "":
            return

        # check if cadherin with this name already exist

        cadherin_already_exists = False

        for rowId in range(rows):
            name = str(self.cmcTable.item(rowId, 0).text()).strip()

            if name == cadherin:
                cadherin_already_exists = True
                break

        if cadherin_already_exists:
            QMessageBox.warning(self, "Cadherin Name Already Exists",
                                "Cadherin name already exist. Please choose different name", QMessageBox.Ok)

            return

        self.cmcTable.insertRow(rows)

        cadherin_item = QTableWidgetItem(cadherin)

        self.cmcTable.setItem(rows, 0, cadherin_item)

        # reset cadherin entry line
        self.cmcMoleculeLE.setText("")

    @pyqtSlot()  # signature of the signal emited by the button
    def on_clearCMCTablePB_clicked(self):

        rows = self.cmcTable.rowCount()

        for i in range(rows - 1, -1, -1):
            self.cmcTable.removeRow(i)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_dirPB_clicked(self):

        name = str(self.nameLE.text()).strip()

        proj_dir = self.plugin.configuration.setting("RecentNewProjectDir")
        if name != "":
            directory = QFileDialog.getExistingDirectory(self, "Specify Location for your project", proj_dir)
            self.plugin.configuration.setSetting("RecentNewProjectDir", directory)
            self.dirLE.setText(directory)

    # setting up validators for the entry fields
    def setUpValidators(self):

        self.membraneFluctuationsLE.setValidator(QDoubleValidator())
        self.secrRateLE.setValidator(QDoubleValidator())
        self.lambdaChemLE.setValidator(QDoubleValidator())
        self.satChemLE.setValidator(QDoubleValidator())

    # initialize properties dialog

    def updateUi(self):

        self.setUpValidators()

        # Multi cad plugin is being deprecated
        self.contactMultiCadCHB.setEnabled(False)

        # have to set base size in QDesigner and then read it to rescale columns.
        # For some reason reading size of the widget does not work properly

        page_ids = self.pageIds()

        self.pageDict["FinalPage"] = [self.page(page_ids[-1]), len(page_ids) - 1]
        self.pageDict["GeneralProperties"] = [self.page(1), 1]
        self.pageDict["CellType"] = [self.page(2), 2]
        self.pageDict["Diffusants"] = [self.page(3), 3]
        self.pageDict["Secretion"] = [self.page(5), 5]
        self.pageDict["Chemotaxis"] = [self.page(6), 6]
        self.pageDict["AdhesionFlex"] = [self.page(7), 7]
        self.pageDict["ContactMultiCad"] = [self.page(8), 8]
        self.pageDict["PythonScript"] = [self.page(9), 9]

        self.removePage(5)
        self.removePage(6)
        self.removePage(7)
        self.removePage(8)

        self.nameLE.selectAll()

        proj_dir = self.plugin.configuration.setting("RecentNewProjectDir")

        print("projDir=", str(proj_dir))

        if str(proj_dir) == "":
            proj_dir = os.environ["PREFIX_CC3D"]

        self.dirLE.setText(proj_dir)

        # self.cellTypeLE.setFocus(True)

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

        # general properties page

        self.piffPB.setHidden(True)
        self.piffLE.setHidden(True)

        # chemotaxis page

        base_size = self.fieldTable.baseSize()

        self.fieldTable.setColumnWidth(0, base_size.width() / 2)
        self.fieldTable.setColumnWidth(1, base_size.width() / 2)
        self.fieldTable.horizontalHeader().setStretchLastSection(True)

        self.satCoefLB.setHidden(True)
        self.satChemLE.setHidden(True)

        # secretion page

        base_size = self.secretionTable.baseSize()

        self.secretionTable.setColumnWidth(0, base_size.width() / 5)
        self.secretionTable.setColumnWidth(1, base_size.width() / 5)
        self.secretionTable.setColumnWidth(2, base_size.width() / 5)
        self.secretionTable.setColumnWidth(3, base_size.width() / 5)
        self.secretionTable.setColumnWidth(4, base_size.width() / 5)
        self.secretionTable.horizontalHeader().setStretchLastSection(True)
        self.secrAddOnContactPB.setHidden(True)
        self.secrOnContactCellTypeCB.setHidden(True)
        self.secrOnContactLE.setHidden(True)

        # AF molecule table
        self.afTable.horizontalHeader().setStretchLastSection(True)

        # CMC cadherin table
        self.cmcTable.horizontalHeader().setStretchLastSection(True)

        width = self.cellTypeTable.horizontalHeader().width()

        print("column 0 width=", self.cellTypeTable.horizontalHeader().sectionSize(0))
        print("column 1 width=", self.cellTypeTable.horizontalHeader().sectionSize(1))
        print("size=", self.cellTypeTable.size())
        print("baseSize=", self.cellTypeTable.baseSize())
        print("width=", width)
        print("column width=", self.cellTypeTable.columnWidth(0))

    def insertModulePage(self, _page):

        # get FinalPage id
        final_id = -1

        page_ids = self.pageIds()

        for page_id in page_ids:

            if self.page(page_id) == self.pageDict["FinalPage"]:
                final_id = page_id

                break

        if final_id == -1:
            print("COULD NOT INSERT PAGE  COULD NOT FIND LAST PAGE ")

            return

        print("FinalId=", final_id)

        self.setPage(final_id - 1, _page)

    def removeModulePage(self, _page):

        page_ids = self.pageIds()

        for page_id in page_ids:

            if self.page(page_id) == _page:
                self.removePage(page_id)
                break

    def validateCurrentPage(self):

        print("THIS IS VALIDATE FOR PAGE ", self.currentId)

        if self.currentId() == 0:
            directory = str(self.dirLE.text()).strip()
            name = str(self.nameLE.text()).strip()

            self.setPage(self.pageDict["PythonScript"][1], self.pageDict["PythonScript"][0])

            if directory == "" or name == "":
                QMessageBox.warning(self, "Missing information",
                                    "Please specify name of the simulation and directory where it should be written to",
                                    QMessageBox.Ok)
                return False

            else:
                if directory != "":
                    self.plugin.configuration.setSetting("RecentNewProjectDir", directory)
                    print("CHECKING DIRECTORY ")

                    # checking if directory is writeable
                    project_dir = os.path.abspath(directory)

                    if not os.path.exists(project_dir):
                        try:
                            os.makedirs(project_dir)
                        except OSError as e:
                            if e.errno != errno.EEXIST:
                                raise OSError(f'Could not create directory {project_dir}')

                    if not os.access(project_dir, os.W_OK):
                        print("CHECKING DIRECTORY ")
                        QMessageBox.warning(self, "Write permission Error",
                                            "You do not have write permissions to %s directory. "
                                            "This error also appears when creating project that has non-ascii "
                                            "characters (either in project name or in project directory). " % (
                                                os.path.abspath(directory)), QMessageBox.Ok)

                        return False

                return True

        # general properties        
        if self.currentId() == 1:

            if self.piffRB.isChecked() and str(self.piffLE.text()).strip() == '':
                QMessageBox.warning(self, "Missing information", "Please specify name of the PIFF file", QMessageBox.Ok)

                return False

            sim_3d_flag = False

            if self.xDimSB.value() > 1 and self.yDimSB.value() > 1 and self.zDimSB.value() > 1:
                sim_3d_flag = True

            if sim_3d_flag:

                self.lengthConstraintLocalFlexCHB.setChecked(False)
                self.lengthConstraintLocalFlexCHB.hide()

            else:
                self.lengthConstraintLocalFlexCHB.show()

            if str(self.latticeTypeCB.currentText()) == "Square" and not sim_3d_flag:
                self.connect2DCHB.show()

            else:
                self.connect2DCHB.hide()
                self.connect2DCHB.setChecked(False)

            return True

        if self.currentId() == 2:
            # we only extract types from table here - it is not a validation strictly speaking
            # extract cell type information form the table

            self.typeTable = []

            for row in range(self.cellTypeTable.rowCount()):
                cell_type = str(self.cellTypeTable.item(row, 0).text())
                freeze = False

                if self.cellTypeTable.item(row, 1).checkState() == Qt.Checked:
                    print("self.cellTypeTable.item(row,1).checkState()=", self.cellTypeTable.item(row, 1).checkState())
                    freeze = True

                self.typeTable.append([cell_type, freeze])

            return True

        if self.currentId() == 3:

            # we only extract diffusants from table here - it is not a validation strictly speaking
            # extract diffusants information form the table
            self.diffusantDict = {}

            for row in range(self.fieldTable.rowCount()):
                field = str(self.fieldTable.item(row, 0).text())
                solver = str(self.fieldTable.item(row, 1).text())

                try:
                    self.diffusantDict[solver].append(field)
                except LookupError:
                    self.diffusantDict[solver] = [field]

            # at this point we can fill all the cell types and fields widgets on subsequent pages

            self.chemCellTypeCB.clear()
            self.chemTowardsCellTypeCB.clear()
            self.chemFieldCB.clear()

            print("Clearing Combo boxes")
            for cell_type_tuple in self.typeTable:

                if str(cell_type_tuple[0]) != "Medium":
                    self.chemCellTypeCB.addItem(cell_type_tuple[0])

                self.chemTowardsCellTypeCB.addItem(cell_type_tuple[0])

            for solver_name, fields in self.diffusantDict.items():

                for field_name in fields:
                    self.chemFieldCB.addItem(field_name)

            # secretion plugin

            self.secrFieldCB.clear()
            self.secrCellTypeCB.clear()
            self.secrOnContactCellTypeCB.clear()

            for cell_type_tuple in self.typeTable:
                self.secrCellTypeCB.addItem(cell_type_tuple[0])
                self.secrOnContactCellTypeCB.addItem(cell_type_tuple[0])
            for solver_name, fields in self.diffusantDict.items():
                for field_name in fields:
                    self.secrFieldCB.addItem(field_name)

            return True

        if self.currentId() == 4:
            print(self.pageDict)

            if self.secretionCHB.isChecked():
                self.setPage(self.pageDict["Secretion"][1], self.pageDict["Secretion"][0])

            else:
                self.removePage(self.pageDict["Secretion"][1])

            if self.chemotaxisCHB.isChecked():
                self.setPage(self.pageDict["Chemotaxis"][1], self.pageDict["Chemotaxis"][0])

            else:
                self.removePage(self.pageDict["Chemotaxis"][1])

            if self.contactMultiCadCHB.isChecked():
                self.setPage(self.pageDict["ContactMultiCad"][1], self.pageDict["ContactMultiCad"][0])

            else:
                self.removePage(self.pageDict["ContactMultiCad"][1])

            if self.adhesionFlexCHB.isChecked():
                self.setPage(self.pageDict["AdhesionFlex"][1], self.pageDict["AdhesionFlex"][0])

            else:
                self.removePage(self.pageDict["AdhesionFlex"][1])

            return True

        if self.currentPage() == self.pageDict["ContactMultiCad"][0]:
            if not self.cmcTable.rowCount():

                QMessageBox.warning(self, "Missing information",
                                    "Please specify at least one cadherin name to be used in ContactMultiCad plugin",
                                    QMessageBox.Ok)

                return False

            else:
                return True

        if self.currentPage() == self.pageDict["AdhesionFlex"][0]:

            if not self.afTable.rowCount():

                QMessageBox.warning(self, "Missing information",
                                    "Please specify at least one adhesion molecule name "
                                    "to be used in AdhesionFlex plugin",
                                    QMessageBox.Ok)

                return False

            else:

                return True

        return True

    def makeProjectDirectories(self, dir, name):

        try:

            self.mainProjDir = os.path.join(dir, name)

            self.plugin.makeDirectory(self.mainProjDir)

            self.simulationFilesDir = os.path.join(self.mainProjDir, "Simulation")

            self.plugin.makeDirectory(self.simulationFilesDir)



        except IOError as e:

            raise IOError

        return

    def generateNewProject(self):

        directory = str(self.dirLE.text()).strip()

        directory = os.path.abspath(directory)

        name = str(self.nameLE.text()).strip()

        self.makeProjectDirectories(directory, name)

        self.generalPropertiesDict = {}

        self.generalPropertiesDict["Dim"] = [self.xDimSB.value(), self.yDimSB.value(), self.zDimSB.value()]
        self.generalPropertiesDict["MembraneFluctuations"] = float(str(self.membraneFluctuationsLE.text()))
        self.generalPropertiesDict["NeighborOrder"] = self.neighborOrderSB.value()
        self.generalPropertiesDict["MCS"] = self.mcsSB.value()
        self.generalPropertiesDict["LatticeType"] = str(self.latticeTypeCB.currentText())
        self.generalPropertiesDict["SimulationName"] = name
        self.generalPropertiesDict["BoundaryConditions"] = OrderedDict()
        self.generalPropertiesDict["BoundaryConditions"]['x'] = self.xbcCB.currentText()
        self.generalPropertiesDict["BoundaryConditions"]['y'] = self.ybcCB.currentText()
        self.generalPropertiesDict["BoundaryConditions"]['z'] = self.zbcCB.currentText()
        self.generalPropertiesDict["Initializer"] = ["uniform", None]

        if self.blobRB.isChecked():

            self.generalPropertiesDict["Initializer"] = ["blob", None]

        elif self.piffRB.isChecked():

            piff_path = str(self.piffLE.text()).strip()
            self.generalPropertiesDict["Initializer"] = ["piff", piff_path]

            # trying to copy piff file into simulation dir of the project directory
            try:

                shutil.copy(piff_path, self.simulationFilesDir)

                base_piff_path = os.path.basename(piff_path)
                relative_piff_path = os.path.join(self.simulationFilesDir, base_piff_path)
                self.generalPropertiesDict["Initializer"][1] = self.getRelativePathWRTProjectDir(relative_piff_path)

                print("relativePathOF PIFF=", self.generalPropertiesDict["Initializer"][1])

            except shutil.Error:
                QMessageBox.warning(self, "Cannot copy PIFF file",
                                    "Cannot copy PIFF file into project directory. "
                                    "Please check if the file exists and that you have necessary write permissions",
                                    QMessageBox.Ok)

            except IOError as e:
                QMessageBox.warning(self, "IO Error", e.__str__(), QMessageBox.Ok)

        self.cellTypeData = {}

        # extract cell type information form the table
        for row in range(self.cellTypeTable.rowCount()):

            cell_type = str(self.cellTypeTable.item(row, 0).text())
            freeze = False

            if self.cellTypeTable.item(row, 1).checkState() == Qt.Checked:
                print("self.cellTypeTable.item(row,1).checkState()=", self.cellTypeTable.item(row, 1).checkState())
                freeze = True

            self.cellTypeData[row] = [cell_type, freeze]

        self.af_data = {}

        for row in range(self.afTable.rowCount()):
            molecule = str(self.afTable.item(row, 0).text())

            self.af_data[row] = molecule

        self.af_formula = str(self.bindingFormulaLE.text()).strip()

        cmc_table = []

        for row in range(self.cmcTable.rowCount()):
            cadherin = str(self.cmcTable.item(row, 0).text())

            cmc_table.append(cadherin)

        self.pde_field_data = {}

        for row in range(self.fieldTable.rowCount()):
            chem_field_name = str(self.fieldTable.item(row, 0).text())

            solver_name = str(self.fieldTable.item(row, 1).text())

            self.pde_field_data[chem_field_name] = solver_name

        self.secretion_data = {}  # format {field:[secrDict1,secrDict2,...]}

        for row in range(self.secretionTable.rowCount()):

            secr_field_name = str(self.secretionTable.item(row, 0).text())
            cell_type = str(self.secretionTable.item(row, 1).text())

            try:
                rate = float(str(self.secretionTable.item(row, 2).text()))
            except Exception:
                rate = 0.0

            on_contact_with = str(self.secretionTable.item(row, 3).text())

            secretion_type = str(self.secretionTable.item(row, 4).text())

            secr_dict = {}

            secr_dict["CellType"] = cell_type
            secr_dict["Rate"] = rate
            secr_dict["OnContactWith"] = on_contact_with
            secr_dict["SecretionType"] = secretion_type

            try:
                self.secretion_data[secr_field_name].append(secr_dict)
            except LookupError:
                self.secretion_data[secr_field_name] = [secr_dict]

        self.chemotaxisData = {}  # format {field:[chemDict1,chemDict2,...]}

        for row in range(self.chamotaxisTable.rowCount()):
            chem_field_name = str(self.chamotaxisTable.item(row, 0).text())

            cell_type = str(self.chamotaxisTable.item(row, 1).text())

            try:
                lambda_ = float(str(self.chamotaxisTable.item(row, 2).text()))
            except Exception:
                lambda_ = 0.0

            chemotax_towards = str(self.chamotaxisTable.item(row, 3).text())

            try:
                sat_coef = float(str(self.chamotaxisTable.item(row, 4).text()))
            except Exception:
                sat_coef = 0.0

            chemotaxis_type = str(self.chamotaxisTable.item(row, 5).text())
            chem_dict = {}

            chem_dict["CellType"] = cell_type
            chem_dict["Lambda"] = lambda_
            chem_dict["ChemotaxTowards"] = chemotax_towards
            chem_dict["SatCoef"] = sat_coef
            chem_dict["ChemotaxisType"] = chemotaxis_type

            try:
                self.chemotaxisData[chem_field_name].append(chem_dict)
            except LookupError:
                self.chemotaxisData[chem_field_name] = [chem_dict]

        # constructing Project XMl Element
        simulation_element = ElementCC3D("Simulation", {"version": cc3d.__version__})
        xml_generator = CC3DMLGeneratorBase(self.simulationFilesDir, name)

        self.generateXML(xml_generator)

        # end of generate XML ------------------------------------------------------------------------------------

        if self.pythonXMLRB.isChecked():
            xml_file_name = os.path.join(self.simulationFilesDir, name + ".xml")
            xml_generator.saveCC3DXML(xml_file_name)

            simulation_element.ElementCC3D("XMLScript", {"Type": "XMLScript"},

                                          self.getRelativePathWRTProjectDir(xml_file_name))

            # end of generate XML ------------------------------------------------------------------------------------

        if self.pythonXMLRB.isChecked() or self.pythonOnlyRB.isChecked():
            # generate Python ------------------------------------------------------------------------------------

            python_generator = CC3DPythonGenerator(xml_generator)

            python_generator.set_python_only_flag(self.pythonOnlyRB.isChecked())

            self.generateSteppablesCode(python_generator)

            # before calling generateMainPythonScript we have to call generateSteppablesCode
            # that generates also steppable registration lines
            python_generator.generate_main_python_script()
            simulation_element.ElementCC3D("PythonScript", {"Type": "PythonScript"},
                                           self.getRelativePathWRTProjectDir(python_generator.mainPythonFileName))

            simulation_element.ElementCC3D("Resource", {"Type": "Python"},
                                           self.getRelativePathWRTProjectDir(python_generator.steppablesPythonFileName))

            # end of generate Python ---------------------------------------------------------------------------------
        # including PIFFile in the .cc3d project description
        if self.generalPropertiesDict["Initializer"][0] == "piff":
            simulation_element.ElementCC3D("PIFFile", {}, self.generalPropertiesDict["Initializer"][1])

        # save Project file
        proj_file_name = os.path.join(self.mainProjDir, name + ".cc3d")

        # simulationElement.CC3DXMLElement.saveXML(projFileName)
        proj_file = open(proj_file_name, 'w')
        proj_file.write('%s' % simulation_element.CC3DXMLElement.getCC3DXMLElementString())
        proj_file.close()

        # open newly created project in the ProjectEditor
        self.plugin.openCC3Dproject(proj_file_name)

    def generateSteppablesCode(self, pythonGenerator):

        if self.growthCHB.isChecked():
            pythonGenerator.generate_growth_steppable()

        if self.mitosisCHB.isChecked():
            pythonGenerator.generate_mitosis_steppable()

        if self.deathCHB.isChecked():
            pythonGenerator.generate_death_steppable()

        pythonGenerator.generate_vis_plot_steppables()

        pythonGenerator.generate_steppable_python_script()

        pythonGenerator.generate_steppable_registration_lines()

    def generateXML(self, generator):

        cell_type_dict = self.cellTypeData
        args = []

        kwds = {}

        kwds['insert_root_element'] = generator.cc3d
        kwds['data'] = cell_type_dict
        kwds['generalPropertiesData'] = self.generalPropertiesDict
        kwds['afData'] = self.af_data
        kwds['formula'] = self.af_formula
        kwds['chemotaxisData'] = self.chemotaxisData
        kwds['pdeFieldData'] = self.pde_field_data
        kwds['secretionData'] = self.secretion_data

        generator.generateMetadataSimulationProperties(*args, **kwds)

        generator.generatePottsSection(*args, **kwds)

        generator.generateCellTypePlugin(*args, **kwds)

        if self.volumeFlexCHB.isChecked():
            generator.generateVolumeFlexPlugin(*args, **kwds)

        if self.surfaceFlexCHB.isChecked():
            generator.generateSurfaceFlexPlugin(*args, **kwds)

        if self.volumeLocalFlexCHB.isChecked():
            generator.generateVolumeLocalFlexPlugin(*args, **kwds)

        if self.surfaceLocalFlexCHB.isChecked():
            generator.generateSurfaceLocalFlexPlugin(*args, **kwds)

        if self.extPotCHB.isChecked():
            generator.generateExternalPotentialPlugin(*args, **kwds)

        if self.extPotLocalFlexCHB.isChecked():
            generator.generateExternalPotentialLocalFlexPlugin(*args, **kwds)

        if self.comCHB.isChecked():
            generator.generateCenterOfMassPlugin(*args, **kwds)

        if self.neighborCHB.isChecked():
            generator.generateNeighborTrackerPlugin(*args, **kwds)

        if self.momentOfInertiaCHB.isChecked():
            generator.generateMomentOfInertiaPlugin(*args, **kwds)

        if self.pixelTrackerCHB.isChecked():
            generator.generatePixelTrackerPlugin(*args, **kwds)

        if self.boundaryPixelTrackerCHB.isChecked():
            generator.generateBoundaryPixelTrackerPlugin(*args, **kwds)

        if self.contactCHB.isChecked():
            generator.generateContactPlugin(*args, **kwds)

        if self.compartmentCHB.isChecked():
            generator.generateCompartmentPlugin(*args, **kwds)

        if self.internalContactCB.isChecked():
            generator.generateContactInternalPlugin(*args, **kwds)

        if self.contactLocalProductCHB.isChecked():
            generator.generateContactLocalProductPlugin(*args, **kwds)

        if self.fppCHB.isChecked():
            generator.generateFocalPointPlasticityPlugin(*args, **kwds)

        if self.elasticityCHB.isChecked():
            generator.generateElasticityTrackerPlugin(*args, **kwds)

            generator.generateElasticityPlugin(*args, **kwds)

        if self.adhesionFlexCHB.isChecked():
            generator.generateAdhesionFlexPlugin(*args, **kwds)

        if self.chemotaxisCHB.isChecked():
            generator.generateChemotaxisPlugin(*args, **kwds)

        if self.lengthConstraintCHB.isChecked():
            generator.generateLengthConstraintPlugin(*args, **kwds)

        if self.lengthConstraintLocalFlexCHB.isChecked():
            generator.generateLengthConstraintLocalFlexPlugin(*args, **kwds)

        if self.connectGlobalCHB.isChecked():
            generator.generateConnectivityGlobalPlugin(*args, **kwds)

        if self.connectGlobalByIdCHB.isChecked():
            generator.generateConnectivityGlobalByIdPlugin(*args, **kwds)

        if self.connect2DCHB.isChecked():
            generator.generateConnectivityPlugin(*args, **kwds)

        if self.secretionCHB.isChecked():
            generator.generateSecretionPlugin(*args, **kwds)

            # if self.pdeSolverCallerCHB.isChecked():

            # xmlGenerator.generatePDESolverCaller()

        # PDE solvers

        # getting a list of solvers to be generated

        list_of_solvers = list(self.diffusantDict.keys())

        for solver in list_of_solvers:
            solver_generator_fcn = getattr(generator, 'generate' + solver)

            solver_generator_fcn(*args, **kwds)

            # if self.fieldTable.rowCount():

            # generator.generateDiffusionSolverFE(*args,**kwds)            

            # generator.generateFlexibleDiffusionSolverFE(*args,**kwds)            

            # generator.generateFastDiffusionSolver2DFE(*args,**kwds)            

            # generator.generateKernelDiffusionSolver(*args,**kwds)            

            # generator.generateSteadyStateDiffusionSolver(*args,**kwds)            

        if self.boxWatcherCHB.isChecked():
            generator.generateBoxWatcherSteppable(*args, **kwds)

        # cell layout initializer

        if self.uniformRB.isChecked():
            generator.generateUniformInitializerSteppable(*args, **kwds)

        elif self.blobRB.isChecked():
            generator.generateBlobInitializerSteppable(*args, **kwds)

        elif self.piffRB.isChecked():
            generator.generatePIFInitializerSteppable(*args, **kwds)

        if self.pifDumperCHB.isChecked():
            generator.generatePIFDumperSteppable(*args, **kwds)

    def findRelativePathSegments(self, basePath, p, rest=[]):

        """

            This function finds relative path segments of path p with respect to base path    

            It returns list of relative path segments and flag whether operation succeeded or not    

        """

        h, t = os.path.split(p)

        pathMatch = False

        if h == basePath:
            pathMatch = True

            return [t] + rest, pathMatch

        print("(h,t,pathMatch)=", (h, t, pathMatch))

        if len(h) < 1: return [t] + rest, pathMatch

        if len(t) < 1: return [h] + rest, pathMatch

        return self.findRelativePathSegments(basePath, h, [t] + rest)

    def findRelativePath(self, basePath, p):

        relativePathSegments, pathMatch = self.findRelativePathSegments(basePath, p)

        if pathMatch:

            relative_path = ""

            for i in range(len(relativePathSegments)):

                segment = relativePathSegments[i]

                relative_path += segment

                if i != len(relativePathSegments) - 1:
                    relative_path += "/"  # we use unix style separators - they work on all (3) platforms

            return relative_path

        else:

            return p

    def getRelativePathWRTProjectDir(self, path):

        return self.findRelativePath(self.mainProjDir, path)
