from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.core.XMLUtils import ElementCC3D
from . import ui_potts
from random import randint
import sys

MAC = "qt_mac_set_native_menubar" in dir()


class PottsDlg(QDialog, ui_potts.Ui_PottsDlg):

    def __init__(self, _currentEditor=None, parent=None):

        super(PottsDlg, self).__init__(parent)

        self.editorWindow = parent

        self.setupUi(self)

        if not MAC:
            self.cancelPB.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

    def keyPressEvent(self, event):

        return

        # cellType = str(self.cellTypeLE.text())

        # cellType = string.rstrip(cellType)

        #

        # if event.key() == Qt.Key_Return:

        #     if cellType != "":

        #         self.on_cellTypeAddPB_clicked()

        #         event.accept()

    def extractInformation(self):

        cellTypeDict = {}

        for row in range(self.cellTypeTable.rowCount()):

            type = str(self.cellTypeTable.item(row, 0).text())

            freeze = False

            if self.cellTypeTable.item(row, 1).checkState() == Qt.Checked:
                print("self.cellTypeTable.item(row,1).checkState()=", self.cellTypeTable.item(row, 1).checkState())

                freeze = True

            cellTypeDict[row] = [type, freeze]

        return cellTypeDict

    def initialize(self, gpd):

        """

        initializes dialog based on current content of the Potts Section

        :param gpd: {dict} dictionary representing current content of the Potts section

        :return: None

        """

        self.xDimSB.setValue(gpd['Dim'][0])

        self.yDimSB.setValue(gpd['Dim'][1])

        self.zDimSB.setValue(gpd['Dim'][2])

        self.membraneFluctuationsLE.setText(str(gpd['MembraneFluctuations']))

        self.mcsSB.setValue(int(gpd['MCS']))

        self.neighborOrderSB.setValue(int(gpd['NeighborOrder']))

    def updateUi(self):

        pass

    def generateXML(self):

        mElement = ElementCC3D('Potts')

        mElement.addComment("newline")

        mElement.addComment("Basic properties of CPM (GGH) algorithm")

        mElement.ElementCC3D("Dimensions",
                             {"x": self.xDimSB.value(), "y": self.yDimSB.value(), "z": self.zDimSB.value()})

        mElement.ElementCC3D("Steps", {}, self.mcsSB.value())

        if self.anneal_mcsSB.value() != 0:
            mElement.ElementCC3D("Anneal", {}, self.anneal_mcsSB.value())

        mElement.ElementCC3D("Temperature", {}, float(str(self.membraneFluctuationsLE.text())))

        mElement.ElementCC3D("NeighborOrder", {}, self.neighborOrderSB.value())

        if str(self.latticeTypeCB.currentText()) != "Square":
            mElement.ElementCC3D("LatticeType", {}, str(self.latticeTypeCB.currentText()))

        for dim_name_bc_widget, dim_name in [(self.xbcCB, 'x'), (self.ybcCB, 'y'), (self.zbcCB, 'z')]:

            try:

                if str(dim_name_bc_widget.currentText()) == 'Periodic':
                    mElement.ElementCC3D('Boundary_' + dim_name, {}, 'Periodic')

            except KeyError:

                pass

                # mElement.ElementCC3D('Boundary_' + dim_name, {}, 'NoFlux')

        if self.auto_gen_rand_seed_CB.isChecked():
            mElement.ElementCC3D('RandomSeed', {}, randint(0, sys.maxsize))

        return mElement
