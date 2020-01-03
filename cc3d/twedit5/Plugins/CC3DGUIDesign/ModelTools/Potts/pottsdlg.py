from collections import OrderedDict
from copy import deepcopy
from itertools import product
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolGUIBase import CC3DModelToolGUIBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.Potts.ui_pottsdlg import Ui_PottsGUI


class PottsGUI(CC3DModelToolGUIBase, Ui_PottsGUI):
    def __init__(self, parent=None, gpd: {} = None, cell_types: [] = None, valid_functions: [] = None):
        super(PottsGUI, self).__init__(parent)
        self.setupUi(self)

        self.gpd = deepcopy(gpd)
        self.valid_functions = valid_functions
        self.default_function = 'Default'
        if self.default_function not in self.valid_functions:
            self.valid_functions.append(self.default_function)

        self.user_decision = None

        self.cell_types = [cell_type for cell_type in cell_types if cell_type != "Medium"]

        if isinstance(self.gpd["MembraneFluctuations"], dict):
            self.show_cell_types = True
            self.type_params = {cell_type: self.gpd["MembraneFluctuations"]['Parameters'][cell_type]
                                for cell_type in self.cell_types}
        else:
            self.show_cell_types = False
            self.type_params = {cell_type: 0.0 for cell_type in self.cell_types}

        self.init_ui()

        self.connect_all_signals()

    def init_ui(self):
        self.xDimSB.setValue(self.gpd["Dim"][0])
        self.yDimSB.setValue(self.gpd["Dim"][1])
        self.zDimSB.setValue(self.gpd["Dim"][2])

        self.fluctuation_FcnCB.addItems(self.valid_functions)
        if self.cell_types:
            self.cell_TypeCB.addItems(self.cell_types)
            self.cell_TypeCB.setCurrentText(self.cell_types[0])
            self.type_ParamLE.setText(str(self.type_params[self.cell_TypeCB.currentText()]))
        else:
            self.type_ParamLE.setEnabled(False)

        if self.show_cell_types:
            self.membraneFluctuationsLE.setText(str(10))
            self.fluctuation_FcnCB.setCurrentText(self.gpd['MembraneFluctuations']['FunctionName'])
        else:
            self.membraneFluctuationsLE.setText(str(self.gpd["MembraneFluctuations"]))
            self.fluctuation_FcnCB.setCurrentText(self.default_function)
        self.membraneFluctuationsLE.setValidator(QDoubleValidator())
        self.type_ParamLE.setValidator(QDoubleValidator())

        self.neighborOrderSB.setValue(self.gpd["NeighborOrder"])

        self.mcsSB.setValue(self.gpd["MCS"])

        self.latticeTypeCB.setCurrentText(self.gpd["LatticeType"])

        self.xbcCB.setCurrentText(self.gpd["BoundaryConditions"]['x'])
        self.ybcCB.setCurrentText(self.gpd["BoundaryConditions"]['y'])
        self.zbcCB.setCurrentText(self.gpd["BoundaryConditions"]['z'])

        self.update_fluctuation_fields()

    def update_fluctuation_fields(self):
        if self.show_cell_types:
            self.cell_TypeCB.show()
            self.type_ParamLE.show()
            self.label_31.hide()
            self.membraneFluctuationsLE.hide()
        else:
            self.cell_TypeCB.hide()
            self.type_ParamLE.hide()
            self.label_31.show()
            self.membraneFluctuationsLE.show()

    def connect_all_signals(self):
        self.fluctuation_FcnCB.currentTextChanged.connect(self.on_function_change)
        self.cell_TypeCB.currentTextChanged.connect(self.on_cell_type_change)
        self.type_ParamLE.textChanged.connect(self.on_cell_param_change)

        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

    def on_function_change(self, fcn: str):
        if fcn == self.default_function:
            self.show_cell_types = False
        else:
            self.show_cell_types = True

        self.update_fluctuation_fields()

    def on_cell_type_change(self, text: str):
        self.type_ParamLE.setText(str(self.type_params[text]))

    def on_cell_param_change(self, text: str):
        self.type_params[self.cell_TypeCB.currentText()] = float(text)

    def on_accept(self) -> None:
        self.user_decision = True
        self.gpd["Dim"] = [self.xDimSB.value(), self.yDimSB.value(), self.zDimSB.value()]
        if self.fluctuation_FcnCB.currentText() == self.default_function:
            self.gpd["MembraneFluctuations"] = float(str(self.membraneFluctuationsLE.text()))
        else:
            self.gpd['MembraneFluctuations'] = {'Parameters': self.type_params,
                                                'FunctionName': self.fluctuation_FcnCB.currentText()}
        self.gpd["NeighborOrder"] = self.neighborOrderSB.value()
        self.gpd["MCS"] = self.mcsSB.value()
        self.gpd["LatticeType"] = str(self.latticeTypeCB.currentText())
        self.gpd["BoundaryConditions"]['x'] = self.xbcCB.currentText()
        self.gpd["BoundaryConditions"]['y'] = self.ybcCB.currentText()
        self.gpd["BoundaryConditions"]['z'] = self.zbcCB.currentText()
        self.close()

    def on_reject(self) -> None:
        self.user_decision = False
        self.close()

