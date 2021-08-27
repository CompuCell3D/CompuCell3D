from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.core.ParameterScanUtils import ParameterScanData
from cc3d.core.ParameterScanUtils import removeWhiteSpaces
from cc3d.core.ParameterScanUtils import extractListOfStrings
from . import ui_parvaldlg
from cc3d.core.ParameterScanEnums import *
from random import random
from math import log, exp

MAC = "qt_mac_set_native_menubar" in dir()


class ParValDlg(QDialog, ui_parvaldlg.Ui_ParValDlg):

    def __init__(self, parent=None):

        super(ParValDlg, self).__init__(parent)

        self.setupUi(self)

        self.updateUi()

        self.generatePB.clicked.connect(self.generate_values)

        self.typeCB.currentIndexChanged.connect(self.change_value_type)

        self.valueType = str(self.typeCB.currentText())

        # dict used provide easy mapping between type combo box index and types
        self.typeToCBindexDict = {FLOAT: 0, INT: 1, STRING: 2}

        self.psd = ParameterScanData()

    def set_identifier(self, val):
        """
        prepopulates identifier line edit
        :param val:
        :return:
        """
        self.identifierLE.setText(val)
        self.psd.previous_name = val

    def record_values(self):

        psd = self.psd

        psd.valueType = self.get_value_type()
        psd.name = self.identifierLE.text()
        psd.customValues = self.get_values()

    def change_value_type(self, _index):

        if not str(self.valuesLE.text()).strip(): return

        self.generate_values()

    def get_value_type(self):
        """
        returns string denoting type of the values in the generated list
        :return:
        """

        try:
            return VALUE_TYPE_DICT_REVERSE[str(self.typeCB.currentText())]
        except LookupError:
            return None

    def get_values(self, _castToType=None):
        """
        returns list of numerical values for parameter scan
        :param _castToType:
        :return:
        """

        value_str = str(self.valuesLE.text())
        if not len(value_str.strip()):
            raise ValueError('Empty value list. Make sure you clicked "Generate Values" prior to closing '
                             'dialog with "OK" button')

        value_str = removeWhiteSpaces(value_str)

        values = []

        if value_str == '':
            return values

        if value_str[-1] == ',':
            value_str = value_str[:-1]

        type_to_compare = self.get_value_type()

        if _castToType:
            type_to_compare = _castToType

            # we have to split values differently depending whether they are strings or numbers

        values = None

        if type_to_compare == STRING:

            values = extractListOfStrings(value_str)

        else:

            if len(value_str):
                values = value_str.split(',')

        if len(values):

            if type_to_compare == FLOAT:

                values = list(map(float, values))

            elif type_to_compare == INT:

                values = list(map(int, values))

        return values

    def generate_values(self):

        try:

            min_val = float(str(self.minLE.text()))

            max_val = float(str(self.maxLE.text()))

            steps = int(str(self.stepsLE.text()))

            value_type = self.get_value_type()

            distr = str(self.distrCB.currentText())

        except ValueError as e:

            return

        if min_val > max_val:
            min_val_str = str(self.minLE.text())

            max_val_str = str(self.maxLE.text())

            min_val_str, max_val_str = max_val_str, min_val_str

            self.minLE.setText(min_val_str)

            self.maxLE.setText(max_val_str)

            min_val, max_val = max_val, min_val

        if value_type == STRING:
            return

        values = []

        if distr == 'linear':

            if steps > 1:

                interval = (max_val - min_val) / float(steps - 1)

                values = [min_val + i * interval for i in range(steps)]

            else:

                values = [min_val]

        elif distr == 'random':

            values = [min_val + random() * (max_val - min_val) for i in range(steps)]

        elif distr == 'log':

            print('generating log distr')

            if min_val < 0. or max_val < 0.:
                QMessageBox.warning(self, "Wrong Min/Max values",
                                    "Please make sure that min and max values are positive "
                                    "for logarithmic distributions")

                return

            min_log, max_log = log(min_val), log(max_val)

            if steps > 1:

                interval = (max_log - min_log) / float(steps - 1)

                values = [min_log + i * interval for i in range(steps)]

            else:

                values = [min_log]

            values = list(map(exp, values))

        if value_type == INT:
            values = list(map(int, values))

        # remove duplicates from the list

        values = list(set(values))

        values.sort()

        # convert to string list
        values = list(map(str, values))

        values_str = ','.join(values)

        self.valuesLE.setText(values_str)

        # after sucessful type change we store new type
        self.valueType = str(self.typeCB.currentText())

    def updateUi(self):

        pass
