import sip

sip.setapi('QString', 1)
sip.setapi('QVariant', 1)

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from fancy_slider import FancySlider
from fancy_combo import FancyCombo

from SteeringPanelView import SteeringPanelView
from SteeringPanelModel import SteeringPanelModel
from SteeringEditorDelegate import SteeringEditorDelegate


class ItemData(object):
    def __init__(self, name, val, min_val=None, max_val=None, decimal_precision=3, enum=None,
                 widget_name=None):
        self._name = name
        self._val = None
        self._type = None

        if val is not None:
            self.val = val

        self._min = min_val
        self._min = min_val
        self._max = max_val
        self._decimal_precision = decimal_precision
        self.enum_allowed_widgets = ['combobox', 'pull-down']
        self._allowed_widget_names = ['lineedit', 'slider', 'combobox', 'pull-down']
        if widget_name in self.enum_allowed_widgets:
            assert isinstance(enum, list) or (enum is None), 'enum argument must be a list of None'
            if enum is None:
                enum = []
            # ensure all types in the enum list are the same
            test_list = [self.val] + enum

            type_set = set((map(lambda x: type(x), test_list)))
            assert len(type_set) == 1, 'enum list elements (together with initial value) must me of the same type. ' \
                                       'Instead I got the following types: {}'.format(','.join(map(lambda x: str(x),(type_set))))

            self._enum = enum
            if val is not None:
                try:
                    map(lambda x: str(x), self.enum).index(str(self.val))
                except ValueError:
                    self._enum = [str(self.val)] + self._enum  # prepending current value
        else:
            self._enum = None



        if widget_name is None:
            self._widget_name = 'lineedit'
        else:
            assert isinstance(widget_name, (str, None)), 'widget_name has to be a Python string or None object'
            assert widget_name.lower() in self._allowed_widget_names, \
                '{} is not supported. We support the following  widgets {}'.format(widget_name,
                                                                                   ','.join(self._allowed_widget_names))
            self._widget_name = widget_name.lower()

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
        self._type = type(self.val)

        print 'val.type=', self._type

    @property
    def enum(self):
        return self._enum

    @property
    def name(self):
        return self._name

    @property
    def widget_name(self):
        return self._widget_name

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def decimal_precision(self):
        return self._decimal_precision

    @decimal_precision.setter
    def decimal_precision(self, decimal_precision):
        self._decimal_precision = decimal_precision

    @property
    def item_type(self):
        return self._type





if __name__ == '__main__':
    item_data = []
    item_data.append(ItemData(name='vol', val=25, min_val=0, max_val=100, widget_name='slider'))
    item_data.append(
        ItemData(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))

    item_data.append(
        ItemData(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))

    item_data.append(
        ItemData(name='lam_vol_combo_float', val=2.0, enum=[1., 2., 3., 4.], widget_name='combobox'))

    item_data.append(
        ItemData(name='lam_vol_combo_int', val=2, enum=[1, 2, 3, 4], widget_name='combobox'))

    item_data.append(
        ItemData(name='lam_vol_combo', val='dupa2', enum=['dupa', 'dupa1', 'dupa2', 'dupa3'], widget_name='combobox'))

    item_data.append(
        ItemData(name='Empty', val=2.0, widget_name='slider'))


    item_data.append(ItemData(name='sur', val=20.2))
    item_data.append(ItemData(name='lam_sur', val=20.2))

    import sys

    app = QApplication(sys.argv)  # needs to be defined first

    window = QWidget()
    layout = QHBoxLayout()

    # model = QStandardItemModel(4, 2)

    # cdf = get_data_frame()
    model = SteeringPanelModel()
    model.update(item_data)
    # model.update_type_conv_fcn(get_types())

    tableView = SteeringPanelView()
    tableView.setModel(model)

    delegate = SteeringEditorDelegate()
    tableView.setItemDelegate(delegate)

    layout.addWidget(tableView)
    window.setLayout(layout)
    tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    # tableView.setVisible(False)
    # tableView.resizeColumnsToContents()
    # tableView.setVisible(True)

    # window.setWindowTitle("Spin Box Delegate")
    # tableView.setWindowTitle("Spin Box Delegate")
    # window.resize(QSize(800, 300))
    window.show()
    # tableView.show()
    sys.exit(app.exec_())
