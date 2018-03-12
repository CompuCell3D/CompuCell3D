import sip

sip.setapi('QString', 1)
sip.setapi('QVariant', 1)

# from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from fancy_slider import FancySlider
# from fancy_combo import FancyCombo

from SteeringParam import SteeringParam
from SteeringPanelView import SteeringPanelView
from SteeringPanelModel import SteeringPanelModel
from SteeringEditorDelegate import SteeringEditorDelegate



if __name__ == '__main__':
    item_data = []
    item_data.append(SteeringParam(name='vol', val=25, min_val=0, max_val=100, widget_name='slider'))
    item_data.append(
        SteeringParam(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))

    item_data.append(
        SteeringParam(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))

    item_data.append(
        SteeringParam(name='lam_vol_combo_float', val=2.0, enum=[1., 2., 3., 4.], widget_name='combobox'))

    item_data.append(
        SteeringParam(name='lam_vol_combo_int', val=2, enum=[1, 2, 3, 4], widget_name='combobox'))

    item_data.append(
        SteeringParam(name='lam_vol_combo', val='dupa2', enum=['dupa', 'dupa1', 'dupa2', 'dupa3'], widget_name='combobox'))

    item_data.append(
        SteeringParam(name='Empty', val=2.0, widget_name='slider'))


    item_data.append(SteeringParam(name='sur', val=20.2))
    item_data.append(SteeringParam(name='lam_sur', val=20.2))

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
