
import sip

sip.setapi('QString', 1)
sip.setapi('QVariant', 1)

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from fancy_slider import SliderWithValue




class EditorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if not index.isValid():
            return

        # column_name = self.get_col_name_from_index(index)
        col_name = self.get_col_name(index)

        if col_name == 'Value':

            # # editor = QLineEdit(parent)
            # # slider = QSlider(parent)
            # # slider.setRange(0,100)
            # # slider.setTickInterval(10)
            # # slider.setOrientation(Qt.Horizontal)
            # # slider.setAutoFillBackground(True)
            # slider = SliderWithValue(parent)
            # slider.setOrientation(Qt.Horizontal)
            # slider.setMinimum(0)
            # slider.setMaximum(100)
            # slider.setValue(5)
            # # slider.setRange(0,100)
            # # slider.setTickInterval(10)
            #
            # editor = slider
            # # editor = QSlider(parent)
            item = index.model().get_item(index)

            if item.widget_name=='slider':
                editor = self.init_slider(parent, index)
            else:
                editor = QLineEdit(parent)

        else:
            return None

        return editor

    def init_slider(self,parent, index):
        """
        initializes slider based on the current value of the index
        :param index: {index}
        :return:{QSlider instance}
        """


        item = index.model().get_item(index)

        slider = SliderWithValue(parent)
        slider.setOrientation(Qt.Horizontal)
        if item.min is None:
            slider.setMinimum(0)
        else:
            slider.setMinimum(item.min)
        if item.max is None:
            slider.setMaximum(10*item.val)
        else:
            slider.setMaximum(item.max)

        slider.setTickInterval(2)

        # slider.setMaximum(1000)
        slider.setValue(item.val)
        # slider.setRange(0,100)
        # slider.setTickInterval(10)


        return slider




    def get_col_name(self, index):
        """
        returns column name
        :param index: {Index}
        :return: {str or None}
        """
        if not index.isValid():
            return None

        model = index.model()
        return model.header_data[index.column()]

    def get_item_type(self, index):
        """
        Returns type of element
        :param index: {index}
        :return: {type}
        """
        if not index.isValid():
            return None

        model = index.model()
        return model.item_data[index.row()].type

    def setEditorData(self, editor, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return
        if column_name == 'Value':
            value = index.model().data(index, Qt.DisplayRole)
            print 'i,j=', index.column(), index.row()
            print'val=', value
            # editor.setText(str(value.toInt()))
            try:
                editor.setText(str(value))
            except:
                editor.setTickPosition(50)
        else:
            return

    def setModelData(self, editor, model, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return

        if column_name == 'Value':

            item = index.model().get_item(index)
            type_conv_fcn = self.get_item_type(index)
            if item.widget_name=='slider':

                try:
                    value = type_conv_fcn(editor.value())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return

            else:
                try:
                    value = type_conv_fcn(editor.text())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return


            #
            # type_conv_fcn = self.get_item_type(index)
            # print 'type_conv_fcn=', type_conv_fcn
            # try:
            #     value = type_conv_fcn(editor.text())
            # except ValueError as exc:
            #     QMessageBox.warning(None,'Type Conversion Error',str(exc))
            #     return
        else:
            return
            # editor.interpretText()
            # value = editor.value()

        model.setData(index, value, Qt.EditRole)


    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ItemData(object):
    def __init__(self, name=None, val=None, min=None, max=None, widget_name=None):
        self._name = name
        self._val = None
        if val is not None:
            self.val = val

        self._type = None
        self._min = min
        self._max = max
        self._enum = None
        if widget_name is None:
            self._widget_name = 'lineedit'
        else:
            self._widget_name = widget_name

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val
        self.type = type(self.val)

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



class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__()

        self.item_data = None
        self.header_data = [ 'Value', 'Type']
        self.item_data_attr_name = {

            0: 'val',
            1: 'type',

        }

    def update(self, item_data):

        self.item_data = item_data


    def headerData(self, p_int, orientation, role=None):

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            try:
                return self.header_data[p_int]
            except IndexError:
                return QVariant()

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            try:
                return self.item_data[p_int].name
            except IndexError:
                return QVariant()


        return QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.item_data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.header_data)

    def get_item(self,index):
        if not index.isValid():
            return

        i = index.row()
        return self.item_data[i]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # print 'Data Call'
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            item = self.item_data[i]
            item_data_to_display = getattr(item, self.item_data_attr_name[j])
            return '{}'.format(item_data_to_display)

        elif role == Qt.BackgroundRole:
            batch = (index.row()) % 2
            if batch == 0:
                return QtGui.QColor('white')
                # return QApplication.palette().base()
            else:
                return QtGui.QColor('gray')
            # return QApplication.palette().alternateBase()


        else:

            return QtCore.QVariant()

    def setData(self, index, value, role=None):
        """
        This needs to be reimplemented if  allowing editing
        :param index:
        :param Any:
        :param role:
        :return:
        """

        if role != QtCore.Qt.EditRole:
            return False

        if not index.isValid():
            return False

        item = self.item_data[index.row()]
        item.val = value
        return True


    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        # print 'flags=',QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        existingFlags = super(TableModel, self).flags(index)
        # print 'existingFlags=',existingFlags
        # existingFlags|=QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

        # if index.column() == PROPERTY_NAME:
        #     existingFlags |= QtCore.Qt.NoItemFlags
        #
        # if index.column() == PROPERTY_VALUE:
        existingFlags |= QtCore.Qt.ItemIsEditable
        # return
        # return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable|Qt.ItemIsEditable

        return existingFlags
        # return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class TableView(QtWidgets.QTableView):
    """
    A simple table to demonstrate the QComboBox delegate.
    """

    def __init__(self, *args, **kwargs):
        QtWidgets.QTableView.__init__(self, *args, **kwargs)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:

            index = self.indexAt(event.pos())
            col_name = self.get_col_name(index)
            if  col_name == 'Value':
                self.edit(index)
        else:
            super(TableView,self).mousePressEvent(event)
        # QTableView.mousePressEvent(event)

    def get_col_name(self,index):

        model = index.model()
        return model.header_data[index.column()]

if __name__ == '__main__':
    item_data = []
    item_data.append(ItemData(name='vol', val=25, min=0, max=100, widget_name='slider'))
    item_data.append(ItemData(name='lam_vol', val=2.0, min=0, max=10.0, widget_name='slider' ))
    item_data.append(ItemData(name='sur', val=20.2))
    item_data.append(ItemData(name='lam_sur', val=20.2))

    import sys

    app = QApplication(sys.argv)  # needs to be defined first

    window = QWidget()
    layout = QHBoxLayout()

    # model = QStandardItemModel(4, 2)

    # cdf = get_data_frame()
    model = TableModel()
    model.update(item_data)
    # model.update_type_conv_fcn(get_types())

    tableView = TableView()
    tableView.setModel(model)

    delegate = EditorDelegate()
    tableView.setItemDelegate(delegate)

    layout.addWidget(tableView)
    window.setLayout(layout)
    tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    # window.setWindowTitle("Spin Box Delegate")
    # tableView.setWindowTitle("Spin Box Delegate")
    window.resize(QSize(800, 300))
    window.show()
    # tableView.show()
    sys.exit(app.exec_())
