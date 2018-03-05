import sip

sip.setapi('QString', 1)
sip.setapi('QVariant', 1)

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from fancy_slider import FancySlider
from fancy_combo import FancyCombo


class EditorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if not index.isValid():
            return

        # column_name = self.get_col_name_from_index(index)
        col_name = self.get_col_name(index)

        if col_name == 'Value':
            item = index.model().get_item(index)

            if item.widget_name == 'slider':
                editor = self.init_slider(parent, index)
            elif item.widget_name in ['pulldown', 'pull-down', 'combobox']:
                editor = self.init_combobox(parent, index)
            else:
                editor = QLineEdit(parent)

        else:
            return None

        return editor

    def init_slider(self, parent, index):
        """
        initializes slider based on the current value of the index
        :param index: {index}
        :return:{QSlider instance}
        """

        item = index.model().get_item(index)

        slider = FancySlider(parent)
        slider.setOrientation(Qt.Horizontal)

        item_type = item.item_type

        if item_type == type(1):  # checking for integer type
            item.decimal_precision = 0

        slider.set_decimal_precision(item.decimal_precision)

        if item.min is None:
            slider.setMinimum(0)
        else:
            slider.setMinimum(item.min)
        if item.max is None:
            slider.setMaximum(10 * item.val)
        else:
            slider.setMaximum(item.max)

        slider.set_default_behavior()
        slider.setValue(item.val)
        # slider.setRange(0,100)
        # slider.setTickInterval(10)
        # slider.setTickInterval(2)
        # slider.setMaximum(1000)

        return slider

    def init_combobox(self, parent, index):
        """
        initializes qcombobox based on the current value of the index and the user-provided provided enum options
        :param index: {index}
        :return:{QSlider instance}
        """

        item = index.model().get_item(index)

        c_box = FancyCombo(parent)
        enum_list = map(lambda x: str(x), item.enum)
        item_pos = enum_list.index(str(item.val))
        c_box.addItems(enum_list)
        c_box.setCurrentIndex(item_pos)

        item_type = item.item_type

        # item

        return c_box

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
        return model.item_data[index.row()].item_type

    def setEditorData(self, editor, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return
        if column_name == 'Value':
            value = index.model().data(index, Qt.DisplayRole)
            print 'i,j=', index.column(), index.row()
            print'val=', value
            # editor.setText(str(value.toInt()))
            if isinstance(editor, QLineEdit):
                editor.setText(str(value))
            elif isinstance(editor, QSlider):
                pass  # this is placeholder the real value is set elsewhere in init_slider
                # editor.setTickPosition(50) n
            elif isinstance(editor, QComboBox):
                pass  # this is placeholder the real value is set elsewhere in init_combobox
                # editor.setValue(str(value))

            else:
                raise ValueError('Editor has usupported type of {}'.format(type(editor)))
            # try:
            #     editor.setText(str(value))
            # except:
            #     editor.setTickPosition(50)
        else:
            return

    def setModelData(self, editor, model, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return

        if column_name == 'Value':

            item = index.model().get_item(index)
            type_conv_fcn = self.get_item_type(index)
            if item.widget_name == 'slider':

                try:
                    value = type_conv_fcn(editor.value())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return
            elif item.widget_name in ['combobox', 'pull-down']:

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
    def __init__(self, name=None, val=None, min_val=None, max_val=None, decimal_precision=3, enum=None,
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
            assert len(type_set) == 1, 'enum list elelements (together with initial value) must me of the same type. ' \
                                       'Instead I got the following types: {}'.format(','.join(map(lambda x: str(x),(type_set))))

            self._enum = enum
            if val is not None:
                try:
                    val_pos = map(lambda x: str(x), self.enum).index(str(self.val))
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


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__()

        self.item_data = None
        self.header_data = ['Value', 'Type']
        self.item_data_attr_name = {
            0: 'val',
            1: 'item_type'
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

    def get_item(self, index):
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

        elif role == Qt.ToolTipRole:
            i = index.row()
            j = index.column()
            item = self.item_data[i]
            return str(item.item_type)

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
            if col_name == 'Value':
                self.edit(index)
        else:
            super(TableView, self).mousePressEvent(event)
        # QTableView.mousePressEvent(event)

    def get_col_name(self, index):

        model = index.model()
        return model.header_data[index.column()]


if __name__ == '__main__':
    item_data = []
    item_data.append(ItemData(name='vol', val=25, min_val=0, max_val=100, widget_name='slider'))
    item_data.append(
        ItemData(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))
    item_data.append(
        ItemData(name='lam_vol_combo_float', val=2.0, enum=[1., 2., 3., 4.], widget_name='combobox'))

    item_data.append(
        ItemData(name='lam_vol_combo_int', val=2, enum=[1, 2, 3, 4], widget_name='combobox'))

    item_data.append(
        ItemData(name='lam_vol_combo', val='dupa2', enum=['dupa', 'dupa1', 'dupa2', 'dupa3'], widget_name='combobox'))

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
