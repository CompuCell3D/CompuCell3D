# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:50:56 2013

@author: cmarshall
"""

import sip

sip.setapi('QString', 1)
sip.setapi('QVariant', 1)

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class EditorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if not index.isValid():
            return

        # column_name = self.get_col_name_from_index(index)
        col_name = self.get_col_name(index)
        if col_name == 'Value':
            editor = QLineEdit(parent)

        else:
            return None

        return editor

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
            editor.setText(str(value))
        else:
            return

    def setModelData(self, editor, model, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return

        if column_name == 'Value':

            type_conv_fcn = self.get_item_type(index)
            print 'type_conv_fcn=', type_conv_fcn
            try:
                value = type_conv_fcn(editor.text())
            except ValueError as exc:
                QMessageBox.warning(None,'Type Conversion Error',str(exc))
                return
        else:
            return
            # editor.interpretText()
            # value = editor.value()

        model.setData(index, value, Qt.EditRole)


    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ItemData(object):
    def __init__(self, name=None, val=None):
        self._name = name
        self._val = None
        if val is not None:
            self.val = val

        self._type = None
        self._min = None
        self._max = None
        self._enum = None

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


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(TableModel, self).__init__()

        self.item_data = None
        self.header_data = ['Name', 'Value', 'Type']
        self.item_data_attr_name = {
            0: 'name',
            1: 'val',
            2: 'type',

        }

    def update(self, item_data):

        self.item_data = item_data

    # def update_type_conv_fcn(self, type_conv_fcn_data):
    #     self.type_conv_fcn_data = type_conv_fcn_data

    def headerData(self, p_int, orientation, role=None):

        # if orientation == Qt.Horizontal and role == Qt.DisplayRole:
        #     return self.datatable.columns[p_int]
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            try:
                return self.header_data[p_int]
            except IndexError:
                return QVariant()

        return QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.item_data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.header_data)


    def data(self, index, role=QtCore.Qt.DisplayRole):
        # print 'Data Call'
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            item = self.item_data[i]
            item_data_to_display = getattr(item, self.item_data_attr_name[j])
            return '{}'.format(item_data_to_display)

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
            # return self.rootItem
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

if __name__ == '__main__':
    item_data = []
    item_data.append(ItemData(name='vol', val=25))
    item_data.append(ItemData(name='lam_vol', val=2.0))
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

    tableView = QTableView()
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
