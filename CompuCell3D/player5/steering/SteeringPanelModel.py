from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class SteeringPanelModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(SteeringPanelModel, self).__init__()

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
        existingFlags = super(SteeringPanelModel, self).flags(index)
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
