from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *

class SteeringPanelModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(SteeringPanelModel, self).__init__()

        self.item_data = None
        self.dirty_flag = False

        self.header_data = [
            'Value',
            # 'Type'
        ]
        self.item_data_attr_name = {
            0: 'val',
            # 1: 'item_type'
        }

    def set_dirty(self, flag):
        self.dirty_flag = flag

    def is_dirty(self):
        return self.dirty_flag

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

            else:
                return QtGui.QColor('gray')

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
        item.dirty_flag = True
        self.dirty_flag = True
        return True

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        existingFlags = super(SteeringPanelModel, self).flags(index)

        existingFlags |= QtCore.Qt.ItemIsEditable

        return existingFlags


