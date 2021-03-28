from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from collections import OrderedDict
from cc3d import CompuCellSetup
from cc3d.player5 import Configuration


class CellTypeColorMapModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(CellTypeColorMapModel, self).__init__()

        self.item_data = None
        self.dirty_flag = False

        self.type_name_idx_in_list = 0
        self.color_idx_in_list = 1
        self.type_id_idx = 0
        self.color_idx = 1
        self.type_name_idx = 2

        self.header_data = [
            'Cell Type Id',
            'Color',
            'Name'
            # 'Type'
        ]
        self.item_data_attr_name = {
            0: 'val',
            # 1: 'item_type'
        }

    def read_cell_type_color_data(self):
        """

        :return:
        """
        self.item_data = OrderedDict()

        type_color_map = Configuration.getSetting('TypeColorMap')
        names_ids = CompuCellSetup.simulation_utils.extract_type_names_and_ids()
        for type_id, type_name in names_ids.items():

            try:
                color = type_color_map[type_id]
            except KeyError:
                color = QColor('black')

            self.item_data[type_id] = [type_name, color]


        self.update(item_data=self.item_data)

    def set_dirty(self, flag):
        self.dirty_flag = flag

    def is_dirty(self):
        return self.dirty_flag

    def update(self, item_data):

        self.item_data = item_data

    def headerData(self, p_int, orientation, role=None):

        if self.item_data is None or not len(self.item_data):
            return

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            try:
                return self.header_data[p_int]
            except IndexError:
                return QVariant()

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            try:
                # return self.item_data[p_int].name
                return self.item_data[p_int]
            except KeyError:
                return QVariant()

        return QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if self.item_data is None:
            return 0
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
            # item is a list of [type name, color]
            item = self.item_data[i]
            if j == self.type_id_idx:
                return list(self.item_data.keys())[i]
            elif j == self.type_name_idx:
                return item[self.type_name_idx_in_list]
            else:
                return QVariant()
            # item_data_to_display = getattr(item, self.item_data_attr_name[j])
            # return '{}'.format(item_data_to_display)

        elif role == Qt.BackgroundRole:
            i = index.row()
            j = index.column()

            if j == self.color_idx:
                # item is a list of [type name, color]
                item = self.item_data[i]
                color = item[self.color_idx_in_list]
                return color
            return QVariant()

        elif role == Qt.ToolTipRole:
            i = index.row()
            j = index.column()
            item = self.item_data[i]

            return str(item[self.type_name_idx_in_list])

        else:
            return QVariant()

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
        existing_flags = super(CellTypeColorMapModel, self).flags(index)

        existing_flags |= QtCore.Qt.ItemIsEditable

        return existing_flags


