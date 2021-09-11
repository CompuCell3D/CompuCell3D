from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from collections import OrderedDict
from cc3d import CompuCellSetup
from cc3d.player5 import Configuration
import warnings
import weakref


class CellTypeColorMapModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, *args):
        super(CellTypeColorMapModel, self).__init__()

        self.item_data = None
        self.dirty_flag = False

        # those indexes label where a given item is in the list that describes particular row
        self.type_id_idx_in_list = -1
        self.type_name_idx_in_list = 0
        self.color_idx_in_list = 1
        self.show_in_3d_idx_in_list = 2

        # those are view column indexes
        self.type_id_idx = 0
        self.color_idx = 1
        self.type_name_idx = 2
        self.show_in_3d_idx = 3

        self.header_data = [
            'Cell Type Id',
            'Color',
            'Name',
            'Show in 3D'
        ]
        self.item_data_attr_name = {
            0: 'val',
        }
        self.vm = None

    def set_view_manager(self, vm):
        self.vm = weakref.ref(vm)

    def read_cell_type_color_data(self):
        """

        :return:
        """

        types_invisible_str = str(Configuration.getSetting('Types3DInvisible'))

        types_invisible = types_invisible_str.replace(" ", "")
        types_invisible = types_invisible.split(",")
        types_invisible_dict = {int(type_id): 1 for type_id in types_invisible}

        self.item_data = OrderedDict()

        type_color_map = Configuration.getSetting('TypeColorMap')
        names_ids = CompuCellSetup.simulation_utils.extract_type_names_and_ids()
        for type_id, type_name in names_ids.items():

            try:
                color = type_color_map[type_id]
            except KeyError:
                color = QColor('black')

            try:
                types_invisible_dict[type_id]
                show_in_3d = 0
            except KeyError:
                show_in_3d = 1

            self.item_data[type_id] = [type_name, color, show_in_3d, type_id]

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
            if j == self.show_in_3d_idx:
                return item[self.show_in_3d_idx_in_list]
            else:
                return QVariant()

        # elif role == Qt.BackgroundRole:
        # background for color column is handled via delegate

        elif role == Qt.ToolTipRole:
            i = index.row()
            item = self.item_data[i]

            return str(item[self.type_name_idx_in_list])

        else:
            return QVariant()

    def setData(self, index, value, role=None):
        """
        This needs to be reimplemented if  allowing editing
        :param index:
        :param value:
        :param role:
        :return:
        """

        if role != QtCore.Qt.EditRole:
            return False

        if not index.isValid():
            return False

        modification_made = False
        row_item = self.item_data[index.row()]
        if index.column() == self.show_in_3d_idx and isinstance(value, int):
            row_item[self.show_in_3d_idx_in_list] = value
            self.store_invisible_types_in_settings()
            modification_made = True
        elif index.column() == self.color_idx:
            row_item[self.color_idx_in_list] = value
            self.store_updated_color_type_map_in_settings(row_item)
            modification_made = True
        else:
            print('not Implemented')

        if modification_made:
            self.vm().trigger_configs_changed()
            self.vm().redo_completed_step()

        self.dirty_flag = True
        return True

    def store_invisible_types_in_settings(self):
        """
        updates settings  - stores updated 3d invisible types
        :return:
        """
        invisible_types = []
        for type_id, item_list in self.item_data.items():
            if not item_list[self.show_in_3d_idx_in_list]:
                invisible_types.append(str(type_id))
        Configuration.setSetting('Types3DInvisible', ','.join(invisible_types))

    def store_updated_color_type_map_in_settings(self, row_item: list):
        """
        updates settings - stores updated color map
        :param row_item: model row item - in this context this is a list of type id, color, show_in_3D_flag
        :return:
        """

        type_color_map = Configuration.getSetting('TypeColorMap')
        try:
            type_id_for_row_item = row_item[self.type_id_idx_in_list]
            color_for_row_item = row_item[self.color_idx_in_list]
            type_color_map[type_id_for_row_item] = color_for_row_item
        except (LookupError, IndexError) as e:
            warnings.warn(f'Could not find entry for TypeColorMap setting {row_item} . Error: {e}')

        Configuration.setSetting('TypeColorMap', type_color_map)

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        existing_flags = super(CellTypeColorMapModel, self).flags(index)

        existing_flags |= QtCore.Qt.ItemIsEditable

        if index.column() == self.color_idx:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

        return existing_flags
