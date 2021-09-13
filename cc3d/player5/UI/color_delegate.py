from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QModelIndex
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QApplication, QTableView


class ColorDelegate(QtWidgets.QItemDelegate):
    """
    A delegate that places a fully functioning QCheckBox cell of the column to which it's applied.
    """
    def __init__(self, parent):
        QtWidgets.QItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        """
        Important, otherwise an editor is created if the user clicks in this cell.
        """

        editor = self.init_color_dialog(parent, index)

        return editor

    def init_color_dialog(self, parent, index):
        """
        initializes color dialog before we show it to user
        :param parent:
        :param index:
        :return:
        """
        dlg = QtWidgets.QColorDialog(parent)
        current_color = self.get_color_at_index(index=index)
        dlg.setCurrentColor(current_color)
        return dlg

    def get_color_at_index(self, index):
        """
        returns qcolor object for a given index in the model
        :param index:
        :return:
        """
        model = index.model()

        row = index.row()
        color_idx = model.color_idx_in_list
        color_at_index = model.item_data[row][color_idx]

        return color_at_index

    def paint(self, painter, option, index):
        """
        Paint a checkbox without the label.
        """
        background = self.get_color_at_index(index=index)
        self.drawBackground(painter, option, index)
        painter.fillRect(option.rect, background)

    # note: editorEvent is not called here because in the view mousePressEvent we explicitly call
    # "edit" function that will trigger createEditor function and then after use closes dialog
    # we will be in the setModelData function

    def setModelData(self, editor, model, index):
        """
        After user closes Color dialog this method gets called and we can modify model settings here
        :param editor: color dialog instance
        :param model: model
        :param index: index which triggered opening of the color dialog
        :return:
        """
        chosen_color = editor.selectedColor()
        if chosen_color.isValid():
            model.setData(index, chosen_color, QtCore.Qt.EditRole)
