from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QModelIndex
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QApplication, QTableView


class ColorDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        dialog = QtWidgets.QColorDialog(parent)
        return dialog

    def setEditorData(self, editor, index):
        color = index.data(QtCore.Qt.BackgroundRole)
        editor.setCurrentColor(color)

    def setModelData(self, editor, model, index):
        color = editor.currentColor()
        model.setData(index, color, QtCore.Qt.BackgroundRole)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    model = QStandardItemModel(4, 4)
    for i in range(model.rowCount()):
        for j in range(model.columnCount()):
            color = QtGui.QColor(QtCore.qrand() % 256, QtCore.qrand() % 256, QtCore.qrand() % 256)
            it = QtGui.QStandardItem("{}{}".format(i, j))
            it.setData(color, QtCore.Qt.BackgroundRole)
            model.setItem(i, j, it)
    w = QTableView()
    w.setModel(model)
    # w.setItemDelegate(ColorDelegate())
    w.show()
    sys.exit(app.exec_())