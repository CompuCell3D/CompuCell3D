from PyQt5.QtCore import *
from PyQt5.QtGui import *

class LatticeDataModel(QAbstractTableModel):

    def __init__(self, parent=None):

        super(LatticeDataModel, self).__init__(parent)
        self.__headers = ["Lattice Data File Name"]
        self.__fileList = []

    def reset(self):
        """
        for backward compatibility I introduce simple reset fcn
        This fcn has to be called each time we change model
        @return: None
        """
        self.beginResetModel()
        self.endResetModel()

    def headerData(self, section, orientation, role):  # By default role == Qt.DisplayRole

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section <= len(self.__headers):
                return QVariant(self.__headers[section])

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return QVariant(section)

        return QVariant()

    def setLatticeDataFileList(self, _fileList):
        """
        This effectively initializes the model - here it is a simle list of vtk files
        @param _fileList:
        @return:
        """
        self.__fileList = _fileList

        # this is important - it informs qt that the model has changed and widget has to be redrawn
        self.reset()


    def rowCount(self, parentIndex=None):

        return len(self.__fileList)

    def columnCount(self, parentIndex):

        return len(self.__headers)

    def data(self, index, role):

        # Display data and tool tips
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        # Specify which data to display in each column!
        if index.column() == 0:  # (0, 1):

            return QVariant(self.__fileList[index.row()])

        return QVariant()
