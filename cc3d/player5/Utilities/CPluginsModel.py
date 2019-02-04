from PyQt5.QtCore import *
from PyQt5.QtGui import *

class CPluginsModel(QAbstractTableModel):

    # Constructor
    def __init__(self, filename, parent=None):

        super(CPluginsModel, self).__init__(parent)
        self.__headers = ["Name", "Description"]
        self.__filename = filename
        self.__plugins = {}
        self.loadData()

    # By default role == Qt.DisplayRole
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section <= len(self.__headers):
                return QVariant(self.__headers[section])
            
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return QVariant(section)
         
        return QVariant()
    
    def rowCount(self, parentIndex=None):
        return len(self.__plugins)
    
    def columnCount(self, parentIndex):
        return len(self.__headers)
    
    def data(self, index, role):
        # Display data and tool tips
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return QVariant()
        
        if not index.isValid():
            return QVariant()

        # Specify which data to display in each column!
        if index.column() == 0:# (0, 1):
            return QVariant(self.__plugins[index.row()][0])
        elif index.column() == 1:
            return QVariant(self.__plugins[index.row()][1])
            
        return QVariant()
    
    def loadData(self):
        i = 0
        try:
            file = open(self.__filename, "r")
            for line in file:
                plugin = line.splitlines()[0].split(": ") # Suppose that the line in the file looks like -- name: description
                if plugin[0] != '':
                    self.__plugins[i] = plugin
                i=i+1
            list(self.__plugins.keys()).sort()
            file.close()
        except IOError:
            print("Cannot open the file: %s" % self.__filename)

