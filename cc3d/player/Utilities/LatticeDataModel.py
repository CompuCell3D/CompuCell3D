
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class LatticeDataModel(QAbstractTableModel):

    def __init__(self, parent=None):
        #Constructor
        super(LatticeDataModel, self).__init__(parent)
        self.__headers = ["Lattice Data File Name"]        
        # self.__filename = ""
        self.__fileList=[]    
        # self.viewManager=_viewManager
        # self.__plugins = {}
        # self.loadData()

    def headerData(self, section, orientation, role): # By default role == Qt.DisplayRole
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section <= len(self.__headers):
                return QVariant(self.__headers[section])
            
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return  QVariant(section)
         
        return QVariant()
    
    def setLatticeDataFileList(self,_fileList):
        self.__fileList=_fileList
        
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
        if index.column() == 0:# (0, 1):
            # return QVariant(self.__plugins[index.row()][0])
            
            return QVariant(self.__fileList[index.row()])
        
        # elif index.column() == 1:
            # return QVariant(self.__plugins[index.row()][1])
            
        return QVariant()
    
    # def loadData(self):
        # i = 0
        # try:
            # file = open(self.__filename, "r")
            # for line in file:
                # plugin = line.splitlines()[0].split(": ") # Suppose that the line in the file looks like -- name: description
                # if plugin[0] != '':
                    # self.__plugins[i] = plugin
                # i=i+1
            # self.__plugins.keys().sort()
            # file.close()
        # except IOError:
            # print "Cannot open the file: %s" % self.__filename
        
    #def 
