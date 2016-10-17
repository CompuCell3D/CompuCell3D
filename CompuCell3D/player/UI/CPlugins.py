
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class CPlugins(QTableView):
   
    def __init__(self, parent, vm): 
        QTableView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # vm - viewmanager, instance of class TabView
        self.vm = vm
        #self.__resizeColumns()

    def setParams(self):
        """
        Sets the parameters if the QTableView when the model is set
        """
        if self.model() is None:
            return
        
        # assert self.model().
        # print self.model()
        for i in range(0, self.model().rowCount()):
            self.setRowHeight(i, 20)

        self.setColumnWidth(0, 130)
        #self.cplugins.setColumnWidth(1, 200)
        self.setAlternatingRowColors (True)
        self.horizontalHeader().setStretchLastSection(True)
        
        self.connect(self, SIGNAL("doubleClicked(const QModelIndex &)"), self.__showPluginView)
        
    def __showPluginView(self, idx):
        # First, get the row number based on the idx (instance of QModelIndex)
        idx0 = idx.sibling(idx.row(), 0)
        idx1 = idx.sibling(idx.row(), 1)
        pluginInfo = [self.model().data(idx0, Qt.DisplayRole).toString(), self.model().data(idx1, Qt.DisplayRole).toString()]
        self.vm.showPluginView(pluginInfo)
        
        """   
        def populateTable(self):   
          self.setColumnCount(2)
          self.setRowCount(15)
          for i in range(0, self.rowCount()):
             self.setRowHeight(i, 20)
          self.setHorizontalHeaderLabels(["Name", "Description"])
          self.setFrameStyle(QFrame.NoFrame)
          self.setColumnWidth(0, 100)
          self.setColumnWidth(1, 120)
        """      
