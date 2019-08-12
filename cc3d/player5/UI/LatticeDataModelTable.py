
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

MODULENAME = '---- UI/LatticeDataModelTable.py: '

class LatticeDataModelTable(QTableView):
   
    def __init__(self, parent, vm): 
        QTableView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setColumnWidth(0, 100)
        self.setAlternatingRowColors (True)
        self.horizontalHeader().setStretchLastSection(True)        
        
        import sys
        if not sys.platform.startswith('darwin'): # on OSX we do not resize row height, we do it only on windows and linux
            pass
            # verticalHeader = self.verticalHeader()
            # verticalHeader.setResizeMode(QHeaderView.Fixed)
            # verticalHeader.setDefaultSectionSize(20)
            
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
        # import sys
        # if not sys.platform.startswith('darwin'): # on OSX we do not resize row height
            # for i in range(0, self.model().rowCount()):
                # self.setRowHeight(i, 20)

        self.setColumnWidth(0, 130)
        #self.cplugins.setColumnWidth(1, 200)
        self.setAlternatingRowColors (True)
        self.horizontalHeader().setStretchLastSection(True)

        self.clicked.connect(self.__pickAndAdvanceMCSDirect)

        # self.connect(self, SIGNAL("clicked(const QModelIndex &)"), self.__pickAndAdvanceMCSDirect)

        # self.connect(self, SIGNAL("clicked(const QModelIndex &)"), self.__pickMCSDirect)        
        # self.connect(self, SIGNAL("doubleClicked(const QModelIndex &)"), self.__pickAndAdvanceMCSDirect)
        
    def prepareToClose(self):
        self.clicked.disconnect(self.__pickAndAdvanceMCSDirect)
        # self.disconnect(self, SIGNAL("clicked(const QModelIndex &)"), self.__pickAndAdvanceMCSDirect)

        # self.disconnect(self, SIGNAL("clicked(const QModelIndex &)"), self.__pickMCSDirect)
        # self.disconnect(self, SIGNAL("doubleClicked(const QModelIndex &)"), self.__pickAndAdvanceMCSDirect)
        
    def __pickAndAdvanceMCSDirect(self,idx):
        # print '__pickAndAdvanceMCSDirect idx=',idx
        # print 'self.vm.simulation=',self.vm.simulation
        self.vm.simulation.set_current_step_direct_access(idx.row())
        self.vm.step_act.trigger()
    
    def __pickMCSDirect(self, idx):
        # First, get the row number based on the idx (instance of QModelIndex)
        
        # idx0 = idx.sibling(idx.row(), 0)
        # print MODULENAME,"__pickMCSDirect():  calling self.vm.simulation.setCurrentStepDirectAccess(idx.row())= ",idx.row()
        self.vm.simulation.set_current_step_direct_access(idx.row())
        # self.vm.stepAct.trigger()
        # idx1 = idx.sibling(idx.row(), 1)
        # pluginInfo = [self.model().data(idx0, Qt.DisplayRole).toString(), self.model().data(idx1, Qt.DisplayRole).toString()]
        # self.vm.showLatticeDataModelView(pluginInfo)
        
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
