
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtXml import *
from PyQt5.QtWidgets import *
#from Utilities.XMLHandler import DomModel

class ModelEditor(QTreeView):
   
    def __init__(self, parent):
        QTreeView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.parent = parent
        
    def getParent(self):
        return self.parent

    def setParams(self):
        # Column widths should be set after setting the model!
        # Fixme: Before setting the column sizes, make sure that 
        # the number of columns is equal to 2!
        self.setColumnWidth(0, 180) # Since Qt 4.2  
        self.setColumnWidth(1, 40)
        self.header().setDefaultAlignment(Qt.AlignHCenter)
        # self.expandToDepth(0)

        """
        modelEditor.setColumnWidth(0, 180) # Since Qt 4.2  
        modelEditor.setColumnWidth(1, 40)
        modelEditor.header().setDefaultAlignment(Qt.AlignHCenter)
        modelEditor.expandToDepth(1)
        """
        #self.header().resizeSections(QHeaderView.ResizeToContents)
        #self.header().setStretchLastSection(True)
      
        """
      headers = QStringList()
      headers << self.trUtf8("Parameter") << self.trUtf8("Value")
      model = QStandardItemModel()
      model.setHorizontalHeaderLabels(headers)
      self.setModel(model)
      self.setColumnWidth(0, 180)
      selectionModel = QItemSelectionModel(model)
      self.setSelectionModel(selectionModel)
        """