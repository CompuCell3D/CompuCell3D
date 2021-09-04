
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from .checkbox_delegate import CheckBoxDelegate


class CellTypeColorMapView(QTableView):
   
    def __init__(self, parent, vm): 
        QTableView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setColumnWidth(0, 100)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        delegate = CheckBoxDelegate(None)
        self.setItemDelegateForColumn(3, delegate)


        # on OSX we do not resize row height, we do it only on windows and linux
        if not sys.platform.startswith('darwin'):
            pass
            # verticalHeader = self.verticalHeader()
            # verticalHeader.setResizeMode(QHeaderView.Fixed)
            # verticalHeader.setDefaultSectionSize(20)
            
        # vm - viewmanager, instance of class TabView
        self.vm = vm
        #self.__resizeColumns()

    def update_content(self):

        model = self.model()
        if model is None:
            return

        model.beginResetModel()
        model.read_cell_type_color_data()
        model.endResetModel()


    # def setParams(self):
    #     """
    #     Sets the parameters if the QTableView when the model is set
    #     """
    #     if self.model() is None:
    #         return
    #
    #     # assert self.model().
    #     # print self.model()
    #     # import sys
    #     # if not sys.platform.startswith('darwin'): # on OSX we do not resize row height
    #         # for i in range(0, self.model().rowCount()):
    #             # self.setRowHeight(i, 20)
    #
    #     self.setColumnWidth(0, 130)
    #     #self.cplugins.setColumnWidth(1, 200)
    #     self.setAlternatingRowColors(True)
    #     self.horizontalHeader().setStretchLastSection(True)


