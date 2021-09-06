
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from .checkbox_delegate import CheckBoxDelegate, ColorEditorDelegate
from .qcolor_delegate import ColorDelegate


class CellTypeColorMapView(QTableView):
   
    def __init__(self, parent, vm): 
        QTableView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        # self.setColumnWidth(0, 30)
        # self.setColumnWidth(1, 50)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)

        self.check_box_delegate = CheckBoxDelegate(None)
        self.setItemDelegateForColumn(3, self.check_box_delegate)

        # color_delegate = ColorDelegate()

        self.color_delegate = ColorEditorDelegate(None)
        self.setItemDelegateForColumn(1, self.color_delegate)


        # self.setColumnWidth(0, 30)
        # self.setColumnWidth(1, 50)



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


    def mousePressEvent(self, event):

        # pg = CompuCellSetup.persistent_globals
        # if pg.steering_panel_synchronizer.locked():
        #     return

        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            model = index.model()
            if index.column() == model.color_idx:
                self.edit(index)
            # if index.column() == model.show_in_3d_idx:
            #     print('trying to edit show in 3d')
            #     self.check_box_delegate.editorEvent(event=event, model=model, option=None, index=index)
            #     # event.ignore()
            #     # self.edit(index)


        else:
            super(CellTypeColorMapView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            model = index.model()
            if index.column() == model.show_in_3d_idx:
                # somewhat hacky solution - editor event checks also for mouseRelease event so we need to
                # reimplement mouseReleaseEvent
                self.check_box_delegate.editorEvent(event=event, model=model, option=None, index=index)
        else:
            super(CellTypeColorMapView, self).mouseReleaseEvent(event)


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


