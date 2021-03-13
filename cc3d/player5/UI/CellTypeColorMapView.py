
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

# self.steering_model = SteeringPanelModel()
# self.steering_model.update(item_data)
# # model.update_type_conv_fcn(get_types())
#
# self.steering_table_view = SteeringPanelView()
# self.steering_table_view.setModel(self.steering_model)
#
# delegate = SteeringEditorDelegate()
# self.steering_table_view.setItemDelegate(delegate)
#
# layout.addWidget(self.steering_table_view)
# self.steering_window.setLayout(layout)
# self.steering_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


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


