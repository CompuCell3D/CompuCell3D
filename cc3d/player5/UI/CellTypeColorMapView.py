from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from .checkbox_delegate import CheckBoxDelegate
from .color_delegate import ColorDelegate


class CellTypeColorMapView(QTableView):
   
    def __init__(self, parent, vm): 
        QTableView.__init__(self, parent)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)

        # note - delegates need to be member of a class to avoid them being garbage-collected. This would cause
        # cryptic crashes
        self.check_box_delegate = CheckBoxDelegate(None)
        self.setItemDelegateForColumn(3, self.check_box_delegate)

        self.color_delegate = ColorDelegate(None)
        self.setItemDelegateForColumn(1, self.color_delegate)

        # on OSX we do not resize row height, we do it only on windows and linux
        if not sys.platform.startswith('darwin'):
            pass
            # verticalHeader = self.verticalHeader()
            # verticalHeader.setResizeMode(QHeaderView.Fixed)
            # verticalHeader.setDefaultSectionSize(20)
            
        # vm - viewmanager, instance of class SimpleTabView
        self.vm = vm

    def update_content(self):

        model = self.model()
        if model is None:
            return

        model.beginResetModel()
        model.read_cell_type_color_data()
        model.endResetModel()

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            model = index.model()
            if index.column() == model.color_idx:
                self.edit(index)
        else:
            super(CellTypeColorMapView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            model = index.model()
            if index.column() == model.show_in_3d_idx:
                # somewhat hacky solution - editor event checks also for mouseRelease event so we need to
                # reimplement mouseReleaseEvent
                option = self.viewOptions()
                self.check_box_delegate.editorEvent(event=event, model=model, option=option, index=index)
        else:
            super(CellTypeColorMapView, self).mouseReleaseEvent(event)
