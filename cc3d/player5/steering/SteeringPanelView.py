from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from cc3d import CompuCellSetup

class SteeringPanelView(QtWidgets.QTableView):
    def __init__(self, *args, **kwargs):

        QtWidgets.QTableView.__init__(self, *args, **kwargs)

    def mousePressEvent(self, event):

        pg = CompuCellSetup.persistent_globals
        if pg.steering_panel_synchronizer.locked():
            return

        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            col_name = self.get_col_name(index)
            if col_name == 'Value':
                self.edit(index)
        else:
            super(SteeringPanelView, self).mousePressEvent(event)
            # QTableView.mousePressEvent(event)

    def get_col_name(self, index):

        model = index.model()

        if not model:
            return None

        return model.header_data[index.column()]

    def sizeHint(self):
        """
        Implements basic size hint
        :return: {Qsize}
        """
        row_count = min(self.model().rowCount() + 1, 20)
        row_size = self.rowHeight(0)

        sugested_vsize = row_count * row_size
        return QSize(600, sugested_vsize)
