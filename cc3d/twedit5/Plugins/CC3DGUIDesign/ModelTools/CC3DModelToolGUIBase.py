from PyQt5.QtCore import QObject, pyqtSignal, QEvent
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# Start-Of-Header

name = 'CC3DModelToolGUI'

author = 'T.J. Sego'

version = '0.0.0'

class_name = 'CC3DModelToolGUIBase'

module_type = 'Core'

short_description = 'Superclass for defining CC3D model tool GUIs'

long_description = """This superclass defines all requisite functionality for a tool GUI in the GUI Design plugin."""

# End-Of-Header


class CC3DModelToolGUIBase(QWidget):
    # Signals
    mtg_close_signal = pyqtSignal()
    mtg_enter_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(CC3DModelToolGUIBase, self).__init__(parent)

        self.__ked = KeyEventDetector(parent=self)
        self.installEventFilter(self.__ked)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.mtg_close_signal.emit()
        super(CC3DModelToolGUIBase, self).closeEvent(a0)

    def enterEvent(self, a0: QEvent) -> None:
        self.mtg_enter_signal.emit()
        super(CC3DModelToolGUIBase, self).enterEvent(a0)


class KeyEventDetector(QObject):
    def __init__(self, parent: CC3DModelToolGUIBase):
        super(KeyEventDetector, self).__init__(parent)
        self.tool = parent

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:

        return super(KeyEventDetector, self).eventFilter(a0, a1)
