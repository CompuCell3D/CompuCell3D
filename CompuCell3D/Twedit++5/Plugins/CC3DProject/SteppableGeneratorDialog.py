import re
from utils.global_imports import *
import ui_steppablegeneratordialog
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()


class SteppableGeneratorDialog(QDialog, ui_steppablegeneratordialog.Ui_SteppableGenerator):
    # signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))

    # _cc3dProject=None
    def __init__(self, parent=None):
        super(SteppableGeneratorDialog, self).__init__(parent)

        self.cc3dProjectTreeWidget = parent

        # self.cc3dProject=_cc3dProject

        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists
        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.projectPath = ""
        self.setupUi(self)

    @pyqtSlot()
    def on_okPB_clicked(self):
        if str(self.steppebleNameLE.text()).strip() == "":
            QMessageBox.warning(self, "Empty Steppable Name", "Please specify steppable name")
        else:
            self.accept()
