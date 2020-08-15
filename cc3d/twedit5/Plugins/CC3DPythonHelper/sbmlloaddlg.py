from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_sbmlloaddlg
import os

MAC = "qt_mac_set_native_menubar" in dir()


class SBMLLoadDlg(QDialog, ui_sbmlloaddlg.Ui_SBMLLoadDlg):

    def __init__(self, _currentEditor=None, parent=None):
        super(SBMLLoadDlg, self).__init__(parent)

        self.editorWindow = parent

        self.setupUi(self)

        self.sbmlPath = ''

        if not MAC:
            self.leaveEmptyPB.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

    def setCurrentPath(self, _sbmlPath):

        self.sbmlPath = _sbmlPath

    @pyqtSlot()
    def on_browsePB_clicked(self):

        filter_list = ''
        filter_list += ("SBML file (*.sbml *.xml);;")
        filter_list += ("All files (*);;")

        file_name = QFileDialog.getOpenFileName(self, "Open SBML file...", self.sbmlPath, filter_list)
        file_name = str(file_name)

        if file_name == '':
            return

        file_name = os.path.abspath(file_name)

        os.path.dirname(file_name)

        model_name, extension = os.path.splitext(os.path.basename(file_name))

        model_nickname = model_name[0:3].upper() if len(model_name) > 3 else model_name.upper()

        self.modelNameLE.setText(model_name)

        self.modelNicknameLE.setText(model_nickname)

        self.fileNameLE.setText(file_name)

    def updateUi(self):

        pass
