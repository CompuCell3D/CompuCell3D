from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_gotolinedlg
import sys
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, dbgMsg

MAC = "qt_mac_set_native_menubar" in dir()


class GoToLineDlg(QDialog, ui_gotolinedlg.Ui_GoToLineDlg):
    gotolineSignal = pyqtSignal(int)

    def __init__(self, _currentEditor=None, parent=None):

        super(GoToLineDlg, self).__init__(parent)

        self.editorWindow = parent

        self.currentEditor = _currentEditor

        self.gotolineSignal.connect(self.editorWindow.goToLine)

        self.setupUi(self)

        # ensuring that only integers greater than 0 can be entered

        self.intValidator = QIntValidator(self.goToLineEdit)

        self.intValidator.setBottom(1)

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        # print dir(self.currentEditor)

        self.intValidator.setTop(self.currentEditor.lines())

        self.goToLineEdit.setValidator(self.intValidator)

        if not MAC:
            self.closeButton.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

    @pyqtSlot()  # signature of the signal emited by the button
    def on_goButton_clicked(self):
        """

        :return:
        """

        # print "this is on go button clicked=",self.goToLineEdit.text()

        line_num = int(self.goToLineEdit.text())

        if line_num:
            self.gotolineSignal.emit(line_num)

        # we close the dialog right after user hits Go button  . If the entry is invalid no action is trigerred  

        self.close()

        return

    def updateUi(self):
        """

        :return:
        """


