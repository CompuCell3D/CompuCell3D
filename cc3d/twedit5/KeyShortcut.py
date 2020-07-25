from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_KeyShortcut
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, dbgMsg

MAC = "qt_mac_set_native_menubar" in dir()


class KeyShortcutDlg(QDialog, ui_KeyShortcut.Ui_KeyShortcutDlg):

    def __init__(self, parent=None, _title='', _initialText=''):

        super(KeyShortcutDlg, self).__init__(parent)

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        self.setupUi(self)

        self.sequence = None

        self.recording = False

        self.keySequence = QKeySequence()

        self.nonShiftModifierPreseed = False

        # self.connect(self.grabButton,SIGNAL("clicked()"), self.startRecording)

        self.grabButton.clicked.connect(self.startRecording)

        self.setInitialShortcutText(_initialText)

        self.setTitle(_title)

    def setInitialShortcutText(self, _text):

        self.keyLabel.setText(_text)

    def setTitle(self, _text):

        self.setWindowTitle(_text)

    def getKeySequence(self):

        return QKeySequence(self.keyLabel.text())

    def updateShortcutDisplay(self):

        # s=self.keySequence.toString(QKeySequence.NativeText)

        s = ''

        dbgMsg("key=", self.key)
        dbgMsg("shiftPeessed=%x" % (Qt.SHIFT & self.key))
        dbgMsg("Qt.SHIFT = %x" % (Qt.SHIFT & ~Qt.SHIFT))
        dbgMsg("Qt.SHIFT = %x" % ~(Qt.SHIFT + 1))

        if self.modifiers:
            dbgMsg("GOT MODIFIERS")
            if self.modifiers & (Qt.CTRL):
                s += "Ctrl+"

            if self.modifiers & (Qt.SHIFT):
                s += "Shift+"

            if self.modifiers & (Qt.ALT):
                s += "Alt+"

            if self.modifiers & (Qt.META):
                s += "Meta+"

            # pressing non modifier key ends recording    

            if self.key != Qt.Key_Shift and self.key != Qt.Key_Control and \
                    self.key != Qt.Key_Alt and self.key != Qt.Key_Meta:
                # dbgMsg("REGULAR KEY=", QChar(self.key).toAscii())

                self.doneRecording()

        if not self.sequence:
            self.keyLabel.setText(s)

        else:

            self.keyLabel.setText(self.sequence.toString(QKeySequence.NativeText))

            # elif self.key:

            # s.append(str(self.key))

            # dbgMsg("DONE RECORDING")

            # self.doneRecording()

    def keyPressEvent(self, e):

        dbgMsg("keyPressEvent")

        if e.key() == -1:
            self.cancelRecording()

        e.accept()

        newModifiers = e.modifiers() & (Qt.SHIFT | Qt.CTRL | Qt.ALT | Qt.META)

        dbgMsg("newModifiers=", newModifiers)

        # if newModifiers and not self.recording and not self.grabKey.isEnabled():

        # self.startRecording()

        dbgMsg("self.recording=", self.recording)

        if not self.recording:
            return

            # check if non-SHIFT modifier has been presed - this affects whether we can use shift in the shortcut or not

        # e.g. SHIFT with a letter is not a valid shortcut but if there
        # is additional modifier pressed than is it valid e.g. Ctrl+Shift+F

        if newModifiers & (Qt.CTRL | Qt.ALT | Qt.META):
            self.nonShiftModifierPreseed = True

        self.key = e.key()

        self.modifiers = int(newModifiers)

        if self.key == Qt.Key_AltGr:  # or else we get unicode salad

            return

        elif self.key == Qt.Key_Shift:

            self.updateShortcutDisplay()

        elif self.key == Qt.Key_Control:

            self.updateShortcutDisplay()

        elif self.key == Qt.Key_Alt:

            self.updateShortcutDisplay()

        elif self.key == Qt.Key_Meta:

            self.updateShortcutDisplay()

        else:

            if self.modifiers & (Qt.SHIFT | Qt.CTRL | Qt.ALT | Qt.META):  # check if any of the modifiers is chc

                if self.isShiftAsModifierAllowed(self.key):

                    self.key |= (self.modifiers)

                else:  # filter out shift

                    self.key |= (self.modifiers & ~Qt.SHIFT)

                self.sequence = QKeySequence(self.key)

                dbgMsg("\t\t\t self.sequence=", self.sequence.toString())

                dbgMsg("self.modifiers=%x" % self.modifiers)

                self.updateShortcutDisplay()

                dbgMsg("GOT THIS KEY", self.key)

            else:  # pressing non-modifier key but none of modifier keys are pressed - not a valid shortcut

                self.cancelRecording()

        dbgMsg("END OF KEY PRESS EVENT")

    def isShiftAsModifierAllowed(self, key):

        if self.nonShiftModifierPreseed:
            return True

        if key >= Qt.Key_F1 and key <= Qt.Key_F35:
            return True

        if key == Qt.Key_Return:

            return True

        elif key == Qt.Key_Space:

            return True

        elif key == Qt.Key_Backspace:

            return True

        elif key == Qt.Key_Escape:

            return True

        elif key == Qt.Key_Print:

            return True

        elif key == Qt.Key_ScrollLock:

            return True

        elif key == Qt.Key_Pause:

            return True

        elif key == Qt.Key_PageUp:

            return True

        elif key == Qt.Key_PageDown:

            return True

        elif key == Qt.Key_Insert:

            return True

        elif key == Qt.Key_Delete:

            return True

        elif key == Qt.Key_Home:

            return True

        elif key == Qt.Key_End:

            return True

        elif key == Qt.Key_Up:

            return True

        elif key == Qt.Key_Down:

            return True

        elif key == Qt.Key_Left:

            return True

        elif key == Qt.Key_Right:

            return True

        else:

            return False

    def keyReleaseEvent(self, e):

        dbgMsg("keyReleaseEvent")

        # self.doneRecording()

    def startRecording(self):

        dbgMsg("start recording")

        self.grabKeyboard()

        self.keyLabel.setText('')

        self.nonShiftModifierPreseed = False

        self.grabButton.setEnabled(False)

        self.keySequence = QKeySequence()

        self.sequence = None

        self.recording = True

    def doneRecording(self):

        dbgMsg(" Done recording")

        self.releaseKeyboard()

        self.recording = False

        self.grabButton.setEnabled(True)

    def cancelRecording(self):

        dbgMsg("Recording Cancelled")

        self.doneRecording()

        self.keySequence = QKeySequence()

        self.sequence = None

        self.recording = False
