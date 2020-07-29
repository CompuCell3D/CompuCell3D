from math import log
from cc3d.twedit5.twedit.utils.global_imports import *
import cc3d.twedit5.twedit.ActionManager as am


# have to implement custom class for QSciScintilla to handle properly wheel even with and without ctrl pressed

class QsciScintillaCustom(QsciScintilla):

    def __init__(self, parent=None, _panel=None):

        super(QsciScintillaCustom, self).__init__(parent)

        self.editorWindow = parent
        try:
            self.line_numbers_enabled = self.editorWindow.configuration.setting('DisplayLineNumbers')
        except AttributeError:
            self.line_numbers_enabled = False

        self.panel = _panel

        self.mousePressEventOrig = self.mousePressEvent

        self.CtrlKeyEquivalent = Qt.Key_Control

        self.scintillaDefinedLetterShortcuts = [ord('D'), ord('L'), ord('T'), ord('U'), ord('/'), ord(']')]

        self.customContextMenu = None

        self.linesChanged.connect(self.linesChangedHandler)

        if sys.platform.startswith("darwin"):
            self.CtrlKeyEquivalent = Qt.Key_Alt

    def wheelEvent(self, event):

        if qApp.keyboardModifiers() == Qt.ControlModifier:
            # Forwarding wheel event to editor windowwheelEvent

            event.ignore()

        else:
            # # calling wheelEvent from base class - regular scrolling
            super(QsciScintillaCustom, self).wheelEvent(event)

    def handleScintillaDefaultShortcut(self, modifierKeysText, event):

        if event.key() in self.scintillaDefinedLetterShortcuts:

            try:

                action = am.actionDict[am.shortcutToActionDict[modifierKeysText + '+' + chr(event.key())]]
                action.trigger()
                event.accept()
            except LookupError:
                super(QsciScintillaCustom, self).keyPressEvent(event)

        else:

            super(QsciScintillaCustom, self).keyPressEvent(event)

    def registerCustomContextMenu(self, _menu):

        self.customContextMenu = _menu

    def unregisterCustomContextMenu(self):

        self.customContextMenu = None

    def contextMenuEvent(self, _event):

        if not self.customContextMenu:

            super(QsciScintillaCustom, self).contextMenuEvent(_event)

        else:

            self.customContextMenu.exec_(_event.globalPos())

    def keyPressEvent(self, event):
        """
            senses if scintilla predefined keyboard shortcut was pressed.
        """

        if event.modifiers() == Qt.ControlModifier:
            self.handleScintillaDefaultShortcut('Ctrl', event)

        elif event.modifiers() & Qt.ControlModifier and event.modifiers() & Qt.ShiftModifier:
            self.handleScintillaDefaultShortcut('Ctrl+Shift', event)

        else:
            super(QsciScintillaCustom, self).keyPressEvent(event)

    def focusInEvent(self, event):
        editor_tab = 0

        if self.panel == self.editorWindow.panels[1]:
            editor_tab = 1

        self.editorWindow.activeTabWidget = self.panel

        self.editorWindow.handleNewFocusEditor(self)

        super(self.__class__, self).focusInEvent(event)


    def linesChangedHandler(self):
        '''
            adjusting width of the line number margin
        '''

        if not self.line_numbers_enabled:
            return

        if self.marginLineNumbers(0):

            number_of_lines = self.lines()
            number_of_digits = int(log(number_of_lines, 10)) + 2 if number_of_lines > 0 else 2
            self.setMarginWidth(0, '0' * number_of_digits)
