"""

Have to check translation to native formats for KeySequence string opertations

"""

from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_KeyboardShortcuts
import sys
import cc3d.twedit5.twedit.ActionManager as am
from .KeyShortcut import KeyShortcutDlg
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, dbgMsg

MAC = "qt_mac_set_native_menubar" in dir()


class KeyboardShortcutsDlg(QDialog, ui_KeyboardShortcuts.Ui_KeyboardShortcutsDlg):
    # signals

    gotolineSignal = QtCore.pyqtSignal(('int',))

    def __init__(self, _currentEditor=None, parent=None):

        super(KeyboardShortcutsDlg, self).__init__(parent)

        self.editorWindow = parent

        self.currentEditor = _currentEditor

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        self.setupUi(self)

        # self.connect(self.shortcutTable,SIGNAL("cellClicked (int,int)"),self.shortcutCellClicked)

        self.shortcutTable.cellClicked.connect(self.shortcutCellClicked)

        self.lastClickPosition = None

        self.changesInActionShortcutList = []  # action name -> shortcut sequence for newly defined shortcuts

        self.shortcutItemDict = {}  # shortcut shortcut -> QTableItem

    # making sure that columns fill entire qtable widget

    def resizeEvent(self, e):

        shortcut_table_size = self.shortcutTable.size()

        self.shortcutTable.setColumnWidth(0, shortcut_table_size.width() / 2)

        self.shortcutTable.setColumnWidth(1, shortcut_table_size.width() / 2)

        e.accept()

    def initializeShortcutTables(self):

        # delete all rows first

        # dbgMsg("self.shortcutTable.rowCount()=",self.shortcutTable.rowCount())

        self.changesInActionShortcutList = []

        for i in range(self.shortcutTable.rowCount() - 1, -1, -1):
            self.shortcutTable.removeRow(i)

            # dbgMsg(" i=",i)

        row_idx = 0

        actions_sorted = list(am.actionToShortcutDict.keys())

        actions_sorted.sort()

        # empty QTableWidgetItem used to extract/prepare item format
        item = QTableWidgetItem()

        flags = item.flags()

        flags &= ~flags

        font = item.font()

        font.setBold(True)

        foreground_brush = item.foreground()
        for action in actions_sorted:
            shortcut = am.actionToShortcutDict[action]

            self.shortcutTable.insertRow(row_idx)

            action_item = QTableWidgetItem(action)

            flags = action_item.flags()

            flags &= ~flags

            action_item.setFlags(flags)

            action_item.setFont(font)

            action_item.setForeground(foreground_brush)

            self.shortcutTable.setItem(row_idx, 0, action_item)

            shortcut_item = QTableWidgetItem(shortcut)

            shortcut_item.setFlags(flags)

            shortcut_item.setFont(font)

            shortcut_item.setForeground(foreground_brush)

            self.shortcutTable.setItem(row_idx, 1, shortcut_item)

            # self.shortcutItemDict[shortcutItem]=shortcut

            row_idx += 1

    def assignNewShortcut(self, _newKeySequence, _actionItem, _shortcutItem):

        # this is simple linear operation - can use dictionaries to speed it up but for now we will use simple solution

        key_sequence_text = str(_newKeySequence.toString())
        action_text = str(_actionItem.text())

        for i in range(self.shortcutTable.rowCount()):

            shortcut_item = self.shortcutTable.item(i, 1)
            if str(shortcut_item.text()) == '':
                continue  # do not look for action with empty shortcut name

            if str(shortcut_item.text()) == key_sequence_text:
                action_item_local = self.shortcutTable.item(i, 0)

                # do nothing if changed shortcut for the action is same as old shortcut
                if str(action_text) != str(action_item_local.text()):

                    shortcutItemLocal = self.shortcutTable.item(i, 1)

                    shortcutItemLocal.setText('')

                    self.changesInActionShortcutList.append(str(action_item_local.text()))

                    self.changesInActionShortcutList.append(QKeySequence(''))

                    self.currentEditor.configuration.setKeyboardShortcut(str(action_item_local.text()), '')

                    break

        _shortcutItem.setText(_newKeySequence.toString())

        self.changesInActionShortcutList.append(action_text)

        self.changesInActionShortcutList.append(_newKeySequence)

        self.currentEditor.configuration.setKeyboardShortcut(action_text, key_sequence_text)

    def shortcutCellClicked(self, _row, _column):

        if _column == 1:

            # display grab shortcut widget

            shortcut_item = self.shortcutTable.item(_row, 1)
            action_item = self.shortcutTable.item(_row, 0)
            shortcut_text = shortcut_item.text()
            action_text = action_item.text()

            key_shortcut_dlg = KeyShortcutDlg(self, str(action_text), str(shortcut_text))
            ret = key_shortcut_dlg.exec_()

            if ret:
                new_key_sequence = key_shortcut_dlg.getKeySequence()

                dbgMsg("THIS IS NEW SHORTCUT:", str(new_key_sequence.toString()))

                self.assignNewShortcut(new_key_sequence, action_item, shortcut_item)

    def reassignNewShortcuts(self):

        for changeIdx in range(0, len(self.changesInActionShortcutList), 2):
            dbgMsg("actionText=", self.changesInActionShortcutList[changeIdx])

            dbgMsg("sequence=", str(self.changesInActionShortcutList[changeIdx + 1].toString()))

            am.setActionKeyboardShortcut(self.changesInActionShortcutList[changeIdx],
                                         self.changesInActionShortcutList[changeIdx + 1])

