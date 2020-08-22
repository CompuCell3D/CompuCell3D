import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from . import ui_itemproperties

MAC = "qt_mac_set_native_menubar" in dir()


class ItemProperties(QDialog, ui_itemproperties.Ui_ItemProperties):

    def __init__(self, parent=None):
        super(ItemProperties, self).__init__(parent)

        self.cc3dProjectTreeWidget = parent

        self.resourceReference = None

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        self.projectPath = ""

        self.setupUi(self)


    def setResourceReference(self, _ref):
        self.resourceReference = _ref

        print("\n\n\n\n\n\n self.resourceReference=", self.resourceReference)

    # initialize properties dialog

    def updateUi(self):
        self.pathLabel.setText(self.resourceReference.path)

        self.typeLabel.setText(self.resourceReference.type)

        self.moduleLE.setText(self.resourceReference.module)

        self.originLE.setText(self.resourceReference.origin)

        self.copyCHB.setChecked(self.resourceReference.copy)

