from cc3d.twedit5.twedit.utils.global_imports import *
from . import ui_newfilewizard
import sys
import os.path
from os.path import *
from cc3d.twedit5.twedit.utils import qt_obj_hash

MAC = "qt_mac_set_native_menubar" in dir()


class NewFileWizard(QWizard, ui_newfilewizard.Ui_NewFileWizard):

    def __init__(self, parent=None):

        super(NewFileWizard, self).__init__(parent)

        self.cc3dProjectTreeWidget = parent

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        # self.gotolineSignal.connect(self.editorWindow.goToLine)

        self.projectPath = ""

        self.setupUi(self)

        # if not MAC:

        # self.cancelButton.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

        # @pyqtSignature("") # signature of the signal emited by the button

        # def on_okButton_clicked(self):

        # self.findChangedConfigs()        

        # self.close()

    @pyqtSlot()  # signature of the signal emited by the button
    def on_nameBrowsePB_clicked(self):

        fileName, _ = QFileDialog.getOpenFileName(self, "Save File", self.projectPath, "*")

        fileName = abspath(str(fileName))  # normalizing path

        self.nameLE.setText(fileName)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_locationBrowsePB_clicked(self):

        dir_name, _ = QFileDialog.getExistingDirectory(self,

                                                       "Directory (within current project) for the new file...",

                                                       self.projectPath)

        dir_name = str(dir_name)

        dir_name = os.path.abspath(dir_name)  # normalizing path

        relative_path = self.findRelativePath(self.projectPath, dir_name)

        if dir_name == relative_path:
            QMessageBox.warning(self, "Directory outside the project",

                                "You are trying to create new file outside project directory.<br> This is not allowed",

                                QMessageBox.Ok)

            relative_path = "Simulation/"

        self.locationLE.setText(relative_path)

    def findRelativePathSegments(self, basePath, p, rest=[]):

        """

            This function finds relative path segments of path p with respect to base path    

            It returns list of relative path segments and flag whether operation succeeded or not    

        """

        h, t = os.path.split(p)

        path_match = False

        if h == basePath:
            path_match = True

            return [t] + rest, path_match

        print("(h,t,pathMatch)=", (h, t, path_match))

        if len(h) < 1: return [t] + rest, path_match

        if len(t) < 1: return [h] + rest, path_match

        return self.findRelativePathSegments(basePath, h, [t] + rest)

    def findRelativePath(self, basePath, p):

        relative_path_segments, pathMatch = self.findRelativePathSegments(basePath, p)

        if pathMatch:

            relative_path = ""

            for i in range(len(relative_path_segments)):

                segment = relative_path_segments[i]

                relative_path += segment

                if i != len(relative_path_segments) - 1:
                    relative_path += "/"  # we use unix style separators - they work on all (3) platforms

            return relative_path

        else:

            return p

            # initialize wizard page

    def updateUi(self):

        self.locationLE.setText("Simulation/")  # default storage of simulation files

        tw = self.cc3dProjectTreeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        pdh = None

        try:

            pdh = tw.plugin.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError:

            # could not find simulation data handler for this item

            return

        self.projectPath = pdh.cc3dSimulationData.basePath

        self.projectDirLE.setText(pdh.cc3dSimulationData.basePath)

        # construct a list of available file types

        if pdh.cc3dSimulationData.xmlScript == "":
            self.fileTypeCB.insertItem(0, "XML Script")

        if pdh.cc3dSimulationData.pythonScript == "":
            self.fileTypeCB.insertItem(0, "Main Python Script")

        return
