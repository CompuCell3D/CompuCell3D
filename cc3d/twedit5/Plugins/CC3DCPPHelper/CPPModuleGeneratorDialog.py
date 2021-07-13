from cc3d.twedit5.twedit.utils.global_imports import *
import re
from . import ui_c_plus_plus_module_dialog

MAC = "qt_mac_set_native_menubar" in dir()


class CPPModuleGeneratorDialog(QDialog, ui_c_plus_plus_module_dialog.Ui_C_Plus_Plus_Module_Dialog):
    def __init__(self, parent=None):

        super(CPPModuleGeneratorDialog, self).__init__(parent)

        self.__ui = parent

        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        self.projectPath = ""
        self.setupUi(self)
        self.steppableDirRegex = re.compile('steppables')
        self.pluginDirRegex = re.compile('plugins')

    @pyqtSlot()  # signature of the signal emited by the button
    def on_okPB_clicked(self):

        error_flag = False

        module_dir = str(self.moduleDirLE.text())

        module_dir.strip()

        if str(self.moduleCoreNameLE.text()).strip() == "":
            QMessageBox.warning(self, "Empty Core Module Name", "Please specify C++ core module name")

            error_flag = True

        if str(self.moduleDirLE.text()).strip() == "":
            QMessageBox.warning(self, "Empty Module Directory Name",
                                "Please specify root directory where subdirectory with module files will be stored")

            error_flag = True

            # performing rudimentary check to make sure that steppable are written into
            # steppables directory and plugins into plugins directory

        if self.steppableRB.isChecked():

            steppable_dir_found = re.search(self.steppableDirRegex, module_dir)

            if not steppable_dir_found:

                ret = QMessageBox.warning(self, "Possible Directory Name Mismatch",
                                          "Are you sure you want to create steppable in <br> %s ?" % module_dir,
                                          QMessageBox.No | QMessageBox.Yes)

                if ret == QMessageBox.No:
                    error_flag = True

        if self.pluginRB.isChecked():

            plugin_dir_found = re.search(self.pluginDirRegex, module_dir)

            if not plugin_dir_found:
                ret = QMessageBox.warning(self, "Possible Directory Name Mismatch",
                                          "Are you sure you want to create plugin in <br> %s ?" % module_dir,
                                          QMessageBox.No | QMessageBox.Yes)

                if ret == QMessageBox.No:
                    error_flag = True

        if not error_flag:
            self.accept()

    @pyqtSlot()  # signature of the signal emited by the button
    def on_moduleDirPB_clicked(self):

        recent_dir = self.moduleDirLE.text()

        dir_name = QFileDialog.getExistingDirectory(self,"Module root directory - subdirectory named after"
                                                        " module core name will be created", recent_dir)

        dir_name = str(dir_name)
        dir_name.rstrip()

        if dir_name != '':
            dir_name = os.path.abspath(dir_name)  # normalizing path

            self.moduleDirLE.setText(dir_name)
