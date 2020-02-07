import sys
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from cc3d.player5.Plugins.ViewManagerPlugins import ui_screenshot_description_browser
from cc3d.player5.Plugins.ViewManagerPlugins.QJsonBrowser import QJsonModel, QJsonTreeView
import weakref
import cc3d
from os.path import exists


class ScreenshotDescriptionBrowser(QDialog, ui_screenshot_description_browser.Ui_screenshotDescriptionDialog):

    def __init__(self, parent=None):
        super(ScreenshotDescriptionBrowser, self).__init__(parent)
        self.stv = weakref.ref(parent)
        self.model = None
        self.view = None

        if sys.platform.startswith('win'):
            # dialogs without context help - only close button exists
            self.setWindowFlags(Qt.Drawer)

        self.projectPath = ""

        self.setupUi(self)

        self.updateUi()

    def load(self):
        stv: cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView.SimpleTabView = self.stv()

        scr_desc_no_found_msg_str = 'Could not find screenshot description file. ' \
                                    'Make sure simulation is running (and ideally is paused) ' \
                                    'and you have generated screenshot description file - by clicking camera button'

        if stv is None or stv.screenshotManager is None:
            # When not running
            from cc3d import CompuCellSetup
            sim_file_name = CompuCellSetup.persistent_globals.simulation_file_name

            if sim_file_name is None or sim_file_name == '':
                self.scr_list_TE.setPlainText(scr_desc_no_found_msg_str)
                return

            from os.path import dirname, join
            scr_desc_json_pth = join(dirname(sim_file_name), 'screenshot_data', 'screenshots.json')
        else:
            # When running
            scr_desc_json_pth = stv.screenshotManager.get_screenshot_filename()


        if not exists(scr_desc_json_pth):
            self.scr_list_TE.setPlainText(scr_desc_no_found_msg_str)
            return

        self.view = QJsonTreeView()
        self.model = QJsonModel()

        self.view.setModel(self.model)


        with open(scr_desc_json_pth, 'r') as j_in:

            document = json.load(j_in)

        try:
            available_screenshots = list(document['ScreenshotData'].keys())
            available_screenshots_str = '\n'.join(available_screenshots)
        except KeyError:
            available_screenshots_str = 'COULD NOT FIND SCREENSHOT CONFIGURATION'

        self.scr_list_TE.setPlainText(available_screenshots_str)

        self.model.load(document)
        # Sanity check
        assert (
                json.dumps(self.model.json(), sort_keys=True) ==
                json.dumps(document, sort_keys=True)
        )
        self.v_layout.insertWidget(3, self.view, stretch=10)

    def updateUi(self):
        """

        :return:
        """
