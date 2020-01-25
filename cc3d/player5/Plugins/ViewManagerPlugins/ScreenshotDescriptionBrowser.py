from PyQt5 import QtCore
from PyQt5.QtCore import *
import weakref
import cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView


class ScreenshotDescriptionBrowser(QObject):

    # newPlotWindowSignal = pyqtSignal(QMutex, object)

    def __init__(self, stv=None):
        QObject.__init__(self, None)
        self.__stv = weakref.ref(stv)

    @property
    def stv(self)->cc3d:
        # dereferencing weakref
        stv_obj = self.__stv()
        if not stv_obj:
            stv_obj = None
        return stv_obj

    def open(self):
        stv : cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView = self.stv
        if stv is None:
            return

        scr_mgr =  stv.screenshotManager
        if scr_mgr is None:
            return
        scr_desc_json_pth = stv.screenshotManager.get_screenshot_filename()

        print('scr_desc_json_pth=', scr_desc_json_pth)
        print(f'this is {self.stv}')


