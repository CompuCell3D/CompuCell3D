from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cc3d.core.enums import *

from cc3d.player5 import Graphics
from .WindowInventory import WindowInventory
import sys
from weakref import ref
from gc import collect


class SubWindow(QFrame):
    def __init__(self, _parent):
        super(SubWindow, self).__init__(_parent)
        self.parent = _parent
        self.main_widget = None
        # self.setWindowFlags(Qt.Window|Qt.CustomizeWindowHint|Qt.WindowMaximizeButtonHint|Qt.WindowMinimizeButtonHint\
        # |Qt.WindowCloseButtonHint|Qt.FramelessWindowHint)

        # note Qt.Drawer looks completely different on OSX than on Windows.
        # QWindow on the other hand on linux displays all windows in dock widget and behaves stranegely
        # thus the settings below
        # are actually the ones that work on all platforms

        if sys.platform.startswith('darwin'):
            # on OSX we apply different settings than on other platforms
            self.setWindowFlags(
                Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
                | Qt.WindowCloseButtonHint | Qt.FramelessWindowHint
            )
        else:
            self.setWindowFlags(
                Qt.Dialog | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
            )

    @property
    def parent(self):
        try:
            o = self._parent()
        except TypeError:
            o = self._parent
        return o

    @parent.setter
    def parent(self, _i):
        try:
            self._parent = ref(_i)
        except TypeError:
            self._parent = _i

    @property
    def main_widget(self):
        try:
            o = self._main_widget()
        except TypeError:
            o = self._main_widget
        return o

    @main_widget.setter
    def main_widget(self, _i):
        try:
            self._main_widget = ref(_i)
        except TypeError:
            self._main_widget = _i

    def setWidget(self, widget):
        """
        Places widget  in the frame's layout

        :param widget:widget to be added to Qframe
        :return:None
        """
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.addWidget(widget)
        # layout.setMargin(0)
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)
        self.main_widget = widget
        self.setLayout(layout)

    def sizeHint(self):
        """
        returns suggested size for qframe

        :return:QSize
        """
        return QSize(400, 400)

    def widget(self):
        """
        main widget displayed in Qframe

        :return: main widget displayed in Qframe
        """
        return self.main_widget

    def mousePressEvent(self, ev):
        """
        handler for mouse click event - updates self.parent.lastActiveRealWindow member variable

        :param ev: mousePressEvent
        :return:None
        """
        self.parent.lastActiveRealWindow = self
        super(SubWindow, self).mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        """
        handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable

        :param ev:  mouseDoubleClickEvent
        :return:None
        """
        self.parent.lastActiveRealWindow = self
        super(SubWindow, self).mouseDoubleClickEvent(ev)

    # def changeEvent(self, ev):
    #     """
    #     sets MainArea's lastActiveRealWindow - currently inactive
    #     :param ev: QEvent
    #     :return:None
    #     """
    #
    #     return
    # if ev.type() == QEvent.ActivationChange:
    #     if self.isActiveWindow():
    #         print 'will activate ', self
    #         self.parent.lastActiveRealWindow = self
    #
    # super(DockSubWindow,self).changeEvent(ev)

    def closeEvent(self, ev):
        """
        handler for close event event - removes sub window from inventory

        :param ev:  closeEvent
        :return:None
        """
        widget = self.widget()
        if widget:
            widget.closeEvent(ev)
        self.parent.removeSubWindow(self)
        super(SubWindow, self).closeEvent(ev)


class PythonSteeringSubWindow(QFrame):
    def __init__(self, _parent=None):
        super(PythonSteeringSubWindow, self).__init__(_parent)
        self.title = 'Steering Panel'
        self.parent = _parent
        self.main_widget = None


class MainArea(QWidget):
    def __init__(self, stv, ui):

        self.MDI_ON = False

        self.stv = stv  # SimpleTabView
        self.UI = ui  # UserInterface

        self.win_inventory = WindowInventory()

        self.lastActiveRealWindow = None  # keeps track of the last active real window
        self.last_suggested_window_position = QPoint(0, 0)

    @property
    def stv(self):
        """
        SimpleTabView inheriting instance; same as self

        :return: SimpleTabView instance
        :rtype: cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView.SimpleTabView
        """
        return self._stv()

    @stv.setter
    def stv(self, _i):
        self._stv = ref(_i)

    @property
    def UI(self):
        """
        Parent UserInterface

        :return: parent
        :rtype: cc3d.player5.UI.UserInterface.UserInterface
        """
        return self._UI()

    @UI.setter
    def UI(self, _i):
        self._UI = ref(_i)

    @property
    def lastActiveRealWindow(self):
        """
        Last active subwindow if any, otherwise None
        """
        try:
            o = self._lastActiveRealWindow()
        except TypeError:
            o = self._lastActiveRealWindow
        return o

    @lastActiveRealWindow.setter
    def lastActiveRealWindow(self, _i):
        try:
            self._lastActiveRealWindow = ref(_i)
        except TypeError:
            self._lastActiveRealWindow = _i

    def suggested_window_position(self):
        """
        returns suggested position of the next window

        :return:QPoint - position of the next window
        """

        rec = QApplication.desktop().screenGeometry()

        if self.last_suggested_window_position.x() == 0 and self.last_suggested_window_position.y() == 0:

            self.last_suggested_window_position = QPoint(int(rec.width() / 5), int(rec.height() / 5))
            return self.last_suggested_window_position
        else:
            from random import randint
            self.last_suggested_window_position = QPoint(randint(int(rec.width() / 5), int(rec.width() / 2)),
                                                         randint(int(rec.height() / 5), int(rec.height() / 2)))
            return self.last_suggested_window_position

    def addSubWindow(self, widget):
        """
        adds subwindow containing widget to the player5

        :param widget: widget to be added to sub windows
        :return:None
        """

        print('INSTANCE OF GraphicsFrameWidget =  ',
              isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget))
        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            obj_type = GRAPHICS_WINDOW_LABEL

        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())

        subWindow = self.createSubWindow(name=window_name)  # sub windowª
        self.setupSubWindow(subWindow, widget, window_name)

        # inserting widget into dictionary
        self.win_inventory.add_to_inventory(obj=subWindow, obj_type=obj_type)

        return subWindow

    def addSteeringSubWindow(self, widget):
        """
        Creates QMdiSubwindow containing widget and adds it to QMdiArea

        :param widget: widget that will be placed in the qmdisubwindow
        :return: None
        """

        mdi_sub_window = PythonSteeringSubWindow(self)
        subWindow = self.createSubWindow(name='Steering Panel')  # sub window
        self.setupSubWindow(subWindow, widget, 'Steering Panel')

        subWindow.resize(widget.sizeHint())

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=STEERING_PANEL_LABEL)
        return mdi_sub_window

    def tileSubWindows(self):
        """
        dummy function to make conform to QMdiArea API

        :return: None
        """
        pass

    def cascadeSubWindows(self):
        """
        dummy function to make conform to QMdiArea API

        :return: None
        """
        pass

    def activeSubWindow(self):
        """
        returns last active subwindow

        :return: SubWindow object
        """
        print('returning lastActiveRealWindow=', self.lastActiveRealWindow)
        return self.lastActiveRealWindow

    def setActiveSubWindow(self, win):
        """
        Activates subwindow win

        :param: win - SubWindow object
        :return: None
        """
        win.activateWindow()
        self.lastActiveRealWindow = win

    def subWindowList(self):
        """
        returns list of all open subwindows

        :return: python list of SubWindow objects
        """
        return list(self.win_inventory.values())

    def createSubWindow(self, name):
        """
        Creates SubWindow with title specified using name parameter

        :param: name -  subwindow title
        :return: SubWindow object
        """

        sub_window = SubWindow(self)
        sub_window.setObjectName(name)
        return sub_window

    def setupSubWindow(self, sub_window, widget, caption):
        """
        Configures subwindow by placing widget in to qframe layout, setting window title (caption)
        and showing subwindow

        :param: sub_window - SubWindow object
        :param: widget - widget to be placed into sub_window
        :param: caption - subwindow title
        :return: None
        """

        if caption is None:
            caption = ''

        sub_window.setWindowTitle(caption)
        sub_window.setWidget(widget)
        sub_window.show()

    def removeSubWindow(self, widget: QFrame) -> None:
        """
        Removes a QFrame from the QWidget.

        If there are any windows left in the inventory, the first one is activated.

        :param widget: subwindow
        :type widget: QFrame
        :return: None
        """
        self.win_inventory.remove_from_inventory(widget)
        widget.deleteLater()
        if widget is self.activateWindow():
            win_list = self.win_inventory.values()
            if win_list:
                self.setActiveSubWindow(win_list[0])
            else:
                self.lastActiveRealWindow = None
        collect()
