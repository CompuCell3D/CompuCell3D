from gc import collect
from weakref import ref
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cc3d.player5.Graphics as Graphics
from .WindowInventory import WindowInventory
from cc3d.core.enums import *


class SubWindow(QMdiSubWindow):
    def __init__(self, _parent=None):
        """
        parent points to QMdiArea
        """
        super(SubWindow, self).__init__(_parent)
        self.parent = _parent
        self.main_widget = None

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

    def sizeHint(self):
        """
        returns suggested size for qframe

        :return: QSize
        """
        return QSize(400, 400)

    # set widget and widget fcns are not overloaded here. the default implementation of QMdiSubwindow is OK
    # def setWidget(self, widget):
    #     """
    #     Places widget  in the frame's layout
    #     :param widget:widget to be added to Qframe
    #     :return:None
    #     """

    # def widget(self):
    #     """
    #     main widget displayed in Qframe
    #     :return: main widget displayed in Qframe
    #     """
    #     return self.main_widget

    def mousePressEvent(self, ev):
        """
        handler for mouse click event - updates self.parent.lastActiveRealWindow member variable

        :param ev: mousePressEvent
        :return: None
        """
        self.parent.lastActiveRealWindow = self
        super(SubWindow, self).mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        """
        handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable

        :param ev: mouseDoubleClickEvent
        :return: None
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

        :param ev: closeEvent
        :return: None
        """
        widget = self.widget()
        if widget:
            widget.closeEvent(ev)
        self.parent.removeSubWindow(self)
        super(SubWindow, self).closeEvent(ev)


class PythonSteeringSubWindow(QMdiSubWindow):
    def __init__(self, _parent=None):
        """
        parent points to QMdiArea
        """
        super(PythonSteeringSubWindow, self).__init__(_parent)
        self.title = 'Steering Panel'
        self.parent = _parent
        self.main_widget = None

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

    # def sizeHint(self):
    #     """
    #     returns suggested size for qframe
    #     :return:QSize
    #     """
    #     return QSize(400, 400)

    # set widget and widget fcns are not overloaded here. the default implementation of QMdiSubwindow is OK
    # def setWidget(self, widget):
    #     """
    #     Places widget  in the frame's layout
    #     :param widget:widget to be added to Qframe
    #     :return:None
    #     """

    # def widget(self):
    #     """
    #     main widget displayed in Qframe
    #     :return: main widget displayed in Qframe
    #     """
    #     return self.main_widget

    # def mousePressEvent(self, ev):
    #     """
    #     handler for mouse click event - updates self.parent.lastActiveRealWindow member variable
    #     :param ev: mousePressEvent
    #     :return:None
    #     """
    #     self.parent.lastActiveRealWindow = self
    #     super(SubWindow,self).mousePressEvent(ev)
    #
    # def mouseDoubleClickEvent(self, ev):
    #     """
    #     handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable
    #     :param ev:  mouseDoubleClickEvent
    #     :return:None
    #     """
    #     self.parent.lastActiveRealWindow = self
    #     super(SubWindow, self).mouseDoubleClickEvent(ev)
    #
    # # def changeEvent(self, ev):
    # #     """
    # #     sets MainArea's lastActiveRealWindow - currently inactive
    # #     :param ev: QEvent
    # #     :return:None
    # #     """
    # #
    # #     return
    #     # if ev.type() == QEvent.ActivationChange:
    #     #     if self.isActiveWindow():
    #     #         print 'will activate ', self
    #     #         self.parent.lastActiveRealWindow = self
    #     #
    #     # super(DockSubWindow,self).changeEvent(ev)
    #
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
        super(QMdiSubWindow, self).closeEvent(ev)


class MainArea(QMdiArea):
    def __init__(self, stv, ui):
        self.MDI_ON = True

        self.stv = stv  # SimpleTabView
        self.UI = ui  # UserInterface

        QMdiArea.__init__(self, ui)

        self.scrollView = QScrollArea(self)
        self.scrollView.setBackgroundRole(QPalette.Dark)
        self.scrollView.setVisible(False)

        # had to introduce separate scrollArea for 2D and 3D widgets.
        # for some reason switching graphics widgets in Scroll area did not work correctly.
        self.scrollView3D = QScrollArea(self)
        self.scrollView3D.setBackgroundRole(QPalette.Dark)
        self.scrollView3D.setVisible(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.win_inventory = WindowInventory()
        self.lastActiveRealWindow = None  # keeps track of the last active real window

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
        Returns suggested position of the next window. For MainAreaMdi it returns QPoint(-1,-1)
        indicating that client code shuld use QMdiArea functionality to place windows

        :return: position of the next window
        :rtype: QPoint
        """

        return QPoint(-1, -1)

    def addSubWindow(self, widget):
        """
        Creates QMdiSubwindow containing widget and adds it to QMdiArea

        :param widget: widget that will be placed in the qmdisubwindow
        :return: None
        """
        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            obj_type = GRAPHICS_WINDOW_LABEL
        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())

        mdi_sub_window = SubWindow(self)
        mdi_sub_window.setWidget(widget)
        mdi_sub_window.setAttribute(Qt.WA_DeleteOnClose)
        mdi_sub_window.setWindowTitle(window_name)

        QMdiArea.addSubWindow(self, mdi_sub_window)

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=obj_type)

        return mdi_sub_window

    def addSteeringSubWindow(self, widget):
        """
        Creates QMdiSubwindow containing widget and adds it to QMdiArea

        :param widget: widget that will be placed in the qmdisubwindow
        :return: None
        """
        mdi_sub_window = PythonSteeringSubWindow(self)
        mdi_sub_window.setWidget(widget)
        mdi_sub_window.setAttribute(Qt.WA_DeleteOnClose)
        mdi_sub_window.setWindowTitle(mdi_sub_window.title)

        QMdiArea.addSubWindow(self, mdi_sub_window)

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=STEERING_PANEL_LABEL)
        return mdi_sub_window

    def removeSubWindow(self, widget: QMdiSubWindow) -> None:
        """
        Removes a QMdiSubWindow from the QMdiArea.

        If there are any windows left in the inventory, the first one is activated.

        :param widget: subwindow
        :type widget: QMdiSubWindow
        :return: None
        """
        QMdiArea.removeSubWindow(self, widget)
        widget.deleteLater()
        self.win_inventory.remove_from_inventory(widget)
        if widget is self.lastActiveRealWindow:
            win_list = self.win_inventory.values()
            if win_list:
                self.lastActiveRealWindow = win_list[0]
            else:
                self.lastActiveRealWindow = None
        collect()
