from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cc3d.player5.Graphics as Graphics
from .WindowInventory import WindowInventory
from cc3d.core.enums import *


class SubWindow(QMdiSubWindow):
    def __init__(self, _parent=None):
        '''
        parent points to QMdiArea
        '''
        super(SubWindow, self).__init__(_parent)
        self.parent = _parent
        self.main_widget = None

    def sizeHint(self):
        '''
        returns suggested size for qframe
        :return:QSize
        '''
        return QSize(400, 400)

    # set widget and widget fcns are not overloaded here. the default implementation of QMdiSubwindow is OK
    # def setWidget(self, widget):
    #     '''
    #     Places widget  in the frame's layout
    #     :param widget:widget to be added to Qframe
    #     :return:None
    #     '''

    # def widget(self):
    #     '''
    #     main widget displayed in Qframe
    #     :return: main widget displayed in Qframe
    #     '''
    #     return self.main_widget

    def mousePressEvent(self, ev):
        '''
        handler for mouse click event - updates self.parent.lastActiveRealWindow member variable
        :param ev: mousePressEvent
        :return:None
        '''
        self.parent.lastActiveRealWindow = self
        super(SubWindow,self).mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        '''
        handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable
        :param ev:  mouseDoubleClickEvent
        :return:None
        '''
        self.parent.lastActiveRealWindow = self
        super(SubWindow, self).mouseDoubleClickEvent(ev)

    # def changeEvent(self, ev):
    #     '''
    #     sets MainArea's lastActiveRealWindow - currently inactive
    #     :param ev: QEvent
    #     :return:None
    #     '''
    #
    #     return
        # if ev.type() == QEvent.ActivationChange:
        #     if self.isActiveWindow():
        #         print 'will activate ', self
        #         self.parent.lastActiveRealWindow = self
        #
        # super(DockSubWindow,self).changeEvent(ev)

    def closeEvent(self, ev):
        '''
        handler for close event event - removes sub window from inventory
        :param ev:  closeEvent
        :return:None
        '''
        self.parent.win_inventory.remove_from_inventory(self)

class PythonSteeringSubWindow(QMdiSubWindow):
    def __init__(self, _parent=None):
        '''
        parent points to QMdiArea
        '''
        super(PythonSteeringSubWindow, self).__init__(_parent)
        self.parent = _parent
        self.main_widget = None

    # def sizeHint(self):
    #     '''
    #     returns suggested size for qframe
    #     :return:QSize
    #     '''
    #     return QSize(400, 400)

    # set widget and widget fcns are not overloaded here. the default implementation of QMdiSubwindow is OK
    # def setWidget(self, widget):
    #     '''
    #     Places widget  in the frame's layout
    #     :param widget:widget to be added to Qframe
    #     :return:None
    #     '''

    # def widget(self):
    #     '''
    #     main widget displayed in Qframe
    #     :return: main widget displayed in Qframe
    #     '''
    #     return self.main_widget

    # def mousePressEvent(self, ev):
    #     '''
    #     handler for mouse click event - updates self.parent.lastActiveRealWindow member variable
    #     :param ev: mousePressEvent
    #     :return:None
    #     '''
    #     self.parent.lastActiveRealWindow = self
    #     super(SubWindow,self).mousePressEvent(ev)
    #
    # def mouseDoubleClickEvent(self, ev):
    #     '''
    #     handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable
    #     :param ev:  mouseDoubleClickEvent
    #     :return:None
    #     '''
    #     self.parent.lastActiveRealWindow = self
    #     super(SubWindow, self).mouseDoubleClickEvent(ev)
    #
    # # def changeEvent(self, ev):
    # #     '''
    # #     sets MainArea's lastActiveRealWindow - currently inactive
    # #     :param ev: QEvent
    # #     :return:None
    # #     '''
    # #
    # #     return
    #     # if ev.type() == QEvent.ActivationChange:
    #     #     if self.isActiveWindow():
    #     #         print 'will activate ', self
    #     #         self.parent.lastActiveRealWindow = self
    #     #
    #     # super(DockSubWindow,self).changeEvent(ev)
    #
    # def closeEvent(self, ev):
    #     '''
    #     handler for close event event - removes sub window from inventory
    #     :param ev:  closeEvent
    #     :return:None
    #     '''
    #     self.parent.win_inventory.remove_from_inventory(self)


class MainArea(QMdiArea):
    def __init__(self, stv,  ui ):
        # self.mdiarea = self
        self.MDI_ON = True

        self.stv = stv # SimpleTabView
        self.UI = ui # UserInterface

        QMdiArea.__init__(self, ui)
        # QMdiArea.__init__(self, parent)

        self.scrollView = QScrollArea(self)
        self.scrollView.setBackgroundRole(QPalette.Dark)
        self.scrollView.setVisible(False)

        #had to introduce separate scrollArea for 2D and 3D widgets. for some reason switching graphics widgets in Scroll area  did not work correctly.
        self.scrollView3D = QScrollArea(self)
        self.scrollView3D.setBackgroundRole(QPalette.Dark)
        self.scrollView3D.setVisible(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pass
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.win_inventory = WindowInventory()
        self.lastActiveRealWindow = None # keeps track of the last active real window

    def suggested_window_position(self):
        '''
        returns suggested position of the next window. For MainAreaMdi it returns QPoint(-1,-1)
        indicating that client code shuld use QMdiArea functionality to place windows
        :return:QPoint - position of the next window
        '''

        return QPoint(-1, -1)

    def addSubWindow(self, widget):
        '''Creates QMdiSubwindow containing widget and adds it to QMdiArea

        :param widget: widget that will be placed in the qmdisubwindow
        :return: None
        '''


        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            # obj_type = 'graphics'
            obj_type = GRAPHICS_WINDOW_LABEL
        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL
            # obj_type = 'plot'

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())

        # mdi_sub_window = QMdiSubWindow()
        mdi_sub_window = SubWindow(self)
        mdi_sub_window.setWidget(widget)
        mdi_sub_window.setAttribute(Qt.WA_DeleteOnClose)
        mdi_sub_window.setWindowTitle(window_name)

        QMdiArea.addSubWindow(self, mdi_sub_window)

        # old code that did not use SubWindow subclass
        # mdi_sub_window = QMdiArea.addSubWindow(self, widget)
        # mdi_sub_window.setWindowTitle(window_name)

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=obj_type)

        return mdi_sub_window

    def addSteeringSubWindow(self, widget):
        '''Creates QMdiSubwindow containing widget and adds it to QMdiArea

        :param widget: widget that will be placed in the qmdisubwindow
        :return: None
        '''


        # mdi_sub_window = QMdiSubWindow()
        mdi_sub_window = PythonSteeringSubWindow(self)
        mdi_sub_window.setWidget(widget)
        mdi_sub_window.setAttribute(Qt.WA_DeleteOnClose)
        mdi_sub_window.setWindowTitle('Steering Panel')

        QMdiArea.addSubWindow(self, mdi_sub_window)

        # old code that did not use SubWindow subclass
        # mdi_sub_window = QMdiArea.addSubWindow(self, widget)
        # mdi_sub_window.setWindowTitle(window_name)

        # self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=obj_type)

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=STEERING_PANEL_LABEL)
        return mdi_sub_window
