"""
Module implementing a TabWidget class substituting QTabWidget.
"""

from PyQt5.QtWidgets import QTabWidget

from PyQt5.QtCore import Qt

class CTabWidget(QTabWidget):
    """
    Class implementing a TabWidget class substituting QTabWidget.
    
    It provides slots to show the previous and next tab and give
    them the input focus and it allows to have a context menu for the tabs.
    
    @signal customTabContextMenuRequested(const QPoint & point, int index) emitted when
        a context menu for a tab is requested
    """
    def nextTab(self):
        """
        Public slot used to show the next tab.
        """
        ind = self.currentIndex() + 1
        if ind == self.count():
            ind = 0
            
        self.setCurrentIndex(ind)
        self.currentWidget().setFocus()

    def prevTab(self):
        """
        Public slot used to show the previous tab.
        """
        ind = self.currentIndex() - 1
        if ind == -1:
            ind = self.count() - 1
            
        self.setCurrentIndex(ind)
        self.currentWidget().setFocus()

    def setTabContextMenuPolicy(self, policy):
        """
        Public method to set the context menu policy of the tab.
        
        @param policy context menu policy to set (Qt.ContextMenuPolicy)
        """
        self.tabBar().setContextMenuPolicy(policy)
        # if policy == Qt.CustomContextMenu:
        #     self.connect(self.tabBar(),
        #         SIGNAL("customContextMenuRequested(const QPoint &)"),
        #         self.__handleTabCustomContextMenuRequested)
        # else:
        #     self.disconnect(self.tabBar(),
        #         SIGNAL("customContextMenuRequested(const QPoint &)"),
        #         self.__handleTabCustomContextMenuRequested)

        if policy == Qt.CustomContextMenu:
            self.tabBar().customContextMenuRequested.connect(self.__handleTabCustomContextMenuRequested)
            # self.connect(self.tabBar(),
            #              SIGNAL("customContextMenuRequested(const QPoint &)"),
            #              self.__handleTabCustomContextMenuRequested)
        else:
            self.tabBar().customContextMenuRequested.dicconnect(self.__handleTabCustomContextMenuRequested)
            # self.disconnect(self.tabBar(),
            #                 SIGNAL("customContextMenuRequested(const QPoint &)"),
            #                 self.__handleTabCustomContextMenuRequested)

    def __handleTabCustomContextMenuRequested(self, point):
        """
        Private slot to handle the context menu request for the tabbar.
        
        @param point point the context menu was requested (QPoint)
        """
        _tabbar = self.tabBar()
        for index in range(_tabbar.count()):
            rect = _tabbar.tabRect(index)
            if rect.contains(point):
                self.customTabContextMenuRequested.emit(_tabbar.mapToParent(point), index)
                # self.emit(SIGNAL("customTabContextMenuRequested(const QPoint &, int)"),
                #           _tabbar.mapToParent(point), index)
                break
    
    def selectTab(self, pos):
        """
        Public method to get the index of a tab given a position.
        
        @param pos position determining the tab index (QPoint)
        @return index of the tab (integer)
        """
        _tabbar = self.tabBar()
        for index in range(_tabbar.count()):
            rect = _tabbar.tabRect(index)
            if rect.contains(pos):
                return index
        
        return -1

    def moveTab(self, curIndex, newIndex):
        """
        Public method to move a tab to a new index.
        
        @param curIndex index of tab to be moved (integer)
        @param newIndex index the tab should be moved to (integer)
        """
        # step 1: save the tab data of tab to be moved
        toolTip = self.tabToolTip(curIndex)
        text = self.tabText(curIndex)
        icon = self.tabIcon(curIndex)
        whatsThis = self.tabWhatsThis(curIndex)
        widget = self.widget(curIndex)
        curWidget = self.currentWidget()
        
        # step 2: move the tab
        self.removeTab(curIndex)
        self.insertTab(newIndex, widget, icon, text)
        
        # step 3: set the tab data again
        self.setTabToolTip(newIndex, toolTip)
        self.setTabWhatsThis(newIndex, whatsThis)
        
        # step 4: set current widget
        self.setCurrentWidget(curWidget)
