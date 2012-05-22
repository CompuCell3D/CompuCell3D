# ------------------------------------------------------------
#
# CDTableOfBlockCellSizes - block cells in CellDraw - Mitja 2011
#
# ------------------------------------------------------------

from PyQt4 import QtGui, QtCore

# ======================================================================
# a helper class for a cell type's dimensions
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("oneBlockCellDimChangedSignal()"))
#
# ======================================================================
class CDTableOfBlockCellSizes(QtGui.QWidget):

    def __init__(self, pParent):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDTableOfBlockCellSizes, self).__init__(pParent)

        # don't show this widget until it's completely ready:
        self.hide()

        # init - windowing GUI stuff:
        #   (for explanations see the miInitGUI() function in the CDTableOfTypes class)
        # this title probably will not show up anywhere,
        #   since the widget will be used within another widget's layout:

        # init - create central table, set it up and show it inside the panel:
        #   (for explanations see the miInitCentralTableWidget() function
        #    in the CDTableOfTypes class)
        self.blockCellDimTable = QtGui.QTableWidget(self)
        self.blockCellDimTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.blockCellDimTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        self.blockCellDimTable.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)
        # self.blockCellDimTable.verticalHeader().hide()
        #
        self.blockCellDimTable.setColumnCount(1)
        self.blockCellDimTable.setRowCount(3)
        self.blockCellDimTable.setHorizontalHeaderLabels( " " )
        print " 123123123123123123123123123123123123123123123123123123123 "
        self.blockCellDimTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        print " 123123123123123123123123123123123123123123123123123123123 "

        self.blockCellDimTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Interactive)
        print " 123123123123123123123123123123123123123123123123123123123 "

        self.blockCellDimTable.setVerticalHeaderLabels( ("x", "y", "z") )
        self.blockCellDimTable.verticalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        self.blockCellDimTable.verticalHeader().setResizeMode(1, QtGui.QHeaderView.Interactive)
        self.blockCellDimTable.verticalHeader().setResizeMode(2, QtGui.QHeaderView.Interactive)
        #
        self.blockCellDimTable.setLineWidth(1)
        self.blockCellDimTable.setMidLineWidth(1)
#         self.blockCellDimTable.setFrameShape(QtGui.QFrame.Panel)
#         self.blockCellDimTable.setFrameShadow(QtGui.QFrame.Plain)

        self.oneBlockCellSizesList = list( (11, 22, 33) )

        self.populateOneRegionBlockCellSizesTable()


        # place the sub-widget in a layout, assign the layout to the CDTableOfBlockCellSizes:
        vbox = QtGui.QVBoxLayout()
        # these are the margins between the table and the "container widget", which
        #   should fill entirely the single element in the big table of types:
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(4)
        vbox.addWidget(self.blockCellDimTable)
        self.setLayout(vbox)
        self.layout().setAlignment(QtCore.Qt.AlignTop)

        # connect the "cellChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.blockCellDimTable.cellChanged[int, int].connect(self.handleTableCellChanged)

        self.show()

        # print "    - DEBUG ----- CDTableOfBlockCellSizes: __init__(): done"


    # ------------------------------------------------------------------
    # assign a dict parameter value to oneBlockCellSizesList:
    # ------------------------------------------------------------------
    def setOneRegionBlockCellSizes(self, pList):
        self.oneBlockCellSizesList = pList
        # print "___ - DEBUG ----- CDTableOfBlockCellSizes: setOneRegionBlockCellSizes() to", self.oneBlockCellSizesList, " done."


    # ------------------------------------------------------------------
    # rebuild the oneBlockCellSizesList global from its table contents:
    # ------------------------------------------------------------------
    def updateOneRegionCellSizes(self):
        print "___ - DEBUG DEBUG DEBUG ----- CDTableOfBlockCellSizes: self.updateOneRegionCellSizes() from ", self.oneBlockCellSizesList
        # parse each table rown separately to build a oneBlockCellSizesList:
        self.oneBlockCellSizesList = (   \
            ( int ( self.blockCellDimTable.item(0, 0).text() ) , \
              int ( self.blockCellDimTable.item(1, 0).text() ) , \
              int ( self.blockCellDimTable.item(2, 0).text() )   )    )
        print "___ - DEBUG ----- CDTableOfBlockCellSizes: self.updateOneRegionCellSizes() to ", self.oneBlockCellSizesList, " done."

    # ------------------------------------------------------------------
    # retrieve the up-to-date oneBlockCellSizesList for external use:
    # ------------------------------------------------------------------
    def getOneRegionDict(self):
        # first rebuild the oneBlockCellSizesList global from its table:
        self.updateOneRegionCellSizes()
        # print "___ - DEBUG ----- CDTableOfBlockCellSizes: getOneRegionDict is ", self.oneBlockCellSizesList, " done."
        return self.oneBlockCellSizesList

    # ------------------------------------------------------------------
    # populate the one-region subtable with data from oneBlockCellSizesList
    # ------------------------------------------------------------------
    def populateOneRegionBlockCellSizesTable(self):
        print "SIZE SIZE SIZE self.blockCellDimTable.height() =", self.blockCellDimTable.height()
        print "SIZE SIZE SIZE self.blockCellDimTable.parentWidget().height() =", self.blockCellDimTable.parentWidget().height()

        print "SIZE SIZE SIZE self.blockCellDimTable.width() =", self.blockCellDimTable.width()
        print "SIZE SIZE SIZE self.blockCellDimTable.parentWidget().width() =", self.blockCellDimTable.parentWidget().width()
        print "___ - DEBUG ----- CDTableOfBlockCellSizes: populateOneRegionBlockCellSizesTable() = ", \
              self.blockCellDimTable.rowCount()

        # prevent blockCellDimTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:
        self.blockCellDimTable.blockSignals(True)

        print "self.oneBlockCellSizesList =", self.oneBlockCellSizesList
        print "self.oneBlockCellSizesList[0] =", self.oneBlockCellSizesList[0]
        print "self.oneBlockCellSizesList[1] =", self.oneBlockCellSizesList[1]
        print "self.oneBlockCellSizesList[2] =", self.oneBlockCellSizesList[2]

        # create a first QTableWidgetItem and place text (i.e. the "x" cell size) from oneBlockCellSizesList in it:
        lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg( self.oneBlockCellSizesList[0] ) )
        # this goes to column 1 in the table:
        self.blockCellDimTable.setItem(0, 0, lItem)

        # create a first QTableWidgetItem and place text (i.e. the "y" cell size) from oneBlockCellSizesList in it:
        lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneBlockCellSizesList[1]) )
        # this goes to column 1 in the table:
        self.blockCellDimTable.setItem(1, 0, lItem)

        # create a first QTableWidgetItem and place text (i.e. the "z" cell size) from oneBlockCellSizesList in it:
        lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneBlockCellSizesList[2]) )
        # this goes to column 1 in the table:
        self.blockCellDimTable.setItem(2, 0, lItem)

#
#
#         # distribute the available space according to the space requirement of each column or row:
#         # TODO TODO: all this widget layout semi/auto/resizing still @#$@*Y@ does not work...
#         w = 0
#         for i in xrange(self.blockCellDimTable.columnCount()):
#             # print "column", i, "has width", self.blockCellDimTable.columnWidth(i)
#             w = w + self.blockCellDimTable.columnWidth(i)
#         w = w + self.blockCellDimTable.verticalHeader().width()
#         w = w + self.blockCellDimTable.verticalScrollBar().width()
#         # print "column and everything is", w
#
#         h = 0
#         for i in xrange(self.blockCellDimTable.rowCount()):
#             # print "row", i, "has height", self.blockCellDimTable.rowHeight(i)
#             h = h + self.blockCellDimTable.rowHeight(i)
#         h = h + self.blockCellDimTable.horizontalHeader().height()
#         h = h + self.blockCellDimTable.horizontalScrollBar().height()
#         # print "column and everything is", h
# #        self.blockCellDimTable.resize(w + 4, h + 4)
# #        self.blockCellDimTable.parentWidget().resize(w + 4, h + 4)
#
        self.blockCellDimTable.resizeRowsToContents()
        self.blockCellDimTable.resizeColumnsToContents()


#        print "SIZE SIZE SIZE self.blockCellDimTable.height() =", self.blockCellDimTable.height()
#        print "SIZE SIZE SIZE self.blockCellDimTable.parentWidget().height() =", self.blockCellDimTable.parentWidget().height()

#        print "SIZE SIZE SIZE self.blockCellDimTable.width() =", self.blockCellDimTable.width()
#        print "SIZE SIZE SIZE self.blockCellDimTable.parentWidget().width() =", self.blockCellDimTable.parentWidget().width()

        # start with no table cell selected, and the user can then click to select:
        self.blockCellDimTable.setCurrentCell(-1,-1)
        # self.blockCellDimTable.parentWidget().resize(self.blockCellDimTable.width(),self.blockCellDimTable.height())

        # allow blockCellDimTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:
        self.blockCellDimTable.blockSignals(False)

        # print "___ - DEBUG ----- CDTableOfBlockCellSizes: populateOneRegionBlockCellSizesTable() : done"


    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   from table items:
    # ------------------------------------------------------------------
    def handleTableCellChanged(self,pRow,pColumn):

        # from <http://lateral.netmanagers.com.ar/stories/BBS48.html> :
        #   "We could define a method in the Main class, and connect it to the
        #    itemChanged signal, but there is no need because we can use AutoConnect."
        #   "If you add to a class "Main" a method with a specific name,
        #    it will be connected to that signal. The name is on_objectname_signalname."

        print "___ - DEBUG ----- CDTableOfBlockCellSizes: handleTableCellChanged() pRow,pColumn =" , pRow,pColumn
        # update the dict:
        self.updateOneRegionCellSizes()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneBlockCellDimChangedSignal()"))

        print "___ - DEBUG ----- CDTableOfBlockCellSizes: handleTableCellChanged()  done."

# end of class CDTableOfBlockCellSizes(QtGui.QWidget)
# ======================================================================


