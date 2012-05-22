#!/usr/bin/env python
#
# cdTypesEditor - add-on panel for CellDraw - Mitja 2012
#
# (original PIF_Generator code written by ??? in 2008 (or 2009???)
#  the original code contained no comments nor code documentation)
# ------------------------------------------------------------






# ------------------------------------------------------------
# the PyQt4 demo port of the tools/settings editor example from Qt v4.x, from
#   which we use in part for its QTreeWidget-related routines, includes the
#    following sip import and setting of api versions:
# This is only needed for Python v2 but is harmless for Python v3.
# import sip
# sip.setapi('QString', 2)
# sip.setapi('QVariant', 2)
# ------------------------------------------------------------

import sys     # for handling command-line arguments, could be removed in final version
import inspect # for debugging functions, could be removed in deployed version

import random  # for generating regions with semi-random-colors for different cell types

import time    # for sleep()


# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# from PyQt4 import QtGui, QtCore, Qt
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--

# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# from PyQt4 import QtGui, QtCore
from PyQt4 import QtCore # from PyQt4.QtCore import *
from PyQt4 import QtGui # from PyQt4.QtGui import *
import PyQt4
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--

# 2012 - Mitja: external class based on QTreeWidget for editing types:
# from cdTypesTree import CDTypesTree

# 2011 - Mitja: external QWidget for selecting the PIFF Generation mode:
from cdControlPIFFGenerationMode import CDControlPIFFGenerationMode

# 2011 - Mitja: external QWidget to input x,y,z block cell dimensions:
from cdTableOfBlockCellSizes import CDTableOfBlockCellSizes

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# debugging functions, remove in final Panel version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]



# ======================================================================
# a helper class for the small table-within table for each region
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))
#
# ======================================================================
class PIFOneRegionTable(QtGui.QWidget):

    def __init__(self, pParent):
        # it is compulsory to call the parent's __init__ class right away:
        super(PIFOneRegionTable, self).__init__(pParent)

        # don't show this widget until it's completely ready:
        self.hide()

        # init - windowing GUI stuff:
        #   (for explanations see the __miInitEditorGUIGeneralBehavior() function below, in the CDTypesEditor class)
        # this title probably will not show up anywhere,
        #   since the widget will be used within another widget's layout:
        self.setWindowTitle("PIFF Cell Types in One Region")

        #
        # QWidget setup - more windowing GUI setup:
        #

        miDialogsWindowFlags = QtCore.Qt.WindowFlags()
        # this panel is a so-called "Tool" (by PyQt and Qt definitions)
        #    we'd use the Tool type of window, except for this oh-so typical Qt bug:
        #    http://bugreports.qt.nokia.com/browse/QTBUG-6418
        #    i.e. it defines a system-wide panel which shows on top of *all* applications,
        #    even when this application is in the background.
        # miDialogsWindowFlags = QtCore.Qt.Tool
        #    so we use a plain QtCore.Qt.Window type instead:
        miDialogsWindowFlags = QtCore.Qt.Window
        #    add a peculiar WindowFlags combination to have no close/minimize/maxize buttons:
        miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
        miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowMinimizeButtonHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(miDialogsWindowFlags)

        # 1. The widget is not modal and does not block input to other widgets.
        # 2. If widget is inactive, the click won't be seen by the widget.
        #    (it does NOT work as Qt docs says it would on Mac OS X: click-throughs don't get disabled)
        # 3. The widget can choose between alternative sizes for widgets to avoid clipping.
        # 4. The native Carbon size grip should be opaque instead of transparent.
        self.setAttribute(QtCore.Qt.NonModal  | \
                          QtCore.Qt.WA_MacNoClickThrough | \
                          QtCore.Qt.WA_MacVariableSize | \
                          QtCore.Qt.WA_MacOpaqueSizeGrip )

        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)



        # init - create central table, set it up and show it inside the panel:
        #   (for explanations see the NOTUSED__miInitCentralTableWidget() function below,
        #    in the CDTypesEditor class)
        self.oneRegionTable = QtGui.QTableWidget(self)
        self.oneRegionTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.oneRegionTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        self.oneRegionTable.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)
        # self.oneRegionTable.verticalHeader().hide()
        #
        self.oneRegionTable.setColumnCount(5)
        self.oneRegionTable.setRowCount(1)
        self.oneRegionTable.setHorizontalHeaderLabels( (" ", "Cell\nType", "Amount", "Fraction", "Volume") )
        self.oneRegionTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(3, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(4, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setStretchLastSection(True)
        self.oneRegionTable.verticalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        #
        self.oneRegionTable.setLineWidth(1)
        self.oneRegionTable.setMidLineWidth(1)
        self.oneRegionTable.setFrameShape(QtGui.QFrame.Panel)
        self.oneRegionTable.setFrameShadow(QtGui.QFrame.Plain)

        # init - create region dict, populate it with empty values:
        self.oneRegionDict = dict()

        self.oneRegionDict = dict( { 1: [QtGui.QColor(64,64,64), "Condensing", 3, 100], \
                                     2: [QtGui.QColor(64,64,64), "NonCondensing", 4, 80]  } )

        # init - create pattern dict, it will be used to distinguish between different types in the region:
        self.patternDict = dict()
        self.patternDict = dict({ 0: QtCore.Qt.SolidPattern, \
                                  1: QtCore.Qt.Dense2Pattern, \
                                  2: QtCore.Qt.Dense4Pattern, \
                                  3: QtCore.Qt.Dense6Pattern, \
                                  4: QtCore.Qt.HorPattern, \
                                  5: QtCore.Qt.VerPattern, \
                                  6: QtCore.Qt.CrossPattern, \
                                  7: QtCore.Qt.BDiagPattern, \
                                  8: QtCore.Qt.FDiagPattern, \
                                  9: QtCore.Qt.DiagCrossPattern    } )
        # print "patternDict = ", self.patternDict

        self.populateOneRegionSubTable()

        # a separate "QDialogButtonBox" widget, to have "add" and "remove" buttons
        #   that allow the user to add/remove lines from the oneRegionTable:
        addTableRowButton = QtGui.QPushButton("+")
        removeTableRowButton = QtGui.QPushButton("-")
        self.buttonBox = QtGui.QDialogButtonBox()
        self.buttonBox.addButton(addTableRowButton, QtGui.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(removeTableRowButton, QtGui.QDialogButtonBox.RejectRole)
        # connects signals from buttons to "slot" methods:
        self.buttonBox.accepted.connect(self.handleAddTableRow)
        self.buttonBox.rejected.connect(self.handleRemoveTableRow)

        # place the sub-widget in a layout, assign the layout to the PIFOneRegionTable:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(4)
        vbox.addWidget(self.oneRegionTable)
        vbox.addWidget(self.buttonBox)
        self.setLayout(vbox)
        self.layout().setAlignment(QtCore.Qt.AlignTop)

        # connect the "cellChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.oneRegionTable.cellChanged[int, int].connect(self.handleTableCellChanged)

        self.show()

        # print "    - DEBUG ----- PIFOneRegionTable: __init__(): done"


    # ------------------------------------------------------------------
    # assign a dict parameter value to oneRegionDict:
    # ------------------------------------------------------------------
    def setOneRegionDict(self, pDict):
        self.oneRegionDict = pDict
        # print "___ - DEBUG ----- PIFOneRegionTable: setOneRegionDict() to", self.oneRegionDict, " done."


    # ------------------------------------------------------------------
    # rebuild the oneRegionDict global from its table contents:
    # ------------------------------------------------------------------
    def updateOneRegionDict(self):
        # print "___ - DEBUG DEBUG DEBUG ----- PIFOneRegionTable: self.updateOneRegionDict() from ", self.oneRegionDict

        # set how many rows are present in the table:
        lTypesCount = self.oneRegionTable.rowCount()

        # get the oneRegionDict keys in order to access elements:
        lKeys = self.oneRegionDict.keys()
       
        # add additional (non-editable) table column with percentages calculated at each table content update:
        lTotalAmounts = 0.0

        # parse each table rown separately to build a oneRegionDict entry:
        for i in xrange(lTypesCount):

            #   the key is NOT retrieved from the first oneRegionTable column,
            #   (since we don't show oneRegionDict keys in oneRegionTable anymore) :
            # lKey = self.oneRegionTable.item(i, 0)
            lKey = lKeys[i]

            # the color remains the same (table color is not editable (yet?) )
            lColor = self.oneRegionDict[lKey][0]
            # the region name is retrieved from the third oneRegionTable column:
            lTypeName = str ( self.oneRegionTable.item(i, 1).text() )
            # the amount (or "percentage" of total) for this cell type is retrieved from the third oneRegionTable column:
            lAmount = float ( self.oneRegionTable.item(i, 2).text() )
            # keep track of the total of all individual cell type amounts:
            lTotalAmounts = lTotalAmounts + lAmount
            # 2011 - add a Volume field for each cell type:
            lVolume = float ( self.oneRegionTable.item(i, 4).text() )

            # rebuild the i-th dict entry:
            self.oneRegionDict[lKey] = [ lColor, lTypeName, lAmount, lVolume ]


        # now fill the additional (non-editable) table column with percentages:
        # (prevent oneRegionTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:)
        self.oneRegionTable.blockSignals(True)
        for i in xrange(lTypesCount):
            lKey = lKeys[i]
            # create a fourth QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg( self.oneRegionDict[lKey][2] / lTotalAmounts ) )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 2 in the table:
            self.oneRegionTable.setItem(i, 3, lItem)

        # (allow oneRegionTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:)
        self.oneRegionTable.blockSignals(False)


        # print "___ - DEBUG ----- PIFOneRegionTable: self.updateOneRegionDict() to ", self.oneRegionDict, " done."

    # ------------------------------------------------------------------
    # retrieve the up-to-date oneRegionDict for external use:
    # ------------------------------------------------------------------
    def getOneRegionDict(self):
        # first rebuild the oneRegionDict global from its table:
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: getOneRegionDict is ", self.oneRegionDict, " done."
        return self.oneRegionDict

    # ------------------------------------------------------------------
    # populate the one-region subtable with data from oneRegionDict
    # ------------------------------------------------------------------
    def populateOneRegionSubTable(self):
#        print "SIZE SIZE SIZE self.oneRegionTable.height() =", self.oneRegionTable.height()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().height() =", self.oneRegionTable.parentWidget().height()

#        print "SIZE SIZE SIZE self.oneRegionTable.width() =", self.oneRegionTable.width()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().width() =", self.oneRegionTable.parentWidget().width()
#        print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() = ", \
#              self.oneRegionTable.rowCount()

        # prevent oneRegionTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:
        self.oneRegionTable.blockSignals(True)

        # set how many rows are needed in the table:
        lTypesCount = len(self.oneRegionDict)
        self.oneRegionTable.setRowCount(lTypesCount)

        # set how many rows are present in the table:
#         print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() = ", \
#               self.oneRegionTable.rowCount()

        # get the oneRegionDict keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
#         print "___ - DEBUG ----- PIFOneRegionTable: lKeys =", lKeys

        # add additional (non-editable) table column with percentages calculated at each table content update:
        lTotalAmounts = 0.0

        for i in xrange(lTypesCount):
            lKey = lKeys[i]

            #   before, we created a QTableWidgetItem **item**, set its string value to the dict key:
            # lItem = QtGui.QTableWidgetItem( QtCore.QString("%1").arg(lKey) )
            #   the table item containing the dict key ought not to be selected/edited:
            # lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            #   this string value *was* shown in column 0 of the table widget:
            # self.oneRegionTable.setItem(i, 0, lItem)

            # create a first QTableWidgetItem and place a swatch in it:
            lItem = QtGui.QTableWidgetItem()
            # (setting the background color like this would not be a good idea,
            #   for example because it can be confused with selection highlight colors:
            #   lItem.setBackground(QtGui.QBrush(self.oneRegionDict[lKey][0]))
            # this way is much better, it generates a rectangle in the same color,
            #   and we add patterns to the color so that it can distinguish types within regions:
            #   print "lKey,self.oneRegionDict[lKey] = ", lKey,self.oneRegionDict[lKey]
            lItem.setIcon(  self.createColorIcon( self.oneRegionDict[lKey][0] )  )
            # 2011 Mitja - remove the pattern for each cell type, so that colors can show better:
            # lItem.setIcon(  self.createColorIcon( self.oneRegionDict[lKey][0], self.patternDict[int(lKey) % 10] )  )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 0 in the table:
            self.oneRegionTable.setItem(i, 0, lItem)

            # create a second QTableWidgetItem and place text (i.e. cell type name) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][1]) )
            # this goes to column 1 in the table:
            self.oneRegionTable.setItem(i, 1, lItem)

            # create a third QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][2]) )
            # this goes to column 2 in the table:
            self.oneRegionTable.setItem(i, 2, lItem)
            lTotalAmounts = lTotalAmounts + self.oneRegionDict[lKey][2]

            # create a fifth QTableWidgetItem and place a value (i.e. volume for this cell type) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][3]) )
            # this goes to column 4 in the table,
            #   because column 3 is just the automatically-computed percentage of cells per type:
            self.oneRegionTable.setItem(i, 4, lItem)



        for i in xrange(lTypesCount):
            lKey = lKeys[i]
            # create a fourth QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg( self.oneRegionDict[lKey][2] / lTotalAmounts ) )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 3 in the table:
            self.oneRegionTable.setItem(i, 3, lItem)
#
#
#         # distribute the available space according to the space requirement of each column or row:
#         # TODO TODO: all this widget layout semi/auto/resizing still @#$@*Y@ does not work...
#         w = 0
#         for i in xrange(self.oneRegionTable.columnCount()):
#             # print "column", i, "has width", self.oneRegionTable.columnWidth(i)
#             w = w + self.oneRegionTable.columnWidth(i)
#         w = w + self.oneRegionTable.verticalHeader().width()
#         w = w + self.oneRegionTable.verticalScrollBar().width()
#         # print "column and everything is", w
#
#         h = 0
#         for i in xrange(self.oneRegionTable.rowCount()):
#             # print "row", i, "has height", self.oneRegionTable.rowHeight(i)
#             h = h + self.oneRegionTable.rowHeight(i)
#         h = h + self.oneRegionTable.horizontalHeader().height()
#         h = h + self.oneRegionTable.horizontalScrollBar().height()
#         # print "column and everything is", h
# #        self.oneRegionTable.resize(w + 4, h + 4)
# #        self.oneRegionTable.parentWidget().resize(w + 4, h + 4)
#
# #        self.oneRegionTable.resizeRowsToContents()
        self.oneRegionTable.resizeColumnsToContents()


#        print "SIZE SIZE SIZE self.oneRegionTable.height() =", self.oneRegionTable.height()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().height() =", self.oneRegionTable.parentWidget().height()

#        print "SIZE SIZE SIZE self.oneRegionTable.width() =", self.oneRegionTable.width()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().width() =", self.oneRegionTable.parentWidget().width()

        # start with no table cell selected, and the user can then click to select:
        self.oneRegionTable.setCurrentCell(-1,-1)
        # self.oneRegionTable.parentWidget().resize(self.oneRegionTable.width(),self.oneRegionTable.height())

        # allow oneRegionTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:
        self.oneRegionTable.blockSignals(False)

        # print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() : done"

    # ------------------------------------------------------------------
    # remove the last row from the table, if there is more than one row:
    # ------------------------------------------------------------------
    def handleRemoveTableRow(self):
        # get the oneRegionTable keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
        lLength = len(self.oneRegionDict)
        # don't remove the last row in the dict:
        if (lLength > 1):
            lLastKey = lKeys[lLength-1]
            # remove the last entry in the oneRegionDict:
            del self.oneRegionDict[lLastKey]
            # update the table:
            self.populateOneRegionSubTable()

        # propagate the change upstream by emitting a signal, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleRemoveTableRow() : done"

    # ------------------------------------------------------------------
    # add a row to the table:
    # ------------------------------------------------------------------
    def handleAddTableRow(self):
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow FROM ", self.oneRegionDict

        # get the oneRegionTable keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
        lLength = len(self.oneRegionDict)
        # generate a new key for the oneRegionDict (just a sequential number)
        lNewKey = lKeys[lLength-1] + 1

        # generate a oneRegionDict entry:
        lThisRegionQColor = QtGui.QColor( self.oneRegionDict[lKeys[lKeys[lLength-1]]][0] )
        lNewColorR = lThisRegionQColor.red()
        lNewColorG = lThisRegionQColor.green()
        lNewColorB = lThisRegionQColor.blue()

#         lNewColorB = lNewColorB + 64
#         lNewColorR = lNewColorR + 64

        if (lNewColorR < 128) :
            lNewColorR = lNewColorR + int(random.random()*64.0)
        else:
            lNewColorR = lNewColorR - int(random.random()*64.0)

        if (lNewColorG < 128) :
            lNewColorG = lNewColorG + int(random.random()*64.0)
        else:
            lNewColorG = lNewColorG - int(random.random()*64.0)

        if (lNewColorB < 128) :
            lNewColorB = lNewColorB + int(random.random()*64.0)
        else:
            lNewColorB = lNewColorB - int(random.random()*64.0)

        # CONTINUA 
        lNewColor = QtGui.QColor( lNewColorR, \
                                  lNewColorG, \
                                  lNewColorB    )

        CDConstants.printOut( "DEBUG ----- PIFOneRegionTable: handleAddTableRow(): lNewColor = " + str(lNewColor), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- PIFOneRegionTable: handleAddTableRow(): lNewColorR = " + str(lNewColorR), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- PIFOneRegionTable: handleAddTableRow(): lNewColorG = " + str(lNewColorG), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- PIFOneRegionTable: handleAddTableRow(): lNewColorB = " + str(lNewColorB), CDConstants.DebugAll )

        self.oneRegionDict[lNewKey] = [ lNewColor, "newType"+str(lNewKey+1),  0.0, 100 ]
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow TO ", self.oneRegionDict
        # update the table:
        self.populateOneRegionSubTable()

        # propagate the change upstream by emitting a signal, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow() done."

    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def createColorIcon(self, pColor, pPattern = None):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        lBrush = QtGui.QBrush(pColor)
        if pPattern != None:
            lBrush.setStyle(pPattern)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), lBrush)
        painter.end()
        return QtGui.QIcon(pixmap)



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

        # print "___ - DEBUG ----- PIFOneRegionTable: handleTableCellChanged() pRow,pColumn =" , pRow,pColumn
        # update the dict:
        self.updateOneRegionDict()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleTableCellChanged()  done."

# end of class PIFOneRegionTable(QtGui.QWidget)
# ======================================================================





# ======================================================================
# a QTreeWidgetItem-based cell-type-level item to edit info about cell types
# ======================================================================
class CDTypesOneCellTypeItem(QtGui.QTreeWidgetItem):

#     def __init__(self, pParent, pAfter):
#     def __init__(self, pParent):

    def __init__(self, pParent, pAfter, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellTypeDict, pTotalAmounts, pIndex):
         # it is compulsory to call the parent's __init__ class right away:
        super(CDTypesOneCellTypeItem, self).__init__(pParent, pAfter)

        print "CDTypesOneCellTypeItem()  pParent, pAfter, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellTypeDict, pTotalAmounts, pIndex =", pParent, pAfter, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellTypeDict, pTotalAmounts, pIndex

        # prevent this CDTypesOneCellTypeItem from emitting any "changed" signals when
        #   updating its content programmatically:
#         self.blockSignals(True)

        self.setText(0, "Cell Type "+str(pCellTypeKey))
        self.setIcon(0, pCellTypeIcon)
        self.setText(1, str(pCellTypeDict[1]))
        self.setIcon(1, pCellTypeIcon)
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        self.cellAmountPerType = QtGui.QTreeWidgetItem(self)
        lPercentage = (pCellTypeDict[2] / pTotalAmounts) * 100.0
        self.cellAmountPerType.setText(0, str("Amount ("+str(lPercentage)+"%)"))
        self.cellAmountPerType.setIcon(0, pCellTypeIcon)
        self.cellAmountPerType.setText(1, str(pCellTypeDict[2]))
        self.cellAmountPerType.setIcon(1, pCellTypeIcon)
        self.cellAmountPerType.setFlags( self.cellAmountPerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.cellVolumePerType = QtGui.QTreeWidgetItem(self)
        self.cellVolumePerType.setText(0, str("Volume (pixels)"))
        self.cellVolumePerType.setIcon(0, pCellTypeIcon)
        self.cellVolumePerType.setText(1, str(pCellTypeDict[3]))
        self.cellVolumePerType.setIcon(1, pCellTypeIcon)
        self.cellVolumePerType.setFlags( self.cellVolumePerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )


#         self.addChild( self.cellAmountPerType )
# 
#         self.cellSizeXPerRegionItem = QtGui.QTreeWidgetItem(self.cellSizePerRegionItem)
#         self.cellSizeXPerRegionItem.setText(0, str("X"))
#         self.cellSizeXPerRegionItem.setIcon(0, pRegionIcon)
#         self.cellSizeXPerRegionItem.setText(1, str(pCellXYZSizeList[0]))
#         self.cellSizeXPerRegionItem.setIcon(1, pRegionIcon)
#         self.cellSizeXPerRegionItem.setFlags( self.cellSizeXPerRegionItem.flags() |   \
#             (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
# 
#         self.cellSizeYPerRegionItem = QtGui.QTreeWidgetItem(self.cellSizePerRegionItem)
#         self.cellSizeYPerRegionItem.setText(0, str("Y"))
#         self.cellSizeYPerRegionItem.setIcon(0, pRegionIcon)
#         self.cellSizeYPerRegionItem.setText(1, str(pCellXYZSizeList[1]))
#         self.cellSizeYPerRegionItem.setIcon(1, pRegionIcon)
#         self.cellSizeYPerRegionItem.setFlags( self.cellSizeYPerRegionItem.flags() |   \
#             (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
# 
#         self.cellSizeZPerRegionItem = QtGui.QTreeWidgetItem(self.cellSizePerRegionItem)
#         self.cellSizeZPerRegionItem.setText(0, str("Z"))
#         self.cellSizeZPerRegionItem.setIcon(0, pRegionIcon)
#         self.cellSizeZPerRegionItem.setText(1, str(pCellXYZSizeList[2]))
#         self.cellSizeZPerRegionItem.setIcon(1, pRegionIcon)
#         self.cellSizeZPerRegionItem.setFlags( self.cellSizeZPerRegionItem.flags() |   \
#             (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
# 
#         self.cellTypeUse = QtGui.QTreeWidgetItem(self)
#         self.cellTypeUse.setText(0, str("Type Use"))
#         self.cellTypeUse.setIcon(0, pRegionIcon)
#         self.cellTypeUse.setText(1, str(pRegionTypeUse))
#         self.cellTypeUse.setIcon(1, pRegionIcon)
#         self.cellTypeUse.setFlags( self.cellTypeUse.flags() &   \
#             ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
# 
#         self.oneRegionTreeWidget = CDTypesOneCellTypeItem(self)
#         self.oneRegionTreeWidget.setOneRegionDict(pRegionCellTypesDict)
#         self.oneRegionTreeWidget.populateOneRegionSubTable()
#         self.oneRegionTreeWidget.setText(0, str("Cell Types"))
#         self.oneRegionTreeWidget.setIcon(0, pRegionIcon)
#         self.oneRegionTreeWidget.setText(1, str(" "))
#         self.oneRegionTreeWidget.setIcon(1, pRegionIcon)
#         self.oneRegionTreeWidget.setFlags( self.oneRegionTreeWidget.flags() &   \
#             ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
# 
#         # explicitly connect the "oneRegionTableChangedSignal()" signal from the
#         #   self.oneRegionTreeWidget object, to our "slot" method
#         #   so that it will respond to any change in the subtable item contents:
#         answer = self.connect(self.oneRegionTreeWidget, \
#                               QtCore.SIGNAL("oneRegionTableChangedSignal()"), \
#                               self.handleOneRegionTableWidgetChanged )



        # allow this CDTypesOneCellTypeItem to emit "changed" signals now that we're done
        #   updating its content programmatically:
#         self.blockSignals(False)


# end of   class CDTypesOneCellTypeItem(QtGui.QTreeWidgetItem)
# ======================================================================





# ======================================================================
# a QTreeWidgetItem-based top level item to edit info about regions of cells
# ======================================================================
class CDTypesOneRegionTypeItem(QtGui.QTreeWidgetItem):

    def __init__(self, pParent, pAfter, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDTypesOneRegionTypeItem, self).__init__(pParent, pAfter)

        print "CDTypesOneRegionTypeItem()  self, pParent, pAfter, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex =", pParent, pAfter, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex

        self.oneRegionCellTypesDict = pRegionCellTypesDict

        lString = "Region Type " + str(pRegionKey)
        self.setText(0, lString)
        self.setIcon(0, pRegionIcon)
        self.setText(1, pRegionName)
        self.setIcon(1, pRegionIcon)
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        self.thisRegionsCellSize = QtGui.QTreeWidgetItem(self)
        self.thisRegionsCellSize.setText(0, str("Cell Size"))
        self.thisRegionsCellSize.setIcon(0, pRegionIcon)
        self.thisRegionsCellSize.setText(1, str(pCellXYZSizeList))
        self.thisRegionsCellSize.setIcon(1, pRegionIcon)
        self.thisRegionsCellSize.setFlags( self.thisRegionsCellSize.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
#         self.addChild( self.thisRegionsCellSize )

        self.thisRegionsCellSizeX = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeX.setText(0, str("X"))
        self.thisRegionsCellSizeX.setIcon(0, pRegionIcon)
        self.thisRegionsCellSizeX.setText(1, str(pCellXYZSizeList[0]))
        self.thisRegionsCellSizeX.setIcon(1, pRegionIcon)
        self.thisRegionsCellSizeX.setFlags( self.thisRegionsCellSizeX.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeY = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeY.setText(0, str("Y"))
        self.thisRegionsCellSizeY.setIcon(0, pRegionIcon)
        self.thisRegionsCellSizeY.setText(1, str(pCellXYZSizeList[1]))
        self.thisRegionsCellSizeY.setIcon(1, pRegionIcon)
        self.thisRegionsCellSizeY.setFlags( self.thisRegionsCellSizeY.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeZ = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeZ.setText(0, str("Z"))
        self.thisRegionsCellSizeZ.setIcon(0, pRegionIcon)
        self.thisRegionsCellSizeZ.setText(1, str(pCellXYZSizeList[2]))
        self.thisRegionsCellSizeZ.setIcon(1, pRegionIcon)
        self.thisRegionsCellSizeZ.setFlags( self.thisRegionsCellSizeZ.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionTypeUse = QtGui.QTreeWidgetItem(self)
        self.regionTypeUse.setText(0, str("Region Type Use"))
        self.regionTypeUse.setIcon(0, pRegionIcon)
        self.regionTypeUse.setText(1, str(pRegionTypeUse))
        self.regionTypeUse.setIcon(1, pRegionIcon)
        self.regionTypeUse.setFlags( self.regionTypeUse.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionCellTypesItem = QtGui.QTreeWidgetItem(self)
        self.regionCellTypesItem.setText(0, str("Cell Types"))
        self.regionCellTypesItem.setIcon(0, pRegionIcon)
        self.regionCellTypesItem.setText(1, str(" "))
        self.regionCellTypesItem.setIcon(1, pRegionIcon)
        self.regionCellTypesItem.setFlags( self.regionCellTypesItem.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        # how many cell type entries are needed for this region type:
        lTypesCount = self.getOneRegionCellsTypeDictElementCount()
        # get the oneRegionDict keys in order to resize the table:
        lKeys = self.getOneRegionCellsTypeDictKeys()

        # compute total of cell amounts, for percentages calculated at each tree content update:
        lTotalAmounts = 0.0
        for i in xrange(lTypesCount):
            lKey = lKeys[i]
            lTotalAmounts = lTotalAmounts + self.oneRegionCellTypesDict[lKey][2]

        lIndexInRegion = 0
        for i in xrange(lTypesCount):
            lKey = lKeys[i]

            # compute a rectangle in the same color,
            #   and modify the color so that it can distinguish cell types within a region:
            #   print "lKey,self.oneRegionDict[lKey] = ", lKey,self.oneRegionDict[lKey]
            lIcon = self.createColorIcon( self.oneRegionCellTypesDict[lKey][0] )

            # create one top-level QTreeWidgetItem representing a region type:
            lOneRegionCellTypes = self.createOneCellTypeItem( self.regionCellTypesItem, lKey, lIcon, pRegionName, self.oneRegionCellTypesDict[lKey], lTotalAmounts, lIndexInRegion )

            lIndexInRegion = lIndexInRegion + 1


    # ------------------------------------------------------------------
    # createOneCellTypeItem() returns one top-level CDTypesOneRegionTypeItem for one region type:
    # ------------------------------------------------------------------
    def createOneCellTypeItem( self, pParent, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellTypeDict, pTotalAmounts, pIndex):


        if pIndex != 0:
            lAfter = self.childAt(pParent, pIndex - 1)
        else:        
            lAfter = None

        # create a CDTypesOneRegionTypeItem and attach it to the "self" QTreeWidget:
        lTopLevelItem = CDTypesOneCellTypeItem(pParent, lAfter, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellTypeDict, pTotalAmounts, pIndex)

        return lTopLevelItem

    # end of      def createOneCellTypeItem( self, pCellTypeKey, pCellTypeIcon, pCellTypeName, pCellXYZSizeList, pRegionTypeUse, pCellTypeDict, pTotalAmounts, pIndex )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # childAt() returns one top-level QTreeWidgetItem for one region type:
    # ------------------------------------------------------------------
    def childAt(self, pParent, pIndex):
        if pParent is not None:
            return pParent.child(pIndex)
        else:
            #return self.topLevelItem(pIndex)
            return None



    # ------------------------------------------------------------------
    # getOneRegionCellsTypeDictElementCount() is a helper function,
    #   returning the number of elements in the typesDict global:
    # ------------------------------------------------------------------
    def getOneRegionCellsTypeDictElementCount(self):
        return len(self.oneRegionCellTypesDict)


    # ------------------------------------------------------------------
    # getOneRegionCellsTypeDictKeys() is a helper function,
    #   returning a list of keys from oneRegionCellTypesDict:
    # ------------------------------------------------------------------
    def getOneRegionCellsTypeDictKeys(self):
        return self.oneRegionCellTypesDict.keys()

    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def createColorIcon(self, pColor, pPattern = None):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        lBrush = QtGui.QBrush(pColor)
        if pPattern != None:
            lBrush.setStyle(pPattern)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), lBrush)
        painter.end()
        return QtGui.QIcon(pixmap)



# end of   class CDTypesOneRegionTypeItem(QtGui.QTreeWidgetItem)
# ======================================================================





# ======================================================================
# a QTreeWidget-based types editor
# ======================================================================
class CDTypesTree(QtGui.QTreeWidget):

    # ----------------------------------------
    def __init__(self, parent=None):
        super(CDTypesTree, self).__init__(parent)

        self.setHeaderLabels(("Region and Cell Types", "Value"))
#         self.header().setResizeMode(0, QtGui.QHeaderView.Stretch)
#         self.header().setResizeMode(2, QtGui.QHeaderView.Stretch)
# 
#         self.groupIcon = QtGui.QIcon()
#         self.groupIcon.addPixmap( \
#             self.style().standardPixmap(QtGui.QStyle.SP_DirClosedIcon), \
#             QtGui.QIcon.Normal, QtGui.QIcon.Off)
#         self.groupIcon.addPixmap( \
#             self.style().standardPixmap(QtGui.QStyle.SP_DirOpenIcon), \
#             QtGui.QIcon.Normal, QtGui.QIcon.On)
# 
#         self.keyIcon = QtGui.QIcon()
#         self.keyIcon.addPixmap( \
#             self.style().standardPixmap(QtGui.QStyle.SP_FileIcon))

        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: __init__() done. ----- ", CDConstants.DebugExcessive )
    # end of   def __init__(self, parent=None)
    # ----------------------------------------


    # --------------------------------------------
    def hideRegionTypeRow(self, pRowNumber):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: hideRegionTypeRow() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   hideRegionTypeRow(self, pRowNumber)
    # --------------------------------------------


    # --------------------------------------------
    def showRegionTypeRow(self, pRowNumber):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: showRegionTypeRow() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   showRegionTypeRow(self, pRowNumber)
    # --------------------------------------------


    # --------------------------------------------
    def setOneRegionTypeItem(self, pRowNumber, pColumnNumber, pItem):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: setOneRegionTypeItem() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   setOneRegionTypeItem(self, pRowNumber)
    # --------------------------------------------



    # ------------------------------------------------------------------
    # createTopRegionItem() returns one top-level CDTypesOneRegionTypeItem for one region type:
    # ------------------------------------------------------------------
    def createTopRegionItem( self, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex ):
        if pIndex != 0:
            # lAfter = self.childAt(parent, pIndex - 1)
            lAfter = self.childAt(None, pIndex - 1)
        else:        
            lAfter = None

        # create a CDTypesOneRegionTypeItem and attach it to the "self" QTreeWidget:
        lTopLevelItem = CDTypesOneRegionTypeItem(self, lAfter, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex)

        return lTopLevelItem

    # end of      def createTopRegionItem( self, pRegionKey, pRegionIcon, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # childAt() returns one top-level QTreeWidgetItem for one region type:
    # ------------------------------------------------------------------
    def childAt(self, parent, index):
        if parent is not None:
            return parent.child(index)
        else:
            return self.topLevelItem(index)




# end of   class CDTypesTree(QtGui.QTreeWidget)
# ======================================================================





# ======================================================================
# a QWidget-based control panel, in application-specific panel style
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))
#
class CDTypesEditor(QtGui.QWidget):

    def __init__(self, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDTypesEditor, self).__init__(pParent)

        # don't show this widget until it's completely ready:
        self.hide()

        # save the parent, whatever that may be:
        lParent = pParent

        # init - windowing GUI stuff:
        #
        self.__miInitEditorGUIGeneralBehavior()

        # init - create central table, set it up and show it inside the panel:
        #
        self.theTypesTree = CDTypesTree()
        self.layout().addWidget(self.__miInitCentralTreeWidget())

        # 2011 - Mitja: to control the "PIFF Generation mode",
        #   we add a set of radio-buttons:
        self.theControlsForPIFFGenerationMode = CDControlPIFFGenerationMode()
        self.layout().addWidget(self.theControlsForPIFFGenerationMode)

        # init - create types dict, populate it with empty values:
        #
        self.typesDict = dict()
        self.debugRegionDict()
#        self.populateTreeWithTypesDict()
        self.debugRegionDict()

        # the QObject.connect( QObject, SIGNAL(), QObject, SLOT() ) function
        #   creates a connection of the given type from the signal in the sender object
        #   to the slot method in the receiver object,
        #   and it returns true if the connection succeeds; otherwise it returns false.

#         # the QTreeWidget type emits a signal named "itemSelectionChanged()"
#         #   whenever the selection changes:  TODO TODO TODO do we need this?
#         answer = self.connect(self.theTypesTree, \
#                               QtCore.SIGNAL("itemSelectionChanged()"), \
#                               self.handleTableItemSelectionChanged )
#         # print "004 - DEBUG ----- CDTypesEditor: self.connect() =", answer

        # connect the "itemChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.theTypesTree.itemChanged.connect(self.handleTreeItemChanged)

        # to be used as "flag name" when switching to Image Sequence use and back:
        self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack = ""

        # print "005 - DEBUG ----- CDTypesEditor: __init__(): done"


    # ------------------------------------------------------------------
    # define functions to initialize this panel:
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init - windowing GUI stuff:
    # ------------------------------------------------------------------
    def __miInitEditorGUIGeneralBehavior(self):

        # how will the CDTypesEditor look like:
        #   this title won't appear anywhere if the CDTypesEditor widget is included in a QDockWidget:
        self.setWindowTitle("Types of Cells and Regions")

        self.setMinimumSize(200,200)
        # self.setMinimumSize(466,322)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # NOTE: the x,y is NOT the top-left edge of the window,
        #    but of its **content** (excluding the menu bar, toolbar, etc.
        # self.setGeometry(750,60,480,480)

        # the following is only useful to fix random placement at initialization
        #   *if* we use this panel as stand-alone, without including it in windows etc.
        #   These are X,Y *screen* coordinates (INCLUDING menu bar, etc.),
        #   where X,Y=0,0 is the top-left corner of the screen:
#         pos = self.pos()
#         pos.setX(800)
#         pos.setY(30)
#         self.move(pos)

        # QHBoxLayout layout lines up widgets horizontally:
        self.tableWindowMainLayout = QtGui.QVBoxLayout()
        self.tableWindowMainLayout.setContentsMargins(2,2,2,2)
        self.tableWindowMainLayout.setSpacing(4)
        self.tableWindowMainLayout.setAlignment( \
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setLayout(self.tableWindowMainLayout)

        #
        # QWidget setup (2) - more windowing GUI setup:
        #
# 
#         miDialogsWindowFlags = QtCore.Qt.WindowFlags()
#         # this panel is a so-called "Tool" (by PyQt and Qt definitions)
#         #    we'd use the Tool type of window, except for this oh-so typical Qt bug:
#         #    http://bugreports.qt.nokia.com/browse/QTBUG-6418
#         #    i.e. it defines a system-wide panel which shows on top of *all* applications,
#         #    even when this application is in the background.
#         miDialogsWindowFlags = QtCore.Qt.Window
#         #    so we use a plain QtCore.Qt.Window type instead:
#         # miDialogsWindowFlags = QtCore.Qt.Window
#         #    add a peculiar WindowFlags combination to have no close/minimize/maxize buttons:
#         miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
#         miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint
# #        miDialogsWindowFlags |= QtCore.Qt.WindowMinimizeButtonHint
# #        miDialogsWindowFlags |= QtCore.Qt.WindowStaysOnTopHint
#         self.setWindowFlags(miDialogsWindowFlags)
# 
        # 1. The widget is not modal and does not block input to other widgets.
        # 2. If widget is inactive, the click won't be seen by the widget.
        #    (it does NOT work as Qt docs says it would on Mac OS X: click-throughs don't get disabled)
        # 3. The widget can choose between alternative sizes for widgets to avoid clipping.
        # 4. The native Carbon size grip should be opaque instead of transparent.
        self.setAttribute(QtCore.Qt.NonModal  | \
                          QtCore.Qt.WA_MacNoClickThrough | \
                          QtCore.Qt.WA_MacVariableSize | \
                          QtCore.Qt.WA_MacOpaqueSizeGrip )

        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
       
        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)

        self.show()

    # end of        def __miInitEditorGUIGeneralBehavior(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def __miInitCentralTreeWidget(self):



        # -------------------------------------------
        # place the table in the Panel's vbox layout:
        # one cell information widget, containing a vbox layout, in which to place the table:
        tableContainerWidget = QtGui.QWidget()

        # this topFiller part is cosmetic and could safely be removed:
        topFiller = QtGui.QWidget()
        topFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        # this infoLabel part is cosmetic and could safely be removed,
        #     unless useful info is provided here:
        self.infoLabel = QtGui.QLabel()
        self.infoLabel.setText("<i>colors</i> in the Cell Scene correspond to <i>region types</i>")
        self.infoLabel.setAlignment = QtCore.Qt.AlignCenter
#         self.infoLabel.setLineWidth(3)
#         self.infoLabel.setMidLineWidth(3)
        self.infoLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)

        # this bottomFiller part is cosmetic and could safely be removed:
        bottomFiller = QtGui.QWidget()
        bottomFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        # vbox.addWidget(topFiller)
        vbox.addWidget(self.theTypesTree)
        vbox.addWidget(self.infoLabel)
        # vbox.addWidget(bottomFiller)
        # finally place the complete layout in a QWidget and return it:
        tableContainerWidget.setLayout(vbox)
        return tableContainerWidget

    # end of   def __miInitCentralTreeWidget(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def NOTUSED__miInitCentralTableWidget(self):
    
        # create a tree widget for the main "central widget" area of the window:
        self.regionsMainTreeWidget = QtGui.QTreeWidget()
        # tree item selection set to: "When the user selects an item,
        # any already-selected item becomes unselected, and the user cannot unselect
        # the selected item by clicking on it." :
        self.regionsMainTreeWidget.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        # clicking on a tree item selects a single item:
        self.regionsMainTreeWidget.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        # actions which will initiate item editing: all of them (double-clicking, etc.) :
        self.regionsMainTreeWidget.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)

        # don't show the tree's rown numbers in the "verticalHeader" on the left side, since its
        #   numbering is inconsistent with the 1st column, which contains cell type ID numbers:
        # self.regionsMainTreeWidget.verticalHeader().hide()
        # show the table's verticalHeader, but remove all its labels:
        self.regionsMainTreeWidget.setHeaderLabels( \
             (" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," ") )

        self.regionsMainTreeWidget.setColumnCount(6)
        # the QTreeQidget doesn't have a setRowCount() function:
        # self.regionsMainTreeWidget.setRowCount(6)
        self.regionsMainTreeWidget.setHeaderLabels( \
             ("#", "Color", "Region", "Cell\nSize", "Use", "Cell Types in Region") )
        self.regionsMainTreeWidget.header().setResizeMode(0, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setResizeMode(1, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setResizeMode(2, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setResizeMode(3, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setResizeMode(4, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setResizeMode(5, QtGui.QHeaderView.Interactive)
        self.regionsMainTreeWidget.header().setStretchLastSection(True)
        # self.regionsMainTreeWidget.header().resizeSection(1, 180)
        self.regionsMainTreeWidget.show()

#        self.regionsMainTreeWidget.setMinimumSize(QtCore.QSize(245,0))
        self.regionsMainTreeWidget.setLineWidth(1)
        self.regionsMainTreeWidget.setMidLineWidth(1)
        self.regionsMainTreeWidget.setFrameShape(QtGui.QFrame.Panel)
        self.regionsMainTreeWidget.setFrameShadow(QtGui.QFrame.Plain)
        self.regionsMainTreeWidget.setObjectName("regionsMainTreeWidget")

        # -------------------------------------------
        # place the table in the Panel's vbox layout:
        # one cell information widget, containing a vbox layout, in which to place the table:
        tableContainerWidget = QtGui.QWidget()

        # this topFiller part is cosmetic and could safely be removed:
        topFiller = QtGui.QWidget()
        topFiller.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Expanding)

        # this infoLabel part is cosmetic and could safely be removed,
        #     unless useful info is provided here:
        self.infoLabel = QtGui.QLabel()
        self.infoLabel.setText("<i>colors</i> in the Cell Scene correspond to <i>region types</i>")
        self.infoLabel.setAlignment = QtCore.Qt.AlignCenter
        self.infoLabel.setLineWidth(3)
        self.infoLabel.setMidLineWidth(3)
        self.infoLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)

        # this bottomFiller part is cosmetic and could safely be removed:
        bottomFiller = QtGui.QWidget()
        bottomFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        # vbox.addWidget(topFiller)
        vbox.addWidget(self.regionsMainTreeWidget)
        vbox.addWidget(self.infoLabel)
        # vbox.addWidget(bottomFiller)
        # finally place the complete layout in a QWidget and return it:
        tableContainerWidget.setLayout(vbox)
        return tableContainerWidget

    # end of   def NOTUSED__miInitCentralTableWidget(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # now define fuctions that actually do something with data:
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # assign new dict content to the typesDict global
    # ------------------------------------------------------------------
    def setTypesDict(self, pDict):
        self.typesDict = None
        self.typesDict = pDict
        self.debugRegionDict()
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor:  self.setTypesDict() to "+str(self.typesDict)+" done. ----- ", CDConstants.DebugVerbose )



    # ------------------------------------------------------------------
    # rebuild the typesDict global from its PIFOneRegionTable objects:
    # ------------------------------------------------------------------
    def __updateRegionsDictFromGUI(self):
        # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() from ", self.typesDict
        return
        # set how many rows are needed in the table:
        lRegionsCount = self.theTypesTree.rowCount()

        # get the typesDict keys in order to access elements:
        lKeys = self.getRegionsDictKeys()

        # parse each table rown separately to build a typesDict entry:
        for i in xrange(lRegionsCount):
            # the key is NOT retrieved from the first theTypesTree column:
            # lKey = self.theTypesTree.item(i, 0)
            lKey = lKeys[i]
            # the color remains the same (table color is not editable (yet?) )
            lColor = self.typesDict[lKeys[i]][0]

            # the region name is retrieved from the 3.rd theTypesTree column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() \n"
            # print "      i, self.theTypesTree.item(i, 2) ", i, self.theTypesTree.item(i, 2)
            lRegionName = str ( self.theTypesTree.item(i, 2).text() )
#
#             # the region cell size is retrieved from the 4.th theTypesTree column:
#             # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() \n"
#             # print "      i, self.theTypesTree.item(i, 3) ", i, self.theTypesTree.item(i, 3)
#             lRegionCellSize = int ( self.theTypesTree.item(i, 3).text() )

            # the region cell sizes are retrieved from the 4.th theTypesTree column:
            CDConstants.printOut ( "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() ", CDConstants.DebugAll )
            CDConstants.printOut ( "      i, self.theTypesTree.cellWidget(i, 3) " + str(i) + " " + str( self.theTypesTree.cellWidget(i, 3) ), CDConstants.DebugAll )
            lRegionBlockCellSizes = self.theTypesTree.cellWidget(i, 3)

            # the region use is retrieved from the 5.th theTypesTree column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() \n"
            # print "      i, self.theTypesTree.item(i, 4) ", i, self.theTypesTree.item(i, 4)
            lRegionInUse = int ( self.theTypesTree.item(i, 4).text() )

            # the cell types dict for each region is obtained from the PIFOneRegionTable widget
            #   which is in the 6.th  theTypesTree column:
            lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)

            # rebuild the i-th dict entry:
            self.typesDict[lKey] = [ lColor, lRegionName, \
                                       lRegionBlockCellSizes.getOneRegionDict(), \
                                       lRegionInUse, \
                                       lOneRegionTableWidget.getOneRegionDict() ]

        # print "___ - DEBUG ----- CDTypesEditor: self.__updateRegionsDictFromGUI() to ", self.typesDict, " done."


    # ------------------------------------------------------------------
    # retrieve the up-to-date typesDict for external use:
    # ------------------------------------------------------------------
    def getTypesDict(self):
        # first rebuild the typesDict global from its PIFOneRegionTable objects:
        self.__updateRegionsDictFromGUI()
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor:  self.getTypesDict() will now return self.typesDict=="+str(self.typesDict) , CDConstants.DebugVerbose )
        return self.typesDict



    # ------------------------------------------------------------------
    # populate the table widget with data from the typesDict global
    # ------------------------------------------------------------------
    def populateTreeWithTypesDict(self):
#         print "___ - DEBUG ----- CDTypesEditor: populateTreeWithTypesDict() = ", \
#               self.theTypesTree.rowCount(), " rows to ", self.getRegionsDictElementCount(), "cells."

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to resize the table:
        keys = self.getRegionsDictKeys()
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False
        
        lIndexInTree = 0

        for i in xrange(self.getRegionsDictElementCount()):

            # prepare all data for one top-level QTreeWidgetItem representing a region type:
            
            # 1.st parameter, set its string value to the dict key for region i:
            lRegionKey = QtCore.QString("%1").arg(keys[i])
            # 2.nd parameter, an icon with the region color:
            lRegionIcon = self.createColorIcon( self.typesDict[keys[i]][0] )
            # 3.rd parameter, the region name from typesDict:
            lRegionName = QtCore.QString("%1").arg(self.typesDict[keys[i]][1])

            # 4.th parameter, a list the cell sizes from typesDict in it:
            lCellXYZSizeList = list( ( -1, -2, -3) )
            CDConstants.printOut ( "self.typesDict[keys[i]][2] = " + str(self.typesDict[keys[i]][2]), CDConstants.DebugAll )
            for j in xrange(3):
                CDConstants.printOut (  "at i = " +str(i)+ ", j = " + str(j) + "self.typesDict[keys[i]][2][j] = " + str(self.typesDict[keys[i]][2][j]), CDConstants.DebugAll )
                lCellXYZSizeList[j] = self.typesDict[keys[i]][2][j]

            # 5.th parameter, the region type use (how many regions in the scene are of this type) from typesDict:
            lRegionTypeUse = QtCore.QString("%1").arg(self.typesDict[keys[i]][3])

            # 6.th parameter, a dict with all cell type data for this region type:
            lRegionCellTypesDict = dict()
            CDConstants.printOut ( "self.typesDict[keys[i]][4] =" + str( self.typesDict[keys[i]][4] ), CDConstants.DebugAll )
            for j in xrange( len(self.typesDict[keys[i]][4]) ):
                CDConstants.printOut ( "at i ="+str( i )+", j ="+str( j )+"self.typesDict[keys[i]][4][j] ="+str( self.typesDict[keys[i]][4][j] ), CDConstants.DebugAll )
                lRegionCellTypesDict[j] = self.typesDict[keys[i]][4][j]

            # create one top-level QTreeWidgetItem representing a region type:
            lChild = self.theTypesTree.createTopRegionItem( lRegionKey, lRegionIcon, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lIndexInTree )

            lIndexInTree = lIndexInTree + 1

        # end of  for i in xrange(self.getRegionsDictElementCount())


        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)


        return
    # end of  def populateTreeWithTypesDict(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def NOTUSED__populateTableWithTypesDict(self):
    

        for i in xrange(self.getRegionsDictElementCount()):


            # create 1.st QTableWidgetItem **item**, set its string value to the dict key:
            lItem = QtGui.QTableWidgetItem( QtCore.QString("%1").arg(keys[i]) )
            # the table item containing the dict key ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this string value is shown in column 0 of the table widget:
            self.regionsMainTreeWidget.setItem(i, 0, lItem)

            # create 2.nd QTableWidgetItem and place a color rectangle in it:
            lItem = QtGui.QTableWidgetItem()
            lItem.setIcon(self.createColorIcon( self.typesDict[keys[i]][0]) )
            # print "self.typesDict[keys[i]][0] =", self.typesDict[keys[i]][0]
            # the table item containing the region color ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 1 in the table:
            self.regionsMainTreeWidget.setItem(i, 1, lItem)
            # clicking on this item selects a single item:

            # create 3.rd QTableWidgetItem and place the region name from typesDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.typesDict[keys[i]][1]) )
            # print "self.typesDict[keys[i]][1] =", self.typesDict[keys[i]][1]
            # the table item containing the text from typesDict ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 2 in the table:
            self.regionsMainTreeWidget.setItem(i, 2, lItem)

#
#             # create a 4.th QTableWidgetItem and place the cell size from typesDict in it:
#             lItem = QtGui.QTableWidgetItem( \
#                        QtCore.QString("%1").arg(self.typesDict[keys[i]][2]) )
#             # print "self.typesDict[keys[i]][2] =", self.typesDict[keys[i]][2]
#             # this goes to column 3 in the table:
#             self.regionsMainTreeWidget.setItem(i, 3, lItem)




            # create a 4.th widget (not a simple QTableWidgetItem) from the CDTableOfBlockCellSizes class,
            #   and place the cell sizes from typesDict in it:
            lOneRegionTableWidget = CDTableOfBlockCellSizes(self)
            # populate it with data from the current row in the main regionsMainTreeWidget:
            aList = list( ( -1, -2, -3) )
            CDConstants.printOut ( "self.typesDict[keys[i]][2] = " + str(self.typesDict[keys[i]][2]), CDConstants.DebugAll )
            for j in xrange(3):
                CDConstants.printOut (  "at i = " +str(i)+ ", j = " + str(j) + "self.typesDict[keys[i]][2][j] = " + str(self.typesDict[keys[i]][2][j]), CDConstants.DebugAll )
                aList[j] = self.typesDict[keys[i]][2][j]
            lOneRegionTableWidget.setOneRegionBlockCellSizes(aList)
            lOneRegionTableWidget.populateOneRegionBlockCellSizesTable()
            # this goes to column 3 in the table:
            self.regionsMainTreeWidget.setCellWidget(i, 3, lOneRegionTableWidget)

            # explicitly connect the "oneRegionTableChangedSignal()" signal from the
            #   lOneRegionTableWidget object, to our "slot" method
            #   so that it will respond to any change in the subtable item contents:
            answer = self.connect(lOneRegionTableWidget, \
                                  QtCore.SIGNAL("oneBlockCellDimChangedSignal()"), \
                                  self.handleOneRegionTableWidgetChanged )

            # this goes to the last column in the table:
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.height()
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.width()
            # first resize the table row to PIFOneRegionTable's height,
            #   then assign the widget as content to this element in the row,
            #   since resizing after assigning would not seem to work:
#             self.regionsMainTreeWidget.verticalHeader().resizeSection(i, lOneRegionTableWidget.height())
#             self.regionsMainTreeWidget.header().resizeSection(i, lOneRegionTableWidget.width())








            # create a 5.th QTableWidgetItem and place the region use from typesDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.typesDict[keys[i]][3]) )
            # print "self.typesDict[keys[i]][3] =", self.typesDict[keys[i]][3]
            # the table item containing the text from typesDict ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 4 in the table:
            self.regionsMainTreeWidget.setItem(i, 4, lItem)




            # create a 6.th widget (not a simple QTableWidgetItem) from the PIFOneRegionTable class:
            lOneRegionTableWidget = PIFOneRegionTable(self)
            # create a PIFOneRegionTable widget and populate it with data
            #   from the current row in the main regionsMainTreeWidget:
            aDict = dict()
            # print "self.typesDict[keys[i]][4] =", self.typesDict[keys[i]][4]
            for j in xrange( len(self.typesDict[keys[i]][4]) ):
                # print "at i =", i, ", j =", j, "self.typesDict[keys[i]][4][j] =", self.typesDict[keys[i]][4][j]
                aDict[j] = self.typesDict[keys[i]][4][j]
            lOneRegionTableWidget.setOneRegionDict(aDict)
            lOneRegionTableWidget.populateOneRegionSubTable()
            self.regionsMainTreeWidget.setCellWidget(i, 5, lOneRegionTableWidget)
            # explicitly connect the "oneRegionTableChangedSignal()" signal from the
            #   lOneRegionTableWidget object, to our "slot" method
            #   so that it will respond to any change in the subtable item contents:
            answer = self.connect(lOneRegionTableWidget, \
                                  QtCore.SIGNAL("oneRegionTableChangedSignal()"), \
                                  self.handleOneRegionTableWidgetChanged )

            # this goes to the last column in the table:
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.height()
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.width()
            # first resize the table row to PIFOneRegionTable's height,
            #   then assign the widget as content to this element in the row,
            #   since resizing after assigning would not seem to work:
            self.regionsMainTreeWidget.verticalHeader().resizeSection(i, lOneRegionTableWidget.height())
            self.regionsMainTreeWidget.header().resizeSection(i, lOneRegionTableWidget.width())



            # finally determine if the table row at index i is to be visible or not, from region use:
            # print "self.typesDict[keys[i]][3] =", self.typesDict[keys[i]][3]
            if self.typesDict[keys[i]][3] < 1:
                self.regionsMainTreeWidget.hideRegionTypeRow(i)
            else:
                self.regionsMainTreeWidget.showRegionTypeRow(i)
                lThereAreRegionRowsInUse = True

#         if lThereAreRegionRowsInUse is False:
#             self.regionsMainTreeWidget.hide()
#         else:
#             self.regionsMainTreeWidget.show()




#         self.regionsMainTreeWidget.resizeRowsToContents()
        # we have to "resize columns to contents" here, otherwise each column will
        #   be as wide as the widest element (maybe?)
        self.regionsMainTreeWidget.resizeColumnsToContents()
        # start with no table cell selected, and the user can then click to select:
        self.regionsMainTreeWidget.setCurrentCell(-1,-1)

        # allow regionsMainTreeWidget to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.regionsMainTreeWidget.blockSignals(False)

        # print "___ - DEBUG ----- CDTypesEditor: populateTableWithTypesDict() : done"


    # end of  def populateTableWithTypesDict(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # populate the table widget with data from the typesDict global
    # ------------------------------------------------------------------
    def updateRegionUseOfTableElements(self, pColor, pHowManyInUse):

        lColor = QtGui.QColor(pColor)

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to update the table's elements accordingly:
        lKeys = self.getRegionsDictKeys()
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False

        for i in xrange(self.getRegionsDictElementCount()):

            CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.typesDict[lKeys[i=="+str(i)+"]] ="+str(self.typesDict[lKeys[i]]), CDConstants.DebugAll )
            CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.typesDict[lKeys[i]][0] ="+str(self.typesDict[lKeys[i]][0]), CDConstants.DebugAll )
            if self.typesDict[lKeys[i]][0].rgba() == lColor.rgba() :
                CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.typesDict[lKeys[i]][3] ="+str(self.typesDict[lKeys[i]][3]), CDConstants.DebugAll )
                self.typesDict[lKeys[i]][3] = pHowManyInUse
                CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.typesDict[lKeys[i]][3] ="+str(self.typesDict[lKeys[i]][3]), CDConstants.DebugAll )

                # create a QTableWidgetItem and in it place the updated region use from typesDict:
                lItem = QtGui.QTableWidgetItem( \
                           QtCore.QString("%1").arg(self.typesDict[lKeys[i]][3]) )
                CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.typesDict[lKeys[i]][3] ="+str(self.typesDict[lKeys[i]][3]), CDConstants.DebugAll )
                # the table item containing the text from typesDict ought not to be selected/edited:
                lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
                # this goes to column 4 in the table:
                self.theTypesTree.setOneRegionTypeItem(i, 4, lItem)

            # finally determine if the current row at index i is to be visible or not:
            if self.typesDict[lKeys[i]][3] < 1:
                self.theTypesTree.hideRegionTypeRow(i)
                CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.theTypesTree.hideRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )
            else:
                self.theTypesTree.showRegionTypeRow(i)
                lThereAreRegionRowsInUse = True
                CDConstants.printOut( "___ updateRegionUseOfTableElements()  ----- self.theTypesTree.showRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )

#         if lThereAreRegionRowsInUse is False:
#             self.theTypesTree.hide()
#         else:
#             self.theTypesTree.show()

        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)


    # end of  def updateRegionUseOfTableElements(self, pColor, pHowManyInUse).
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def updateTableOfTypesForImageSequenceOn(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor: updateTableOfTypesForImageSequenceOn() starting ----- ", CDConstants.DebugExcessive )

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to update the table's elements accordingly:
        lKeys = self.getRegionsDictKeys()

        # the *only*  row to be shown is this one:
        lColorForCellSeedsInImageSequence = QtGui.QColor(QtCore.Qt.magenta)
        lTypeNameForCellSeedsInImageSequence = str("magentaType")

        lOneRegionTableWidget = None
        for i in xrange(self.getRegionsDictElementCount()):

#             CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- self.typesDict[lKeys[i=="+str(i)+"]] ="+str(self.typesDict[lKeys[i]]), CDConstants.DebugAll )
#             CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- self.typesDict[lKeys[i]][0] ="+str(self.typesDict[lKeys[i]][0]), CDConstants.DebugAll )
            # determine if the current row at index i is to be visible or not:
            if self.typesDict[lKeys[i]][0].rgba() == lColorForCellSeedsInImageSequence.rgba() :
                # get each used cell type's name and target volume size:
                lCellTypeName = self.typesDict[lKeys[i]][4][0][1]
                if (lCellTypeName != lTypeNameForCellSeedsInImageSequence):
                    # if the cell type name has already been modified by the user, don't change it:
                    lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- keeping self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )
                else:
                    # if the cell type name has NOT been modified by the user, change it to something useful for Image Sequence use:
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- storing self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )
                    self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack = lCellTypeName
                    self.typesDict[lKeys[i]][4][0][1] = "cellSeedType"
                    # modify the name in the specific region's table widget itself:
                    lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)
                    lOneRegionTableTypeNameItem = lOneRegionTableWidget.oneRegionTable.item(0, 1)
                    lOneRegionTableTypeNameItem.setText("cellSeedType")
                    
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- NOW self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )

                self.theTypesTree.showRegionTypeRow(i)
                CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- self.theTypesTree.showRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )
                lThereAreRegionRowsInUse = True
            else:
                self.theTypesTree.hideRegionTypeRow(i)
                CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- self.theTypesTree.hideRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )


        # show the entire table content again:
        #    hide the vertical header:
        self.theTypesTree.verticalHeader().hide()
        #    restore the original horizontal header labels:
        self.theTypesTree.setHeaderLabels( \
             ("#", "Color", "Region", "Cell\nSize", "Use", "Cell Seed Types in Volume") )
        # hide all columns except for the last one, for types inside the volume:
        self.theTypesTree.hideColumn(0)
        self.theTypesTree.hideColumn(1)
        self.theTypesTree.hideColumn(2)
        self.theTypesTree.hideColumn(3)
        self.theTypesTree.hideColumn(4)
        self.theTypesTree.showColumn(5)
        # same adjustments for the only used sub- table widget:
        if (lOneRegionTableWidget != None):
            lOneRegionTableTypeNameItem = lOneRegionTableWidget.oneRegionTable.setHorizontalHeaderLabels( (" ", "Cell Type", "Amount", "Fraction", "Number of Seed Pixels") )
            lOneRegionTableWidget.oneRegionTable.showColumn(0)
            lOneRegionTableWidget.oneRegionTable.showColumn(1)
            lOneRegionTableWidget.oneRegionTable.hideColumn(2)
            lOneRegionTableWidget.oneRegionTable.hideColumn(3)
            lOneRegionTableWidget.oneRegionTable.showColumn(4)
            lOneRegionTableWidget.oneRegionTable.horizontalHeader().setStretchLastSection(True)


        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)


        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor: updateTableOfTypesForImageSequenceOn() ending. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOn(self).
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def updateTableOfTypesForImageSequenceOff(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor: updateTableOfTypesForImageSequenceOff() starting ----- ", CDConstants.DebugExcessive )

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to update the table's elements accordingly:
        lKeys = self.getRegionsDictKeys()

        # the only row that was to be shown in Image Sequence was this one:
        lColorForCellSeedsInImageSequence = QtGui.QColor(QtCore.Qt.magenta)
        lTypeNameForCellSeedsInImageSequence = str("magentaType")

        # the table rows to be shown are those with regions used by the PIFF scene:
        lOneRegionTableWidget = None
        for i in xrange(self.getRegionsDictElementCount()):

            if self.typesDict[lKeys[i]][0].rgba() == lColorForCellSeedsInImageSequence.rgba() :
                if (self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack != ""):
                    # if the cell type name had already been modified by the user, restore it:
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- removing self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )
                    self.typesDict[lKeys[i]][4][0][1] = self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack
                    # modify the name in the specific region's table widget itself:
                    lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)
                    lOneRegionTableTypeNameItem = lOneRegionTableWidget.oneRegionTable.item(0, 1)
                    lOneRegionTableTypeNameItem.setText(self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack)
                    # reset the "flag" name:
                    self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack = ""
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- keeping self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )
                else:
                    # if the cell type name has NOT been modified by the user, change it to something useful for Image Sequence use:
                    lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)
                    CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOn()  ----- restored self.typesDict[lKeys[i=="+str(i)+"]][4][0][1] ="+str(self.typesDict[lKeys[i]][4][0][1]), CDConstants.DebugAll )


            CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOff()  ----- self.typesDict[lKeys[i=="+str(i)+"]] ="+str(self.typesDict[lKeys[i]]), CDConstants.DebugAll )
            CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOff()  ----- self.typesDict[lKeys[i]][0] ="+str(self.typesDict[lKeys[i]][0]), CDConstants.DebugAll )
            # determine if the current row at index i is to be visible or not:
            if self.typesDict[lKeys[i]][3] < 1:
                self.theTypesTree.hideRegionTypeRow(i)
                CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOff()  ----- self.theTypesTree.hideRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )
            else:
                self.theTypesTree.showRegionTypeRow(i)
                CDConstants.printOut( "___ updateTableOfTypesForImageSequenceOff()  ----- self.theTypesTree.showRegionTypeRow(i=="+str(i)+")", CDConstants.DebugAll )



        # show the entire table content again:
        #    show the vertical header:
        self.theTypesTree.verticalHeader().show()
        #    restore the original horizontal header labels:
        self.theTypesTree.setHeaderLabels( \
             ("#", "Color", "Region", "Cell\nSize", "Use", "Cell Types in Region") )
        # show all columns:
        self.theTypesTree.showColumn(0)
        self.theTypesTree.showColumn(1)
        self.theTypesTree.showColumn(2)
        self.theTypesTree.showColumn(3)
        self.theTypesTree.showColumn(4)
        self.theTypesTree.showColumn(5)
        # same adjustments for the only used sub- table widget:
        if (lOneRegionTableWidget != None):
            lOneRegionTableTypeNameItem = lOneRegionTableWidget.oneRegionTable.setHorizontalHeaderLabels( (" ", "Cell\nType", "Amount", "Fraction", "Volume") )
            lOneRegionTableWidget.oneRegionTable.showColumn(0)
            lOneRegionTableWidget.oneRegionTable.showColumn(1)
            lOneRegionTableWidget.oneRegionTable.showColumn(2)
            lOneRegionTableWidget.oneRegionTable.showColumn(3)
            lOneRegionTableWidget.oneRegionTable.showColumn(4)
            lOneRegionTableWidget.oneRegionTable.horizontalHeader().setStretchLastSection(True)

        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)

        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor: updateTableOfTypesForImageSequenceOff() ending. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOff(self).
    # ------------------------------------------------------------------







    # ------------------------------------------------------------------
    # debugRegionDict() is a debugging aid function to print out information
    #   about the regionDict global
    # ------------------------------------------------------------------
    def debugRegionDict(self):
        CDConstants.printOut("--------------------------------------------- CDTypesEditor.debugRegionDict() end", CDConstants.DebugExcessive )
        lCount = self.getRegionsDictElementCount()
        lKeys = self.getRegionsDictKeys()
        print " CDTypesEditor class, regionDict global: "
        print " "
        print " table rows =", lCount
        print " table keys =", lKeys
        print " table elements per row =", self.getRegionsDictMaxRowCount()
       
        print " table row content: "
        for i in xrange(lCount):
            print "i =                                     =", i
            print "key =                       lKeys[i]     =", lKeys[i]
            print "color =    self.typesDict[lKeys[i]][0] =", self.typesDict[lKeys[i]][0]
            print "region name =   self.typesDict[lKeys[i]][1] =", self.typesDict[lKeys[i]][1]
            print "cellsize = self.typesDict[lKeys[i]][2] =", self.typesDict[lKeys[i]][2]
            for j in xrange(3):
                print "cellsizes -- at i =", i, ", j =", j, "self.typesDict[lKeys[i]][2][j] =", self.typesDict[lKeys[i]][2][j]
            print "use =      self.typesDict[lKeys[i]][3] =", self.typesDict[lKeys[i]][3]
            for j in xrange( len(self.typesDict[lKeys[i]][4]) ):
                print "at i =", i, ", j =", j, "self.typesDict[lKeys[i]][4][j] =", \
                      self.typesDict[lKeys[i]][4][j]
        CDConstants.printOut("--------------------------------------------- CDTypesEditor.debugRegionDict() end", CDConstants.DebugExcessive )

    # ------------------------------------------------------------------
    # getRegionsDictKeys() is a helper function for populateTreeWithTypesDict(),
    #   it just returns the keys in the typesDict global:
    # ------------------------------------------------------------------
    def getRegionsDictKeys(self):
        return self.typesDict.keys()

    # ------------------------------------------------------------------
    # getRegionsDictElementCount() is a helper function for populateTreeWithTypesDict(),
    #   it just returns number of elements in the typesDict global:
    # ------------------------------------------------------------------
    def getRegionsDictElementCount(self):
        return len(self.typesDict)

    # ------------------------------------------------------------------
    # TODO TODO TODO: maybe we don't need this function???   :
    # ------------------------------------------------------------------
    # getRegionsDictMaxRowCount() is a helper function for populateTreeWithTypesDict(),
    #   it just returns the maximum number of elements in any typesDict entry
    def getRegionsDictMaxRowCount(self):
        maxRowCount = 0
        keys = self.getRegionsDictKeys()
        # see how many regions are present at most in the table:
        for i in xrange(self.getRegionsDictElementCount()):
            howMany = len(self.typesDict[keys[i]])
            if howMany > maxRowCount:
                maxRowCount = howMany
            # print "len(self.typesDict[keys[i==",i,"]]) = ", howMany

        # print "___ - DEBUG ----- CDTypesEditor: getRegionsDictMaxRowCount() =", \
        #       howMany
        return maxRowCount


#
#     # ------------------------------------------------------------------
#     # handle mouse click events in table elements:
#     # ------------------------------------------------------------------
#     def handleTableItemSelectionChanged(self):
#         lSelectionModel = self.regionsMainTreeWidget.selectionModel()
#         print "___ - DEBUG ----- CDTypesEditor: handleTableItemSelectionChanged() lSelectionModel =" , lSelectionModel, " done."
#
#         # TODO if any action is necessary
#         #
#         #if len(lSelectionModel.selectedRows()):
#         #    self.deleteCellTypeButton.setEnabled(True)
#         #else:
#         #    self.deleteCellTypeButton.setEnabled(False)


    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def createColorIcon(self, color):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), color)
        painter.end()
        return QtGui.QIcon(pixmap)


    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from the tree's "plain type" items
    #   (this method is not called when changes occur in table cells built with setCellWidget() )
    # ------------------------------------------------------------------
    def handleTreeItemChanged(self,pItem,pColumn):

        print "___ - DEBUG ----- CDTypesEditor: handleTreeItemChanged() pItem,pColumn =" , pItem,pColumn
        # update the dict:
        self.__updateRegionsDictFromGUI()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from table cells built with setCellWidget() :
    # ------------------------------------------------------------------
    def handleOneRegionTableWidgetChanged(self):

        # update the dict:
        self.__updateRegionsDictFromGUI()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

        print "___ - DEBUG ----- CDTypesEditor: handleOneRegionTableWidgetChanged() done."







# end of class CDTypesEditor(QtGui.QWidget)
# ======================================================================














# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDTypesEditor) case it is simply used for testing:
# ======================================================================
if __name__ == '__main__':


    print "001 - DEBUG - mi __main__ xe 01"
    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    print "002 - DEBUG - mi __main__ xe 02"

    # the cell type/colormap panel:
    mainPanel = CDTypesEditor()

    print "003 - DEBUG - mi __main__ xe 03"

    # temporarily assign colors/names to a dictionary:
    #   place all used colors/type names into a dictionary, to be locally accessible

#     miDict = dict({ 1: [ QtGui.QColor(QtCore.Qt.blue), "TestRegion", [8, 4, 2], 3, \
#                          [[QtGui.QColor(QtCore.Qt.blue), "TestTypeFirst", 0.5, 100], \
#                           [QtGui.QColor(QtCore.Qt.blue), "TestTypeSecond", 0.5, 80]] \
#                           ],  \
#                     2: [ QtGui.QColor(QtCore.Qt.green), "SampleRegion", [9, 7, 6], 1, \
#                          [[QtGui.QColor(QtCore.Qt.green), "SampleTypeFirst", 0.2, 50], \
#                           [QtGui.QColor(QtCore.Qt.green), "SampleTypeSecond", 0.8, 75]] \
#                           ] \
#                           } )
    miDict = dict({ 1: [ QtGui.QColor(QtCore.Qt.green), "green", [10, 10, 1], 1, \
                            [  [QtGui.QColor(QtCore.Qt.green), "greenType1", 1.0, 100], \
                               [QtGui.QColor(QtCore.Qt.green), "greenType2", 2.0, 50]  ]   ], \
                    6: [ QtGui.QColor(QtCore.Qt.magenta), "magenta", [4, 3, 7], 1, \
                            [  [QtGui.QColor(QtCore.Qt.magenta), "magentaType", 1.0, 100]  ]   ]   }  )
    mainPanel.setTypesDict(miDict)
    mainPanel.populateTreeWithTypesDict()

    # show() and raise_() have to be called here:
    mainPanel.show()
    # raise_() is a necessary workaround to a PyQt-caused (or Qt-caused?) bug on Mac OS X:
    #   unless raise_() is called right after show(), the window/panel/etc will NOT become
    #   the foreground window and won't receive user input focus:
    mainPanel.raise_()
    print "004 - DEBUG - mi __main__ xe 04"

#     time.sleep(3.0)
#     mainPanel.updateTableOfTypesForImageSequenceOn()
# 
#     time.sleep(3.0)
#     mainPanel.updateTableOfTypesForImageSequenceOff()
# 
#     time.sleep(3.0)
#     mainPanel.updateTableOfTypesForImageSequenceOn()
# 
#     time.sleep(3.0)
#     mainPanel.updateTableOfTypesForImageSequenceOff()

    sys.exit(miApp.exec_())
    print "005 - DEBUG - mi __main__ xe 05"

# end if __name__ == '__main__'
