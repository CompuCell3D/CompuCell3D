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

# 2011 - Mitja: external QWidget for selecting the PIFF Generation mode:
from cdControlPIFFGenerationMode import CDControlPIFFGenerationMode

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# debugging functions, remove in final Panel version
def PWN_debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def PWN_debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]




# ======================================================================
# QTreeWidgetItem-based cell-type-level item, to edit data for *one* cell type
# 
#  this class stores the following cell type data:
#    QTreeWidgetItem-related:
#       pParent = either a QTreeWidgetItem or a QTreeWidget
#       pPreceding = inserted this new item (into the parent) after pPreceding item
#    Scene-related:
#       pCellTypeList = the cell type data:
#            [ QColor for cell type,
#              name (string) for cell type,
#              amount within region (relative to total of all cell amounts),
#              final volume for cell size *if* generated using Potts ]
#            e.g. [QtGui.QColor(64,64,64), "Condensing", 4, 80]
#       pTotalAmounts = total of all cell amounts within region
#       pCellTypeKey = this cell type's key within the region dict
# 
# ======================================================================
class CDOneCellTypeItem(QtGui.QTreeWidgetItem):

    def __init__(self, pParent, pPreceding, \
            pCellTypeList = [QtGui.QColor(0,0,0), "NoNameCell", 0.0, 0], \
            pTotalAmounts = 1, pCellTypeKey = 0):
         # it is compulsory to call the parent's __init__ class right away:
        super(CDOneCellTypeItem, self).__init__(pParent, pPreceding)

        CDConstants.printOut( "   - DEBUG ----- CDOneCellTypeItem: __init__():    pParent = "+str(pParent)+",\n   - pPreceding = "+str(pPreceding)+",\n   - pCellTypeList = "+str(pCellTypeList)+",\n   - pTotalAmounts = "+str(pTotalAmounts)+",\n   - pCellTypeKey = "+str(pCellTypeKey),CDConstants.DebugExcessive )

        self.__oneCellTypeList = pCellTypeList
        self.__totalCellAmounts = pTotalAmounts
        self.__cellTypeKey = pCellTypeKey

        # create the main QTreeWidgetItem entry: both the legend and the value
        self.setText(0, "Cell Type "+str(self.__cellTypeKey))
        self.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.setText(1, str(self.__oneCellTypeList[1]))
        self.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        self.cellAmountPerType = QtGui.QTreeWidgetItem(self)
        lPercentage = (self.__oneCellTypeList[2] / self.__totalCellAmounts) * 100.0
        # format a string to have two decimal places for showing the cell type percentage:
        self.cellAmountPerType.setText(0, str("Amount ("+str( '{0:.2f}'.format(lPercentage) )+"%)"))
        self.cellAmountPerType.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellAmountPerType.setText(1, str(self.__oneCellTypeList[2]))
        self.cellAmountPerType.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellAmountPerType.setFlags( self.cellAmountPerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.cellVolumePerType = QtGui.QTreeWidgetItem(self)
        self.cellVolumePerType.setText(0, str("Volume (pixels)"))
        self.cellVolumePerType.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellVolumePerType.setText(1, str(self.__oneCellTypeList[3]))
        self.cellVolumePerType.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellVolumePerType.setFlags( self.cellVolumePerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )


        # we don't connect any signals directly to QTreeWidgetItem elements:
        #   "itemChanged" signals are handled directly at the the QTreeWidget level.
        #   and calling "blockSignals()" is also done for the entire tree.


    # ------------------------------------------------------------------
    # assign all cell type data for CDOneCellTypeItem:
    # ------------------------------------------------------------------
    def setOneCellTypeData(self, pCellTypeList, pTotalAmounts, pCellTypeKey):
        CDConstants.printOut( "    - DEBUG ----- CDOneCellTypeItem: setOneCellTypeData():\n   - pCellTypeList = "+str(pCellTypeList)+",\n   - pTotalAmounts = "+str(pTotalAmounts)+",\n   - pCellTypeKey = "+str(pCellTypeKey),CDConstants.DebugExcessive )

        self.__oneCellTypeList = pCellTypeList
        self.__totalCellAmounts = pTotalAmounts
        self.__cellTypeKey = pCellTypeKey


    # ------------------------------------------------------------------
    # retrieve all up-to-date cell type data from CDOneCellTypeItem for external use:
    # ------------------------------------------------------------------
    def getOneCellTypeData(self):
        # first rebuild self.__oneCellTypeList list from the CDOneCellTypeItem:
        self.__updateOneCellTypeDataFromEditor()
        return self.__oneCellTypeList


    # ------------------------------------------------------------------
    # populate the CDOneCellTypeItem editor with data from the self.__oneCellTypeList global
    # ------------------------------------------------------------------
    def populateOneCellTypeItemFromData(self):
        # update the main QTreeWidgetItem entry: both the legend and the value
        self.setText(0, "Cell Type "+str(self.__cellTypeKey))
        self.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.setText(1, str(self.__oneCellTypeList[1]))
        self.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        # update the child QTreeWidgetItem for cell amount and percentage entries:
        lPercentage = (self.__oneCellTypeList[2] / self.__totalCellAmounts) * 100.0
        # format a string to have two decimal places for showing the cell type percentage:
        self.cellAmountPerType.setText(0, str("Amount ("+str( '{0:.2f}'.format(lPercentage) )+"%)"))
        self.cellAmountPerType.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellAmountPerType.setText(1, str(self.__oneCellTypeList[2]))
        self.cellAmountPerType.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellAmountPerType.setFlags( self.cellAmountPerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        # update the child QTreeWidgetItem for the cell volume entry:
        self.cellVolumePerType.setText(0, str("Volume (pixels)"))
        self.cellVolumePerType.setIcon(0, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellVolumePerType.setText(1, str(self.__oneCellTypeList[3]))
        self.cellVolumePerType.setIcon(1, self.__createCellColorIcon(self.__oneCellTypeList[0]) )
        self.cellVolumePerType.setFlags( self.cellVolumePerType.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )


    # ------------------------------------------------------------------
    # rebuild the self.__oneCellTypeList global from CDOneCellTypeItem contents:
    # ------------------------------------------------------------------
    def __updateOneCellTypeDataFromEditor(self):
        self.__oneCellTypeList = [   \
              self.__getCellColorFromIcon( self.icon(1) ), \
              str ( self.text(1) ) , \
              float ( self.cellAmountPerType.text(1) ) , \
              int ( self.cellVolumePerType.text(1) )   ]


    # ------------------------------------------------------------------
    # generate a QIcon containing a rectangle in a specified QColor
    # ------------------------------------------------------------------
    def __createCellColorIcon(self, pColor, pPattern = None):
        lPixmap = QtGui.QPixmap(32, 32)
        lPixmap.fill(QtCore.Qt.transparent)
        lPainter = QtGui.QPainter(lPixmap)
        lPainter.setPen(QtCore.Qt.NoPen)
        lBrush = QtGui.QBrush(pColor)
        if pPattern != None:
            lBrush.setStyle(pPattern)
        lPainter.fillRect(QtCore.QRect(0, 0, 32, 32), lBrush)
        lPainter.end()
        return QtGui.QIcon(lPixmap)


    # ------------------------------------------------------------------
    # return the QColor from a QIcon's 32x32 pixmap at (15,15) coordinates
    # ------------------------------------------------------------------
    def __getCellColorFromIcon(self, pIcon):
        lIconColor = QtGui.QColor(pIcon.pixmap(32,32).toImage().pixel(15,15))
        return lIconColor



# end of   class CDOneCellTypeItem(QtGui.QTreeWidgetItem)
# ======================================================================





# ======================================================================
# a QTreeWidgetItem-based top level item to edit data for *one* region type
#
#  this class stores the following region type data:
#    QTreeWidgetItem-related:
#       pParent = the QTreeWidget to which to attach as a top-level item
#       pPreceding = inserted this new item (into the parent) after pPreceding item
#    Scene-related:
#       pRegionQColor = QColor for region type
#       pRegionName = name (string) for region type
#       pCellXYZSizeList = dimensions for cells in region type, e.g. [10,10,10]
#       pRegionTypeUse = how many times is this region type used in the main scene
#       pRegionCellTypesDict = all cell types in region, organized in a dict, e.g.
#          {0: [QtGui.QColor(64,64,64), 'NonCondensing', 1.0, 100],
#           1: [QtGui.QColor(64,64,64), 'Condensing', 2.0, 50]}
#       pCellTypeKey = this cell type's key within the region dict
# 
# ======================================================================
class CDOneRegionTypeItem(QtGui.QTreeWidgetItem):

    def __init__(self, pParent, pPreceding, \
            pRegionQColor = QtGui.QColor(0,0,0), pRegionName = "NoNameRegion", \
            pCellXYZSizeList = [0,0,0], pRegionTypeUse = 0, \
            pRegionCellTypesDict = {0: [QtGui.QColor(0,0,0), 'NoNameCell', 0.0, 0]}, pRegionKey = 0):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDOneRegionTypeItem, self).__init__(pParent, pPreceding)

        CDConstants.printOut( "   ... DEBUG ----- CDOneRegionTypeItem: __init__():  self="+str(self)+",\n   ... pParent="+str(pParent)+",\n   ... pPreceding="+str(pPreceding)+",\n   ... pRegionQColor="+str(pRegionQColor)+",\n   ... pRegionName="+str(pRegionName)+",\n   ... pCellXYZSizeList="+str(pCellXYZSizeList)+",\n   ... pRegionTypeUse="+str(pRegionTypeUse)+",\n   ... pRegionCellTypesDict="+str(pRegionCellTypesDict)+",\n   ... pRegionKey="+str(pRegionKey), CDConstants.DebugExcessive )

        self.__regionColor = pRegionQColor
        self.__regionName = pRegionName
        self.__cellXYZSizeList = pCellXYZSizeList
        self.__regionUse = pRegionTypeUse
        self.__oneRegionCellTypesDict = pRegionCellTypesDict
        self.__regionTypeKey = pRegionKey

        lString = "Region Type " + str(self.__regionTypeKey)
        self.setText(0, lString)
        self.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.setText(1, self.__regionName)
        self.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        self.thisRegionsCellSize = QtGui.QTreeWidgetItem(self)
        self.thisRegionsCellSize.setText(0, str("Cell Size"))
        self.thisRegionsCellSize.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSize.setText(1, str(self.__cellXYZSizeList))
        self.thisRegionsCellSize.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSize.setFlags( self.thisRegionsCellSize.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
#         self.addChild( self.thisRegionsCellSize )

        self.thisRegionsCellSizeX = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeX.setText(0, str("X"))
        self.thisRegionsCellSizeX.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeX.setText(1, str(self.__cellXYZSizeList[0]))
        self.thisRegionsCellSizeX.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeX.setFlags( self.thisRegionsCellSizeX.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeY = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeY.setText(0, str("Y"))
        self.thisRegionsCellSizeY.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeY.setText(1, str(self.__cellXYZSizeList[1]))
        self.thisRegionsCellSizeY.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeY.setFlags( self.thisRegionsCellSizeY.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeZ = QtGui.QTreeWidgetItem(self.thisRegionsCellSize)
        self.thisRegionsCellSizeZ.setText(0, str("Z"))
        self.thisRegionsCellSizeZ.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeZ.setText(1, str(self.__cellXYZSizeList[2]))
        self.thisRegionsCellSizeZ.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeZ.setFlags( self.thisRegionsCellSizeZ.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionTypeUse = QtGui.QTreeWidgetItem(self)
        self.regionTypeUse.setText(0, str("Region Type Use"))
        self.regionTypeUse.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.regionTypeUse.setText(1, str(self.__regionUse))
        self.regionTypeUse.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.regionTypeUse.setFlags( self.regionTypeUse.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionCellTypesItem = QtGui.QTreeWidgetItem(self)
        self.regionCellTypesItem.setText(0, str("Cell Types"))
        self.regionCellTypesItem.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.regionCellTypesItem.setText(1, str(" "))
        self.regionCellTypesItem.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.regionCellTypesItem.setFlags( self.regionCellTypesItem.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        # how many cell type entries are needed for this region type:
        lTypesCount = self.__getOneRegionCellsTypeDictElementCount()
        # get the oneRegionDict keys in order to resize the table:
        lCellsTypeKeys = self.__getOneRegionCellsTypeDictKeys()

        # compute total of cell amounts, for percentages calculated at each tree content update:
        lTotalAmounts = 0.0
        for i in xrange(lTypesCount):
            lOneCellTypeKey = lCellsTypeKeys[i]
            lTotalAmounts = lTotalAmounts + self.__oneRegionCellTypesDict[lOneCellTypeKey][2]

        lIndexInRegion = 0
        for i in xrange(lTypesCount):
            lOneCellTypeKey = lCellsTypeKeys[i]

            # compute a rectangle in the same color,
            #   and modify the color so that it can distinguish cell types within a region:
            #   print "lOneCellTypeKey,self.oneRegionDict[lOneCellTypeKey] = ", lOneCellTypeKey,self.oneRegionDict[lOneCellTypeKey]
            lRegionColor = self.__createRegionQColor( self.__oneRegionCellTypesDict[lOneCellTypeKey][0] )

            # create one QTreeWidgetItem representing a cell type:
            lOneCellTypesItem = self.__createOneCellTypeItem( self.regionCellTypesItem, lOneCellTypeKey, lRegionColor, self.__regionName, self.__oneRegionCellTypesDict[lOneCellTypeKey], lTotalAmounts, lIndexInRegion )

            lIndexInRegion = lIndexInRegion + 1


    # ------------------------------------------------------------------
    # assign all cell type data for CDOneRegionTypeItem:
    # ------------------------------------------------------------------
    def setOneRegionTypeData(self, pRegionQColor, pRegionName, pCellXYZSizeList, \
            pRegionTypeUse, pRegionCellTypesDict, pRegionKey):
    
        CDConstants.printOut( "   ... DEBUG ----- CDOneRegionTypeItem: pRegionQColor="+str(pRegionQColor)+",\n   ... pRegionName="+str(pRegionName)+",\n   ... pCellXYZSizeList="+str(pCellXYZSizeList)+",\n   ... pRegionTypeUse="+str(pRegionTypeUse)+",\n   ... pRegionCellTypesDict="+str(pRegionCellTypesDict)+",\n   ... pRegionKey="+str(pRegionKey),CDConstants.DebugExcessive )

        self.__regionColor = pRegionQColor
        self.__regionName = pRegionName
        self.__cellXYZSizeList = pCellXYZSizeList
        self.__regionUse = pRegionTypeUse
        self.__oneRegionCellTypesDict = pRegionCellTypesDict
        self.__regionTypeKey = pRegionKey


    # ------------------------------------------------------------------
    # retrieve all up-to-date cell type data from CDOneRegionTypeItem for external use:
    # ------------------------------------------------------------------
    def getOneRegionTypeData(self):
        # first rebuild self.__oneCellTypeList list from the CDOneCellTypeItem:


##### TODOTODO : do we need to call self.__updateOneRegionTypeDataFromEditor() every time getOneRegionTypeData() is called???
        CDConstants.printOut( "   >>>> DEBUG ----- CDOneRegionTypeItem: getOneRegionTypeData() BEFORE calling __updateOneRegionTypeDataFromEditor()  :\n   >>>> self.__regionColor="+str(self.__regionColor)+",\n   >>>> self.__regionName="+str(self.__regionName)+",\n   >>>> self.__cellXYZSizeList="+str(self.__cellXYZSizeList)+",\n   >>>> self.__regionUse="+str(self.__regionUse)+",\n   >>>> self.__oneRegionCellTypesDict="+str(self.__oneRegionCellTypesDict)+",\n   >>>> self.__regionTypeKey="+str(self.__regionTypeKey), CDConstants.DebugExcessive )

        self.__updateOneRegionTypeDataFromEditor()

        CDConstants.printOut( "   >>>> DEBUG ----- CDOneRegionTypeItem: getOneRegionTypeData() AFTER calling __updateOneRegionTypeDataFromEditor()  :\n   >>>> self.__regionColor="+str(self.__regionColor)+",\n   >>>> self.__regionName="+str(self.__regionName)+",\n   >>>> self.__cellXYZSizeList="+str(self.__cellXYZSizeList)+",\n   >>>> self.__regionUse="+str(self.__regionUse)+",\n   >>>> self.__oneRegionCellTypesDict="+str(self.__oneRegionCellTypesDict)+",\n   >>>> self.__regionTypeKey="+str(self.__regionTypeKey), CDConstants.DebugExcessive )
##### TODOTODO : do we need to call self.__updateOneRegionTypeDataFromEditor() every time getOneRegionTypeData() is called???



        return (  self.__regionColor, \
            self.__regionName, \
            self.__cellXYZSizeList, \
            self.__regionUse, \
            self.__oneRegionCellTypesDict, \
            self.__regionTypeKey )
        


    # ------------------------------------------------------------------
    # populate the CDOneRegionTypeItem editor with self.__oneRegionCellTypesDict etc.
    # ------------------------------------------------------------------
    def populateOneRegionTypeItemFromData(self):
    
        lString = "Region Type " + str(self.__regionTypeKey)
        self.setText(0, lString)
        self.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.setText(1, self.__regionName)
        self.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.setFlags(self.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))

        self.thisRegionsCellSize.setText(0, str("Cell Size"))
        self.thisRegionsCellSize.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSize.setText(1, str(self.__cellXYZSizeList))
        self.thisRegionsCellSize.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSize.setFlags( self.thisRegionsCellSize.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )
#         self.addChild( self.thisRegionsCellSize )

        self.thisRegionsCellSizeX.setText(0, str("X"))
        self.thisRegionsCellSizeX.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeX.setText(1, str(self.__cellXYZSizeList[0]))
        self.thisRegionsCellSizeX.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeX.setFlags( self.thisRegionsCellSizeX.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeY.setText(0, str("Y"))
        self.thisRegionsCellSizeY.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeY.setText(1, str(self.__cellXYZSizeList[1]))
        self.thisRegionsCellSizeY.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeY.setFlags( self.thisRegionsCellSizeY.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.thisRegionsCellSizeZ.setText(0, str("Z"))
        self.thisRegionsCellSizeZ.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeZ.setText(1, str(self.__cellXYZSizeList[2]))
        self.thisRegionsCellSizeZ.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.thisRegionsCellSizeZ.setFlags( self.thisRegionsCellSizeZ.flags() |   \
            (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionTypeUse.setText(0, str("Region Type Use"))
        self.regionTypeUse.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.regionTypeUse.setText(1, str(self.__regionUse))
        self.regionTypeUse.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.regionTypeUse.setFlags( self.regionTypeUse.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        self.regionCellTypesItem.setText(0, str("Cell Types"))
        self.regionCellTypesItem.setIcon(0, self.__createRegionColorIcon(self.__regionColor) )
        self.regionCellTypesItem.setText(1, str(" "))
        self.regionCellTypesItem.setIcon(1, self.__createRegionColorIcon(self.__regionColor) )
        self.regionCellTypesItem.setFlags( self.regionCellTypesItem.flags() &   \
            ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)  )

        # how many cell type entries are needed for this region type:
        lTypesCount = self.__getOneRegionCellsTypeDictElementCount()
        # get the oneRegionDict keys in order to resize the table:
        lCellsTypeKeys = self.__getOneRegionCellsTypeDictKeys()

        # compute total of cell amounts in region, from amounts of individual cell type items:
        lTotalAmounts = 0.0
        for i in xrange(lTypesCount):
            lOneCellItem = self.regionCellTypesItem.child(i)
            lOneCellTypeList = lOneCellItem.getOneCellTypeData()
            lTotalAmounts = lTotalAmounts + lOneCellTypeList[2]

        for i in xrange(lTypesCount):
            lOneCellTypeKey = lCellsTypeKeys[i]
            lOneCellItem = self.regionCellTypesItem.child(i)
            lOneCellItem.setOneCellTypeData(self.__oneRegionCellTypesDict[lOneCellTypeKey], lTotalAmounts, lOneCellTypeKey)
            lOneCellItem.populateOneCellTypeItemFromData()
            
            

    # ------------------------------------------------------------------
    # rebuild self.__oneRegionCellTypesDict etc. from CDOneRegionTypeItem contents:
    # ------------------------------------------------------------------
    def __updateOneRegionTypeDataFromEditor(self):
        # make sure that the content of the editor and the dict are consistent:
        if (self.regionCellTypesItem.childCount() != \
            self.__getOneRegionCellsTypeDictElementCount() ):
            CDConstants.printOut("CDOneRegionTypeItem - __updateOneRegionTypeDataFromEditor(), self.regionCellTypesItem.childCount() == " +str(self.regionCellTypesItem.childCount())+" is not the same as self.__getOneRegionCellsTypeDictElementCount() == "+str(self.__getOneRegionCellsTypeDictElementCount()), CDConstants.DebugImportant)            

        # get the self.__oneRegionCellTypesDict keys:
        lCellsTypeKeys = self.__getOneRegionCellsTypeDictKeys()
        for i in xrange(self.regionCellTypesItem.childCount()):
            lOneCellTypeKey = lCellsTypeKeys[i]
            # get one CDOneCellTypeItem at a time:
            lOneCellTypesItem = self.regionCellTypesItem.child(i)
            # obtain its cell type data list:
            self.__oneRegionCellTypesDict[lOneCellTypeKey] = lOneCellTypesItem.getOneCellTypeData()

        # the value of self.__regionTypeKey can not be changed in the editor

        # all the other data is more straightforwardly obtained:
        self.__regionColor = self.__getCellColorFromIcon( self.icon(1) )
        self.__regionName = str( self.text(1) )
        self.__cellXYZSizeList = [ \
            int ( self.thisRegionsCellSizeX.text(1) ) , \
            int ( self.thisRegionsCellSizeY.text(1) ) , \
            int ( self.thisRegionsCellSizeZ.text(1) ) ]
        self.__regionUse = int ( self.regionTypeUse.text(1) )



    # ------------------------------------------------------------------
    # __createOneCellTypeItem() returns one top-level CDOneCellTypeItem for one region type:
    # ------------------------------------------------------------------
    def __createOneCellTypeItem( self, pParent, pCellTypeKey, pRegionTypeQColor, pRegionName, pCellTypeList, pTotalAmounts, pIndex):

        if pIndex != 0:
            lPreceding = self.__childAt(pParent, pIndex - 1)
        else:        
            lPreceding = None

        # create a CDOneCellTypeItem and attach it to the "self" QTreeWidget:
        lOneCellItem = CDOneCellTypeItem(pParent, lPreceding, \
            pCellTypeList, pTotalAmounts, pCellTypeKey)
        
        # we don't really need/have to get and refill the data again
        #   and then ask the item to repopulate, but we're testing if
        #   both the CDOneCellTypeItem init and its separate calls work the same way:
        lTestCellTypeList = lOneCellItem.getOneCellTypeData()
        lOneCellItem.setOneCellTypeData(lTestCellTypeList, pTotalAmounts, pCellTypeKey)
        lOneCellItem.populateOneCellTypeItemFromData()

        return lOneCellItem

    # end of      def __createOneCellTypeItem( self, pCellTypeKey, pRegionTypeQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pCellTypeList, pTotalAmounts, pIndex )
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # __childAt() returns one top-level QTreeWidgetItem for one region type:
    # ------------------------------------------------------------------
    def __childAt(self, pParent, pIndex):
        if pParent is not None:
            return pParent.child(pIndex)
        else:
            #return self.topLevelItem(pIndex)
            return None


    # ------------------------------------------------------------------
    # __getOneRegionCellsTypeDictElementCount() is a helper function,
    #   returning the number of elements in the typesDict global:
    # ------------------------------------------------------------------
    def __getOneRegionCellsTypeDictElementCount(self):
        return len(self.__oneRegionCellTypesDict)


    # ------------------------------------------------------------------
    # __getOneRegionCellsTypeDictKeys() is a helper function,
    #   returning a list of keys from __oneRegionCellTypesDict:
    # ------------------------------------------------------------------
    def __getOneRegionCellsTypeDictKeys(self):
        return self.__oneRegionCellTypesDict.keys()


    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def __createRegionColorIcon(self, pColor, pPattern = None):
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
    # generate a QColor in a specified color
    # ------------------------------------------------------------------
    def __createRegionQColor(self, pColor):

        # pColor could be one of many ways of defining a color,
        #    but this function returns a proper QColor type:
        return QtGui.QColor(pColor)

    # ------------------------------------------------------------------
    # return the QColor from a QIcon's 32x32 pixmap at (15,15) coordinates
    # ------------------------------------------------------------------
    def __getCellColorFromIcon(self, pIcon):
        lIconColor = QtGui.QColor(pIcon.pixmap(32,32).toImage().pixel(15,15))
        return lIconColor



# end of   class CDOneRegionTypeItem(QtGui.QTreeWidgetItem)
# ======================================================================





# ======================================================================
# a QTreeWidget-based types editor
# ======================================================================
class CDTypesTree(QtGui.QTreeWidget):

    # ----------------------------------------
    def __init__(self, parent=None):
        super(CDTypesTree, self).__init__(parent)

        self.setHeaderLabels(("Property", "Value"))

        self.itemActivated.connect(self.__handleTreeItemActivated)

#         self.header().setResizeMode(0, QtGui.QHeaderView.Stretch)
#         self.header().setResizeMode(1, QtGui.QHeaderView.Stretch)
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
    def PWN_hideRegionTypeRow(self, pRowNumber):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: hideRegionTypeRow() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   hideRegionTypeRow(self, pRowNumber)
    # --------------------------------------------


    # --------------------------------------------
    def PWN_showRegionTypeRow(self, pRowNumber):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: showRegionTypeRow() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   showRegionTypeRow(self, pRowNumber)
    # --------------------------------------------


    # --------------------------------------------
    def PWN_setOneRegionTypeItem(self, pRowNumber, pColumnNumber, pItem):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: setOneRegionTypeItem() done. ----- ", CDConstants.DebugExcessive )
        pass

    # end of   setOneRegionTypeItem(self, pRowNumber)
    # --------------------------------------------



    # ------------------------------------------------------------------
    # createTopRegionItem() returns one top-level CDOneRegionTypeItem for one region type:
    # ------------------------------------------------------------------
    def createTopRegionItem( self, pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pRegionKey, pIndex ):
        if pIndex != 0:
            # lPreceding = self.__childAt(parent, pIndex - 1)
            lPreceding = self.__childAt(None, pIndex - 1)
        else:        
            lPreceding = None

        # create a CDOneRegionTypeItem and attach it to the "self" QTreeWidget:
        lTopLevelItem = CDOneRegionTypeItem(self, lPreceding, pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pRegionKey)

##### TODOTODO :
        # we don't really need/have to get and refill the data again
        #   and then ask the item to repopulate, but we're testing if
        #   both the CDOneRegionTypeItem init and its separate calls work the same way:

        (lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey) = lTopLevelItem.getOneRegionTypeData()

        lTopLevelItem.setOneRegionTypeData( lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey)

        lTopLevelItem.populateOneRegionTypeItemFromData()

        return lTopLevelItem

    # end of      def createTopRegionItem( self, pRegionKey, pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pIndex )
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # updateTopRegionItem() loads new data into an existing top-level CDOneRegionTypeItem,
    #    for one region type, located with its "pRegionQColor":
    # ------------------------------------------------------------------
    def updateTopRegionItem( self, pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pRegionKey ):

        # scan all top-level items one by one, and compare by region color:
        lTopLevelItem = None
        for i in xrange(self.topLevelItemCount()):
            lTmpItem = self.__childAt(None, i)
            (lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey) = lTmpItem.getOneRegionTypeData()

            if pRegionQColor.rgba() == lRegionQColor.rgba() :
                CDConstants.printOut( "___ ----- CDTypesTree: updateTopRegionItem()  ----- the asked pRegionQColor.rgba()=="+str(pRegionQColor.rgba())+ \
                    " is the SAME as the existing lRegionQColor.rgba()=="+str(lRegionQColor.rgba()), CDConstants.DebugAll )
                lTopLevelItem = lTmpItem
                lTmpItem = None
                break
            else:
                CDConstants.printOut( "___ ----- CDTypesTree: updateTopRegionItem()  ----- the asked pRegionQColor.rgba()=="+str(pRegionQColor.rgba())+ \
                    " is DIFFERENT than the existing lRegionQColor.rgba()=="+str(lRegionQColor.rgba()), CDConstants.DebugAll )
                lTmpItem = None

        if (lTopLevelItem != None):
            # update the CDOneRegionTypeItem with the correct region key:
            lTopLevelItem.setOneRegionTypeData( pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pRegionKey)
            lTopLevelItem.populateOneRegionTypeItemFromData()

        return lTopLevelItem

    # end of   def updateTopRegionItem( self, pRegionQColor, pRegionName, pCellXYZSizeList, pRegionTypeUse, pRegionCellTypesDict, pRegionKey )
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # getTopRegionItemData() returns one top-level CDOneRegionTypeItem for one region type:
    # retrieve all up-to-date region type data from CDTypesTree for external use:
    # ------------------------------------------------------------------
    def getTopRegionItemData( self, pIndex ):
    
        return self.topLevelItem(pIndex).getOneRegionTypeData()

    # end of      def getTopRegionItemData( self, pIndex )
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # __childAt() returns one top-level QTreeWidgetItem for one region type:
    # ------------------------------------------------------------------
    def __childAt(self, parent, index):
        if parent is not None:
            return parent.child(index)
        else:
            return self.topLevelItem(index)



    # ------------------------------------------------------------------
    # __handleTreeItemActivated() handles "itemActivated" signals from CDTypesTree:
    #       The logic will happen in the editor delegate. This is needed to let
    #       the delegate run by making this editable
    # ------------------------------------------------------------------
    def __handleTreeItemActivated(self, pItem, pColumn):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesTree: __handleTreeItemActivated() called with pItem="+str(pItem)+" pColumn="+str(pColumn)+" ----- ", CDConstants.DebugTODO )
        
        # only allow the editing of the "Value" column items
        if (pColumn > 0):
            pItem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
            # Force the pItem in to edit mode so the delegate picks it up
            self.editItem(pItem, pColumn)
            # Set the pItem back to not editable. The delegate will still do its
            #    job, but the read-only state will already be set when done!
            pItem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)


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

        # init windowing GUI stuff:
        #
        self.__miInitEditorGUIGeneralBehavior()

        #
        # create the central tree-based-editor and set it up:
        #
        self.theTypesTree = CDTypesTree()
        
        #  place the tree-based-editor inside the panel:
        self.layout().addWidget( self.theTypesTree )
#         self.layout().addWidget(self.__miInitCentralTreeWidget())

        # to control the "PIFF Generation mode", we add a set of radio-buttons:
        self.theControlsForPIFFGenerationMode = CDControlPIFFGenerationMode()
        self.layout().addWidget(self.theControlsForPIFFGenerationMode)


        # init - create an empty dict for types,
        #   and "populate" the editor with the empty dict:
        #
        self.typesDict = dict()
        self.__debugRegionDict()
        self.populateEditorFromTypesDict()



        # the QObject.connect( QObject, SIGNAL(), QObject, SLOT() ) function
        #   creates a connection of the given type from the signal in the sender object
        #   to the slot method in the receiver object,
        #   and it returns true if the connection succeeds; otherwise it returns false.

        # TODOTODO - 
        # 1. do we need to create an itemChanged signal in theTypesTree?
        # 2. do we need a handler for that signal?
        #
        # connect the "itemChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.theTypesTree.itemChanged[QtGui.QTreeWidgetItem,int].connect(self.__handleItemChanged)

        # to be used as "flag name" when switching to Image Sequence use and back:
        self.theBackupCellTypeNameWhenSwitchingToImageSequenceAndBack = ""

        # print "005 - DEBUG ----- CDTypesEditor.__init__(): done"


    # ------------------------------------------------------------------
    # define functions to initialize this panel:
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init - windowing GUI stuff:
    # ------------------------------------------------------------------
    def __miInitEditorGUIGeneralBehavior(self):

        CDConstants.printOut( "    - DEBUG ----- CDTypesEditor.__miInitEditorGUIGeneralBehavior(): starting", CDConstants.DebugExcessive )

        # how will the CDTypesEditor look like:
        #   this title won't appear anywhere if the CDTypesEditor widget is included in a QDockWidget:
        self.setWindowTitle("Types of Cells and Regions")

        self.setMinimumSize(200,200)
        # self.setMinimumSize(466,322)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # NOTE: the x,y is NOT the top-left edge of the widget,
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
        self.typesEditorWidgetMainLayout = QtGui.QVBoxLayout()
        self.typesEditorWidgetMainLayout.setContentsMargins(2,2,2,2)
        self.typesEditorWidgetMainLayout.setSpacing(4)
        self.typesEditorWidgetMainLayout.setAlignment( \
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setLayout(self.typesEditorWidgetMainLayout)

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

        # do not delete the widget when it is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)

        self.show()

        CDConstants.printOut( "    - DEBUG ----- CDTypesEditor.__miInitEditorGUIGeneralBehavior() done.", CDConstants.DebugExcessive )

    # end of        def __miInitEditorGUIGeneralBehavior(self)
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
        self.__debugRegionDict()
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.self.setTypesDict() to "+str(self.typesDict)+" done. ----- ", CDConstants.DebugVerbose )


    # ------------------------------------------------------------------
    # retrieve the up-to-date typesDict for external use:
    # ------------------------------------------------------------------
    def getTypesDict(self):
        # first rebuild the typesDict global from its PIFOneRegionTable objects:
        self.__updateTypesDictFromEditor()
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.self.getTypesDict() will now return self.typesDict=="+str(self.typesDict) , CDConstants.DebugVerbose )
        return self.typesDict




    # ------------------------------------------------------------------
    # populate the main QTreeWidget editor with data from the typesDict global
    # ------------------------------------------------------------------
    def populateEditorFromTypesDict(self):
        CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.populateEditorFromTypesDict() BEGIN: now " + \
            str(self.theTypesTree.topLevelItemCount()) + " top-level items in tree and " + \
            str(self.__getRegionsDictElementCount()) + " regions.", CDConstants.DebugExcessive )

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to access individual dict entries:
        lKeys = self.__getRegionsDictKeys()
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False
        
        lIndexInTree = 0

        for i in xrange(self.__getRegionsDictElementCount()):

            # prepare all data for one top-level QTreeWidgetItem representing a region type:
            
            # 1.st parameter, set its string value to the dict key for region i:
            lRegionKey = QtCore.QString("%1").arg(lKeys[i])
            # 2.nd parameter, a QColor with the region color:
            lRegionQColor = self.__createQColor( self.typesDict[lKeys[i]][0] )
            # 3.rd parameter, the region name from typesDict:
            lRegionName = QtCore.QString("%1").arg(self.typesDict[lKeys[i]][1])

            # 4.th parameter, a list the cell sizes from typesDict in it:
            lCellXYZSizeList = list( ( -1, -2, -3) )
            CDConstants.printOut ( "self.typesDict[lKeys[i]][2] = " + str(self.typesDict[lKeys[i]][2]), CDConstants.DebugAll )
            for j in xrange(3):
                CDConstants.printOut (  "at i = " +str(i)+ ", j = " + str(j) + "self.typesDict[lKeys[i]][2][j] = " + str(self.typesDict[lKeys[i]][2][j]), CDConstants.DebugAll )
                lCellXYZSizeList[j] = self.typesDict[lKeys[i]][2][j]

            # 5.th parameter, the region type use (how many regions in the scene are of this type) from typesDict:
            lRegionTypeUse = QtCore.QString("%1").arg(self.typesDict[lKeys[i]][3])

            # 6.th parameter, a dict with all cell type data for this region type:
            lRegionCellTypesDict = dict()
            CDConstants.printOut ( "self.typesDict[lKeys[i]][4] =" + str( self.typesDict[lKeys[i]][4] ), CDConstants.DebugAll )
            for j in xrange( len(self.typesDict[lKeys[i]][4]) ):

                CDConstants.printOut ( "at i ="+str( i )+", j ="+str( j )+"self.typesDict[lKeys[i]][4][j] ="+str( self.typesDict[lKeys[i]][4][j] ), CDConstants.DebugAll )

                lRegionCellTypesDict[j] = self.typesDict[lKeys[i]][4][j]

            # create one top-level QTreeWidgetItem representing a region type:
            lChild = self.theTypesTree.createTopRegionItem( lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey, lIndexInTree )

            # finally determine if the QTreeWidgetItem at index i is to be visible or not,
            #    according to its region type use in the scene:
            CDConstants.printOut ( "self.typesDict[lKeys[i]][3] ="+str(self.typesDict[lKeys[i]][3]), CDConstants.DebugAll )
            if self.typesDict[lKeys[i]][3] < 1:
                lChild.setHidden(True)
            else:
                lChild.setHidden(False)
                lThereAreRegionRowsInUse = True

            lIndexInTree = lIndexInTree + 1

        # end of  for i in xrange(self.__getRegionsDictElementCount())


        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)

        CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.populateEditorFromTypesDict() END: now " + \
            str(self.theTypesTree.topLevelItemCount()) + " top-level items in tree and " + \
            str(self.__getRegionsDictElementCount()) + " regions. ----- DONE.", CDConstants.DebugExcessive )

        return
    # end of  def populateEditorFromTypesDict(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # populate the table widget with data from the typesDict global
    # ------------------------------------------------------------------
    def updateRegionUseInTypesEditor(self, pColor, pHowManyInUse):

        lTheRegionToBeUpdatedColor = QtGui.QColor(pColor)

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # check how many rows are present in the table:
        lRegionsCount = self.theTypesTree.topLevelItemCount()
        CDConstants.printOut("___         ----- CDTypesEditor.updateRegionUseInTypesEditor() ----- lRegionsCount="+str(lRegionsCount), CDConstants.DebugTODO )

        # get the typesDict keys in order to access individual dict entries:
        lRegionKeys = self.__getRegionsDictKeys()
        CDConstants.printOut("___         ----- CDTypesEditor.updateRegionUseInTypesEditor() ----- lRegionKeys="+str(lRegionKeys), CDConstants.DebugTODO )
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False

        for i in xrange(lRegionsCount):
            lOneRegionKey = lRegionKeys[i]
            CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- self.typesDict[lRegionKeys[i=="+str(i)+"]=="+str(lRegionKeys[i])+"] ="+str(self.typesDict[ lRegionKeys[i] ]), CDConstants.DebugAll )
            CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- self.typesDict[lOneRegionKey=="+str(lOneRegionKey)+"]] ="+str(self.typesDict[lOneRegionKey]), CDConstants.DebugAll )
            CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- self.typesDict[lOneRegionKey][0] ="+str(self.typesDict[lOneRegionKey][0]), CDConstants.DebugAll )
            CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- self.typesDict[lOneRegionKey][0].rgba() ="+str(self.typesDict[lOneRegionKey][0].rgba()), CDConstants.DebugAll )
            CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- lTheRegionToBeUpdatedColor.rgba() ="+str(lTheRegionToBeUpdatedColor.rgba()), CDConstants.DebugAll )
            if self.typesDict[lOneRegionKey][0].rgba() == lTheRegionToBeUpdatedColor.rgba() :
                # prepare all data for one top-level QTreeWidgetItem representing a region type:

                # first update the region use in the main typesDict for region types:
                CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- IT WAS self.typesDict[lOneRegionKey][3] ="+str(self.typesDict[lOneRegionKey][3]), CDConstants.DebugAll )
                self.typesDict[lOneRegionKey][3] = pHowManyInUse
                CDConstants.printOut( "___         ----- CDTypesEditor.updateRegionUseInTypesEditor()  ----- NOW IS self.typesDict[lOneRegionKey][3] ="+str(self.typesDict[lOneRegionKey][3]), CDConstants.DebugAll )

                # then prepare all other parameters to update the region's entry in the editor tree:
                lRegionKey = QtCore.QString("%1").arg(lOneRegionKey)
                lRegionQColor = self.__createQColor( self.typesDict[lOneRegionKey][0] )
                lRegionName = QtCore.QString("%1").arg(self.typesDict[lOneRegionKey][1])
                lCellXYZSizeList = list( ( -1, -2, -3) )
                for j in xrange(3):
                    lCellXYZSizeList[j] = self.typesDict[lOneRegionKey][2][j]
                lRegionTypeUse = QtCore.QString("%1").arg(self.typesDict[lOneRegionKey][3])
                lRegionCellTypesDict = dict()
                for j in xrange( len(self.typesDict[lOneRegionKey][4]) ):
                    lRegionCellTypesDict[j] = self.typesDict[lOneRegionKey][4][j]

                lChild = self.theTypesTree.updateTopRegionItem( lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey )

                # finally determine if the QTreeWidgetItem at index i is to be visible or not,
                #    according to its region type use in the scene:
                if self.typesDict[lOneRegionKey][3] < 1:
                    lChild.setHidden(True)
                else:
                    lChild.setHidden(False)
                    lThereAreRegionRowsInUse = True

#         if lThereAreRegionRowsInUse is False:
#             self.theTypesTree.hide()
#         else:
#             self.theTypesTree.show()

        # allow theTypesTree to emit "itemChanged" signals now that we're done
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(False)


    # end of  def updateRegionUseInTypesEditor(self, pColor, pHowManyInUse).
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # rebuild the typesDict global by retrieving all the values in theTypesTree:
    # ------------------------------------------------------------------
    def __updateTypesDictFromEditor(self):
        CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.__updateTypesDictFromEditor() BEGIN: now " + \
            str(self.theTypesTree.topLevelItemCount()) + " top-level items in tree and " + \
            str(self.__getRegionsDictElementCount()) + " regions.", CDConstants.DebugExcessive )

        # set how many rows are needed in the table:
        lRegionsCount = self.theTypesTree.topLevelItemCount()
        CDConstants.printOut("___         ----- CDTypesEditor.__updateTypesDictFromEditor() ----- lRegionsCount="+str(lRegionsCount), CDConstants.DebugTODO )

        # get the typesDict keys in order to access individual dict entries:
        lKeys = self.__getRegionsDictKeys()
        CDConstants.printOut("___         ----- CDTypesEditor.__updateTypesDictFromEditor() ----- lKeys="+str(lKeys), CDConstants.DebugTODO )

        # parse each top-level item separately to build a typesDict entry:
        for i in xrange(lRegionsCount):
        
            # get all data from one top-level item in the types tree, i.e. data for one region type:
            (lRegionQColor, lRegionName, lCellXYZSizeList, lRegionTypeUse, lRegionCellTypesDict, lRegionKey) = self.theTypesTree.getTopRegionItemData(i)

            CDConstants.printOut( "   >>>> DEBUG ----- CDTypesEditor.__updateTypesDictFromEditor() AFTER calling self.theTypesTree.getTopRegionItemData(i=="+str(i)+") :\n   >>>> lRegionQColor="+str(lRegionQColor)+",\n   >>>> lRegionName="+str(lRegionName)+",\n   >>>> lCellXYZSizeList="+str(lCellXYZSizeList)+",\n   >>>> lRegionTypeUse="+str(lRegionTypeUse)+",\n   >>>> lRegionCellTypesDict="+str(lRegionCellTypesDict)+",\n   >>>> lRegionKey="+str(lRegionKey), CDConstants.DebugTODO )

            # the top-level key is NOT retrieved from theTypesTree items:
            #    not: lKey = self.theTypesTree.item(i, 0)
#             lKey = lKeys[i]
            # the color can be retrieved (although the color is not editable (yet?) )
#             lColor = self.typesDict[lKeys[i]][0]

            # the region name is retrieved from the 3.rd theTypesTree column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor.self.__updateTypesDictFromEditor() \n"
            # print "      i, self.theTypesTree.item(i, 2) ", i, self.theTypesTree.item(i, 2)
#             lRegionName = str ( self.theTypesTree.item(i, 2).text() )
#
#             # the region cell size is retrieved from the 4.th theTypesTree column:
#             # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor.self.__updateTypesDictFromEditor() \n"
#             # print "      i, self.theTypesTree.item(i, 3) ", i, self.theTypesTree.item(i, 3)
#             lRegionCellSize = int ( self.theTypesTree.item(i, 3).text() )

            # the region cell sizes are retrieved from the 4.th theTypesTree column:
#             CDConstants.printOut ( "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor.self.__updateTypesDictFromEditor() ", CDConstants.DebugAll )
#             CDConstants.printOut ( "      i, self.theTypesTree.cellWidget(i, 3) " + str(i) + " " + str( self.theTypesTree.cellWidget(i, 3) ), CDConstants.DebugAll )
#             lRegionBlockCellSizes = self.theTypesTree.cellWidget(i, 3)

            # the region use is retrieved from the 5.th theTypesTree column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDTypesEditor.self.__updateTypesDictFromEditor() \n"
            # print "      i, self.theTypesTree.item(i, 4) ", i, self.theTypesTree.item(i, 4)
#             lRegionInUse = int ( self.theTypesTree.item(i, 4).text() )

            # the cell types dict for each region is obtained from the PIFOneRegionTable widget
            #   which is in the 6.th  theTypesTree column:
#             lOneRegionTableWidget = self.theTypesTree.cellWidget(i, 5)

            # rebuild the related dict entry - ***warning*** if lRegionKey is not casted into an int,
            #    the key will become a Unicode string, and entries will be duplicated:
            self.typesDict[int(lRegionKey)] = [ lRegionQColor, lRegionName, \
                                       lCellXYZSizeList, \
                                       lRegionTypeUse, \
                                       lRegionCellTypesDict ]



        CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.__updateTypesDictFromEditor() to self.typesDict =\n    "+str(self.typesDict)+ "\n     done.", CDConstants.DebugTODO )
        
        CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.__updateTypesDictFromEditor() END: now " + \
            str(self.theTypesTree.topLevelItemCount()) + " top-level items in tree and " + \
            str(self.__getRegionsDictElementCount()) + " regions. ----- DONE.", CDConstants.DebugExcessive )

    # end of    def __updateTypesDictFromEditor(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def PWN_NOTUSED__populateTableWithTypesDict(self):
    

        for i in xrange(self._getRegionsDictElementCount()):


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

        # print "___ - DEBUG ----- CDTypesEditor.NOTUSED__populateTableWithTypesDict() : done"


    # end of  def NOTUSED__populateTableWithTypesDict(self)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def PWN_updateTableOfTypesForImageSequenceOn(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOn() starting ----- ", CDConstants.DebugExcessive )

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to update the table's elements accordingly:
        lKeys = self._getRegionsDictKeys()

        # the *only*  row to be shown is this one:
        lColorForCellSeedsInImageSequence = QtGui.QColor(QtCore.Qt.magenta)
        lTypeNameForCellSeedsInImageSequence = str("magentaType")

        lOneRegionTableWidget = None
        for i in xrange(self._getRegionsDictElementCount()):

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


        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOn() ending. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOn(self).
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def updateTableOfTypesForImageSequenceOn(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOn() DO THIS. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOn(self).
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def updateTableOfTypesForImageSequenceOff(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOff() DO THIS. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOff(self).
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # adjust the table widget for handling image sequence data:
    # ------------------------------------------------------------------
    def PWN_updateTableOfTypesForImageSequenceOff(self):
        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOff() starting ----- ", CDConstants.DebugExcessive )

        # prevent theTypesTree from emitting any "itemChanged" signals when
        #   updating its content programmatically:
        self.theTypesTree.blockSignals(True)

        # get the typesDict keys in order to update the table's elements accordingly:
        lKeys = self._getRegionsDictKeys()

        # the only row that was to be shown in Image Sequence was this one:
        lColorForCellSeedsInImageSequence = QtGui.QColor(QtCore.Qt.magenta)
        lTypeNameForCellSeedsInImageSequence = str("magentaType")

        # the table rows to be shown are those with regions used by the PIFF scene:
        lOneRegionTableWidget = None
        for i in xrange(self._getRegionsDictElementCount()):

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

        CDConstants.printOut( "___ - DEBUG ----- CDTypesEditor.updateTableOfTypesForImageSequenceOff() ending. ----- ", CDConstants.DebugExcessive )
    # end of  def updateTableOfTypesForImageSequenceOff(self).
    # ------------------------------------------------------------------







    # ------------------------------------------------------------------
    # __debugRegionDict() is a debugging aid function to print out information
    #   about the typesDict global
    # ------------------------------------------------------------------
    def __debugRegionDict(self):
        CDConstants.printOut("--------------------------------------------- CDTypesEditor.__debugRegionDict() begin", CDConstants.DebugExcessive )
        lCount = self.__getRegionsDictElementCount()
        lKeys = self.__getRegionsDictKeys()
        CDConstants.printOut( " CDTypesEditor class, typesDict global: ", CDConstants.DebugAll )
        CDConstants.printOut( " ", CDConstants.DebugAll )
        CDConstants.printOut( " tree-based-editor rows = "+str(lCount), CDConstants.DebugAll )
        CDConstants.printOut( " tree-based-editor keys = "+str(lKeys), CDConstants.DebugAll )
        CDConstants.printOut( " max no of cell types = "+str( self.__getRegionsDictMaxNoOfCellTypes() ), CDConstants.DebugAll )
       
        CDConstants.printOut( " tree-based-editor row content: ", CDConstants.DebugAll )
        for i in xrange(lCount):
            CDConstants.printOut( "i =                                     = "+str(i), CDConstants.DebugAll )
            CDConstants.printOut( "key =                       lKeys[i]     = "+str(lKeys[i]), CDConstants.DebugAll )
            CDConstants.printOut( "color =    self.typesDict[lKeys[i]][0] = "+str(self.typesDict[lKeys[i]][0]), CDConstants.DebugAll )
            CDConstants.printOut( "region name =   self.typesDict[lKeys[i]][1] = "+str(self.typesDict[lKeys[i]][1]), CDConstants.DebugAll )
            CDConstants.printOut( "cellsize = self.typesDict[lKeys[i]][2] = "+str(self.typesDict[lKeys[i]][2]), CDConstants.DebugAll )
            for j in xrange(3):
                CDConstants.printOut( "cellsizes -- at i = "+str(i)+", j = "+str(j)+" , self.typesDict[lKeys[i]][2][j] = "+str(self.typesDict[lKeys[i]][2][j]), CDConstants.DebugAll )
            CDConstants.printOut( "use =      self.typesDict[lKeys[i]][3] = "+str(self.typesDict[lKeys[i]][3]), CDConstants.DebugAll )
            for j in xrange( len(self.typesDict[lKeys[i]][4]) ):
                CDConstants.printOut( "at i = "+str(i)+", j = "+str(j)+" , self.typesDict[lKeys[i]][4][j] = "+str(self.typesDict[lKeys[i]][4][j]), CDConstants.DebugAll )
        CDConstants.printOut("--------------------------------------------- CDTypesEditor.__debugRegionDict() end", CDConstants.DebugExcessive )

    # ------------------------------------------------------------------
    # __getRegionsDictKeys() is a helper function,
    #   it just returns the keys in the typesDict global:
    # ------------------------------------------------------------------
    def __getRegionsDictKeys(self):
        return self.typesDict.keys()

    # ------------------------------------------------------------------
    # __getRegionsDictElementCount() is a helper function,
    #   it just returns number of elements in the typesDict global:
    # ------------------------------------------------------------------
    def __getRegionsDictElementCount(self):
        return len(self.typesDict)

    # ------------------------------------------------------------------
    # __getRegionsDictMaxNoOfCellTypes() is a helper function,
    #   it returns the maximum number of cell type entries in any typesDict entry
    def __getRegionsDictMaxNoOfCellTypes(self):
        maxRowCount = 0
        keys = self.__getRegionsDictKeys()
        # see how many regions are present at most in the table:
        for i in xrange(self.__getRegionsDictElementCount()):
            howMany = len(self.typesDict[keys[i]][4])
            if howMany > maxRowCount:
                maxRowCount = howMany

#         CDConstants.printOut("___ - DEBUG ----- CDTypesEditor.__getRegionsDictMaxNoOfCellTypes() = " + \
#             str(maxRowCount), CDConstants.DebugVerbose )
        return maxRowCount


#
#     # ------------------------------------------------------------------
#     # handle mouse click events in table elements:
#     # ------------------------------------------------------------------
#     def handleTableItemSelectionChanged(self):
#         lSelectionModel = self.regionsMainTreeWidget.selectionModel()
#         print "___ - DEBUG ----- CDTypesEditor.handleTableItemSelectionChanged() lSelectionModel = " , lSelectionModel, " done."
#
#         # TODO if any action is necessary
#         #
#         #if len(lSelectionModel.selectedRows()):
#         #    self.deleteCellTypeButton.setEnabled(True)
#         #else:
#         #    self.deleteCellTypeButton.setEnabled(False)


    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # 
    # NOTUSED because we generate the color icon at a lower level,
    #    within the types editor class.
    # ------------------------------------------------------------------
    def NOTUSED__createColorIcon(self, color):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), color)
        painter.end()
        return QtGui.QIcon(pixmap)



    # ------------------------------------------------------------------
    # generate a QColor in a specified color
    # ------------------------------------------------------------------
    def __createQColor(self, pColor):

        # pColor could be one of many ways of defining a color,
        #    but this function returns a proper QColor type:
        return QtGui.QColor(pColor)


    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from tree items
    # ------------------------------------------------------------------
    def __handleItemChanged(self,pItem,pColumn):

        print "___ - DEBUG ----- CDTypesEditor.__handleItemChanged() pItem,pColumn =" , pItem,pColumn
        # update the dict:
        self.__updateTypesDictFromEditor()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from table cells built with setCellWidget() :
    # ------------------------------------------------------------------
    def PWN_handleOneRegionTableWidgetChanged(self):

        # update the dict:
        self.__updateTypesDictFromEditor()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

        print "___ - DEBUG ----- CDTypesEditor.handleOneRegionTableWidgetChanged() done."










    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def PWN___miInitCentralTreeWidget(self):

        CDConstants.printOut( "    - DEBUG ----- CDTypesEditor.__miInitCentralTreeWidget() starting.", CDConstants.DebugExcessive )

        # -------------------------------------------
        # place the table in the Panel's vbox layout:
        # one cell information widget, containing a vbox layout, in which to place the table:
        tableContainerWidget = QtGui.QWidget()
# 
#         # this topFiller part is cosmetic and could safely be removed:
#         topFiller = QtGui.QWidget()
#         topFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
# 
#         # this infoLabel part is cosmetic and could safely be removed,
#         #     unless useful info is provided here:
#         self.infoLabel = QtGui.QLabel()
#         self.infoLabel.setText("<i>colors</i> in the Cell Scene correspond to <i>region types</i>")
#         self.infoLabel.setAlignment = QtCore.Qt.AlignCenter
# #         self.infoLabel.setLineWidth(3)
# #         self.infoLabel.setMidLineWidth(3)
#         self.infoLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
# 
#         # this bottomFiller part is cosmetic and could safely be removed:
#         bottomFiller = QtGui.QWidget()
#         bottomFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
# 
        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        # vbox.addWidget(topFiller)
        vbox.addWidget(self.theTypesTree)
#         vbox.addWidget(self.infoLabel)
        # vbox.addWidget(bottomFiller)
        # finally place the complete layout in a QWidget and return it:
        tableContainerWidget.setLayout(vbox)
        return tableContainerWidget

        CDConstants.printOut( "    - DEBUG ----- CDTypesEditor.__miInitCentralTreeWidget() done.", CDConstants.DebugExcessive )

    # end of   def __miInitCentralTreeWidget(self)
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def PWN_NOTUSED__miInitCentralTableWidget(self):
    
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





# end of class CDTypesEditor(QtGui.QWidget)
# ======================================================================














# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDTypesEditor) case it is simply used for testing:
# ======================================================================
if __name__ == '__main__':


    CDConstants.printOut( " 001 - DEBUG - mi __main__ xe 01 ", CDConstants.DebugAll )
    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)

    CDConstants.printOut( " 002 - DEBUG - mi __main__ xe 02 ", CDConstants.DebugAll )

    # the cell type/colormap panel:
    mainPanel = CDTypesEditor()

    CDConstants.printOut( " 003 - DEBUG - mi __main__ xe 03 ", CDConstants.DebugAll )

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
                       [  [QtGui.QColor(QtCore.Qt.green), "NonCondensing", 1.0, 100], \
                          [QtGui.QColor(QtCore.Qt.green), "Condensing", 2.0, 50]  ]   ], \
                    6: [ QtGui.QColor(QtCore.Qt.magenta), "magenta", [4, 3, 7], 1, \
                       [  [QtGui.QColor(QtCore.Qt.magenta), "DifferentiatedCondensing", 1.0, 100]  ]   ]   }  )
    mainPanel.setTypesDict(miDict)
    mainPanel.populateEditorFromTypesDict()

    # show() and raise_() have to be called here:
    mainPanel.show()
    # raise_() is a necessary workaround to a PyQt-caused (or Qt-caused?) bug on Mac OS X:
    #   unless raise_() is called after show(), the window/panel/etc will NOT become
    #   the foreground window and won't receive user input focus:
    mainPanel.raise_()
    mainPanel.resize(600,600)
    CDConstants.printOut( " 004 - DEBUG - mi __main__ xe 04 ", CDConstants.DebugAll )

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
    CDConstants.printOut( " 005 - DEBUG - mi __main__ xe 05 ", CDConstants.DebugAll )

# end if __name__ == '__main__'
# Local Variables:
# coding: US-ASCII
# End:
