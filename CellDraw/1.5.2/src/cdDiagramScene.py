#!/usr/bin/env python

# This is only needed for Python v2 but is harmless for Python v3.
#
# 2010 - Mitja: apparently this is only necessary if run as main class:
if __name__ == '__main__':
    import sip
    sip.setapi('QString', 2)


# 2010 - Mitja:
import inspect # for debugging functions, remove in final version
# debugging functions, remove in final version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]



# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
#
# def dumpstacks(signal, frame):
#     id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
#     code = []
#     for threadId, stack in sys._current_frames().items():
#         code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
#         for filename, lineno, name, line in traceback.extract_stack(stack):
#             code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
#             if line:
#                 code.append("  %s" % (line.strip()))
#     CDConstants.printOut(str("\n".join(code)) , CDConstants.DebugTODO )


# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
#
# import signal
# 
# signal.signal(signal.SIGQUIT, dumpstacks)









import sys    # sys is necessary to inquire about "sys.platform" and "sys.version_info"

import math

# 2010 - Mitja: for cut/copy/paste operations on scene items,
#   we would need deepcopy, if it worked with Qt objects, but it doesn't.
# import copy

from PyQt4 import QtCore, QtGui

# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
import PyQt4.QtCore
import PyQt4.QtGui
import PyQt4
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# 2011 - Mitja: external class for drawing an image layer on a QGraphicsScene:
from cdImageLayer import CDImageLayer

# 2011 - Mitja: external class for handling a sequence of images:
from cdImageSequence import CDImageSequence

# 2011 - Mitja: external class for buttons, labels, etc:
from cdControlPanel import CDControlPanel

# 2011 - Mitja: external class for controlling image picking mode: buttons/sliders:
from cdControlInputImage import CDControlInputImage

# 2011 - Mitja: external class for accessing image sequence controls:
from cdControlImageSequence import CDControlImageSequence

# 2011 - Mitja: external class for accessing clusters controls:
from cdControlClusters import CDControlClusters

# 2011 - Mitja: external class for controlling drawing toggle: regions vs. cells:
from cdControlRegionOrCell import CDControlRegionOrCell

# 2011 - Mitja: external class for selecting layer mode:
from cdControlLayerSelection import CDControlLayerSelection

# 2011 - Mitja: external class for scene scale / zoom control:
from cdControlSceneScaleZoom import CDControlSceneScaleZoom

# 2011 - Mitja: external class for scene item edit controls:
from cdControlSceneItemEdit import PIFControlSceneItemEdit

# 2011 - Mitja: external class for setting types of regions and cells:
from cdControlTypes import CDControlTypes

# 2011 - Mitja: # CDSceneAssistant - starting assistant/wizard for CellDraw
from cdSceneAssistant import CDSceneAssistant



# the following import is about a resource file generated thus:
#     "pyrcc4 cdDiagramScene.qrc -o cdDiagramScene_rc.py"
# which requires the file cdDiagramScene.qrc to correctly point to the files in ":/images"
# only that way will the icons etc. be available after the import:
import cdDiagramScene_rc


# ------------------------------------------------------------
# ------------------------------------------------------------
class Arrow(QtGui.QGraphicsLineItem):
    def __init__(self, startItem, endItem, parent=None, scene=None):
        super(Arrow, self).__init__(parent, scene)

        self.arrowHead = QtGui.QPolygonF()

        self.myStartItem = startItem
        self.myEndItem = endItem
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.myColor = QtCore.Qt.black
        self.setPen(QtGui.QPen(self.myColor, 2.0, QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

    def setColor(self, color):
        self.myColor = color

    def startItem(self):
        return self.myStartItem

    def endItem(self):
        return self.myEndItem

    def boundingRect(self):
        extra = (self.pen().width() + 20) / 2.0
        p1 = self.line().p1()
        p2 = self.line().p2()
        return QtCore.QRectF(p1, QtCore.QSizeF(p2.x() - p1.x(), p2.y() - p1.y())).normalized().adjusted(-extra, -extra, extra, extra)

    def shape(self):
        path = super(Arrow, self).shape()
        path.addPolygon(self.arrowHead)
        return path

    def updatePosition(self):
        line = QtCore.QLineF(self.mapFromItem(self.myStartItem, 0, 0), self.mapFromItem(self.myEndItem, 0, 0))
        self.setLine(line)

    def paint(self, painter, option, widget=None):
        if (self.myStartItem.collidesWithItem(self.myEndItem)):
            return

        myStartItem = self.myStartItem
        myEndItem = self.myEndItem
        myColor = self.myColor
        myPen = self.pen()
        myPen.setColor(self.myColor)
        arrowSize = 20.0
        painter.setPen(myPen)
        painter.setBrush(self.myColor)

        centerLine = QtCore.QLineF(myStartItem.pos(), myEndItem.pos())
        endPolygon = myEndItem.polygon()
        p1 = endPolygon.first() + myEndItem.pos()

        intersectPoint = QtCore.QPointF()
        for i in endPolygon:
            p2 = i + myEndItem.pos()
            polyLine = QtCore.QLineF(p1, p2)
            intersectType = polyLine.intersect(centerLine, intersectPoint)
            if intersectType == QtCore.QLineF.BoundedIntersection:
                break
            p1 = p2

        self.setLine(QtCore.QLineF(intersectPoint, myStartItem.pos()))
        line = self.line()

        angle = math.acos(line.dx() / line.length())
        if line.dy() >= 0:
            angle = (math.pi * 2.0) - angle

        arrowP1 = line.p1() + QtCore.QPointF(math.sin(angle + math.pi / 3.0) * arrowSize,
                                        math.cos(angle + math.pi / 3) * arrowSize)
        arrowP2 = line.p1() + QtCore.QPointF(math.sin(angle + math.pi - math.pi / 3.0) * arrowSize,
                                        math.cos(angle + math.pi - math.pi / 3.0) * arrowSize)

        self.arrowHead.clear()
        for point in [line.p1(), arrowP1, arrowP2]:
            self.arrowHead.append(point)

        painter.drawLine(line)
        painter.drawPolygon(self.arrowHead)
        if self.isSelected():
            painter.setPen(QtGui.QPen(myColor, 1.0, QtCore.Qt.DashLine))
            myLine = QtCore.QLineF(line)
            myLine.translate(0, 4.0)
            painter.drawLine(myLine)
            myLine.translate(0,-8.0)
            painter.drawLine(myLine)


# ------------------------------------------------------------
# ------------------------------------------------------------
class DiagramTextItem(QtGui.QGraphicsTextItem):
    signalLostFocus = QtCore.pyqtSignal(QtGui.QGraphicsTextItem)
    signalSelectedChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)

    def __init__(self, parent=None, scene=None):
        super(DiagramTextItem, self).__init__(parent, scene)

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

    def itemChange(self, change, value):
        if change == QtGui.QGraphicsItem.ItemSelectedChange:
            self.signalSelectedChange.emit(self)
        return value

    def focusOutEvent(self, event):
        self.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.signalLostFocus.emit(self)
        super(DiagramTextItem, self).focusOutEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.textInteractionFlags() == QtCore.Qt.NoTextInteraction:
            self.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
        super(DiagramTextItem, self).mouseDoubleClickEvent(event)


# ------------------------------------------------------------
# 2010 - Mitja: this is the main item class for the diagram scene:
# ------------------------------------------------------------
# (the QGraphicsPolygonItem class is a child of QGraphicsItem)
class DiagramItem(QtGui.QGraphicsPolygonItem):
    # 2010 - Mitja: these are possible types for the diagram type:
    #    originally 4 types:    RectangleConst, TenByTenBoxConst, StartEndConst, TwoByTwoBoxConst = range(4)
    RectangleConst, TenByTenBoxConst, StartEndConst, TwoByTwoBoxConst, PathConst = range(5)

    def __init__(self, pDiagramType, contextMenu, parent=None, scene=None):
        super(DiagramItem, self).__init__(parent, scene)

        CDConstants.printOut("DiagramItem DIAGRAMITEM debugWhoIsTheRunningFunction() is "+str(debugWhoIsTheRunningFunction())+" debugWhoIsTheParentFunction() is "+str(debugWhoIsTheParentFunction()), CDConstants.DebugTODO )

        if (CDConstants.globalDebugLevel >= CDConstants.DebugAll):
            import traceback
            traceback.print_stack()

        #  self.setFillRule(QtCore.Qt.WindingFill) from Qt documentation:
        # Specifies that the region is filled using the non zero winding rule.
        # With this rule, we determine whether a point is inside the shape by
        # using the following method. Draw a horizontal line from the point to a
        # location outside the shape. Determine whether the direction of the line
        # at each intersection point is up or down. The winding number is
        # determined by summing the direction of each intersection. If the number
        # is non zero, the point is inside the shape. This fill mode can also in
        # most cases be considered as the intersection of closed shapes.

        self.setFillRule(QtCore.Qt.WindingFill)

        self.arrows = []
        # store each item's scaling factors separately for X and Y
        self.myScaleX = 1.0
        self.myScaleY = 1.0
        # 2011 - Mitja: store a backup copy of the item's pen and brush:
        self.bakPen = QtGui.QPen()
        self.bakBrush = QtGui.QBrush()
        # 2011 - Mitja: add a unique counter to identify each DiagramItem in the scene:
        self.regionID = 0

        #   is this item a region of cells or a single cell?
        self.itsaRegionOrCell = CDConstants.ItsaRegionConst
        self.setRegionOrCell(self.itsaRegionOrCell)

        # 2010 - Mitja: add code for handling insertion of Path items:
        # CDConstants.printOut( " "+str( "type(pDiagramType).__name__ = ", type(pDiagramType).__name__ )+" ", CDConstants.DebugTODO )
        if (type(pDiagramType).__name__ == "int") :
            # we are instantiating a normal type of diagram item:
            self.diagramType = pDiagramType

        else:
#         else (type(pDiagramType).__name__ is "QPainterPath"):
            CDConstants.printOut("DEBUG DEBUG ----- DiagramItem(): type(pDiagramType).__name__ = "+str(type(pDiagramType).__name__), CDConstants.DebugTODO )
            # since we received a QPainterPath parameter       
            # we are instantiating a Path type of diagram item:
            self.diagramType = DiagramItem.PathConst
            self.thePathToBuildAPolygon = pDiagramType

        self.contextMenu = contextMenu

        lThisIsAnUnusedStartEndPath = QtGui.QPainterPath()
        if self.diagramType == self.StartEndConst:
            lThisIsAnUnusedStartEndPath.moveTo(200, 50)
            lThisIsAnUnusedStartEndPath.arcTo(150, 0, 50, 50, 0, 90)
            lThisIsAnUnusedStartEndPath.arcTo(50, 0, 50, 50, 90, 90)
            lThisIsAnUnusedStartEndPath.arcTo(50, 50, 50, 50, 180, 90)
            lThisIsAnUnusedStartEndPath.arcTo(150, 50, 50, 50, 270, 90)
            lThisIsAnUnusedStartEndPath.lineTo(200, 25)
            self.myPolygon = lThisIsAnUnusedStartEndPath.toFillPolygon()
        elif self.diagramType == self.TenByTenBoxConst:
            self.myPolygon = QtGui.QPolygonF([
                    QtCore.QPointF(-5, -5), QtCore.QPointF(-5, 5),
                    QtCore.QPointF(5, 5), QtCore.QPointF(5, -5),
                    QtCore.QPointF(-5, -5)])
#                     QtCore.QPointF(-100, 0), QtCore.QPointF(0, 100),
#                     QtCore.QPointF(100, 0), QtCore.QPointF(0, -100),
#                     QtCore.QPointF(-100, 0)])
        elif self.diagramType == self.RectangleConst:
            self.myPolygon = QtGui.QPolygonF([
                    QtCore.QPointF(-50, -50), QtCore.QPointF(50, -50),
                    QtCore.QPointF(50, 50), QtCore.QPointF(-50, 50),
                    QtCore.QPointF(-50, -50)])
#                     QtCore.QPointF(-100, -100), QtCore.QPointF(100, -100),
#                     QtCore.QPointF(100, 100), QtCore.QPointF(-100, 100),
#                     QtCore.QPointF(-100, -100)])
        elif self.diagramType == self.TwoByTwoBoxConst:
            self.myPolygon = QtGui.QPolygonF([
                    QtCore.QPointF(-1, -1), QtCore.QPointF(-1, 1),
                    QtCore.QPointF(1, 1), QtCore.QPointF(1, -1),
                    QtCore.QPointF(-1, -1)])
#             self.myPolygon = QtGui.QPolygonF([
#                     QtCore.QPointF(-120, -80), QtCore.QPointF(-70, 80),
#                     QtCore.QPointF(120, 80), QtCore.QPointF(70, -80),
#                     QtCore.QPointF(-120, -80)])
        elif self.diagramType == self.PathConst:
            # CDConstants.printOut( " "+str( "self.thePathToBuildAPolygon =", self.thePathToBuildAPolygon )+" ", CDConstants.DebugTODO )
            # convert from painter path to polygon:
            self.thePathToBuildAPolygon = self.thePathToBuildAPolygon.simplified()

            self.myPolygon = self.thePathToBuildAPolygon.toFillPolygon()

            # 2010 - Mitja: create a Path item here:
            # self.setPath(self.thePathToBuildAPolygon)

        # build a temporary QGraphicsPolygonItem and assign the constructed polygon to it:
#         lTempGraphicsItem = QtGui.QGraphicsPolygonItem()
#         lTempGraphicsItem.setPolygon(self.myPolygon)
        # self.setPolygon(self.myPolygon)

#         CDConstants.printOut( " "+str( "tmp polyg (diagramType=", self.diagramType,").boundingRect =",lTempGraphicsItem.polygon().boundingRect() )+" ", CDConstants.DebugTODO )
#         CDConstants.printOut( " "+str( "          hello, I'm %s, parent is %s" % (debugWhoIsTheRunningFunction(), debugWhoIsTheParentFunction()) )+" ", CDConstants.DebugTODO )
#         CDConstants.printOut( " "+str( "          boundingRegionGranularity =", lTempGraphicsItem.boundingRegionGranularity() )+" ", CDConstants.DebugTODO )
#         lTempGraphicsItem.setBoundingRegionGranularity(1.0)
#         CDConstants.printOut( " "+str( "          boundingRegionGranularity =", lTempGraphicsItem.boundingRegionGranularity() )+" ", CDConstants.DebugTODO )
#        
#         # obtain the QGraphicsPolygonItem's bounding QRegion in local coordinates, passing an identity QTransform:
#         lBoundingRegion = lTempGraphicsItem.boundingRegion(QtGui.QTransform())
#         CDConstants.printOut( " "+str( "          boundingRegion =", lBoundingRegion )+" ", CDConstants.DebugTODO )
#         CDConstants.printOut( " "+str( "          boundingRegion.boundingRect() =", lBoundingRegion.boundingRect() )+" ", CDConstants.DebugTODO )
#         CDConstants.printOut( " "+str( "          boundingRegion.rectCount() =", lBoundingRegion.rectCount() )+" ", CDConstants.DebugTODO )

        # create a QPainterPath from the bounding QRegion:
#         lNewPainterPath = QtGui.QPainterPath()
#         lNewPainterPath.addRegion(lBoundingRegion)
#        
#         lNewPainterPath = lNewPainterPath.simplified()
#        
#         CDConstants.printOut( " "+str( "          lNewPainterPath =", lNewPainterPath )+" ", CDConstants.DebugTODO )
#         CDConstants.printOut( " "+str( "          lNewPainterPath.boundingRect() =", lNewPainterPath.boundingRect() )+" ", CDConstants.DebugTODO )

        # re-define the QGraphicsPolygonItem's QPolygonF from the newly built QPainterPath:
        # self.setPolygon = QtGui.QPolygonF(QtCore.QRectF(10, 10, 100, 200))
        # tmpPolygon = lNewPainterPath.toFillPolygon()
#        self.setPolygon(tmpPolygon)


        # 2011 - Mitja: re-center the incoming path so that its bounding rectangle
        #   is 0-centered, and instead add a displacement to the item if necessary.

        # extract the boundingRect() to center the polygon's individual points()
        lBoundingRect = self.myPolygon.boundingRect()
        lCenterOfItemX = (lBoundingRect.width() * 0.5) + lBoundingRect.topLeft().x()
        lCenterOfItemY = (lBoundingRect.height() * 0.5) + lBoundingRect.topLeft().y()

        CDConstants.printOut("in DiagramItem._init_(): the bounding rect is = "+str(lBoundingRect) , CDConstants.DebugVerbose )
        CDConstants.printOut("in DiagramItem._init_(): the rect's center is = "+str(lCenterOfItemX)+" "+str(lCenterOfItemY) , CDConstants.DebugVerbose )

        lZeroCenteredPolygon = QtGui.QPolygonF()
        for lPointF in self.myPolygon:
            lZeroCenteredPointF = QtCore.QPointF( ( lPointF.x() - lCenterOfItemX ),
                                                  ( lPointF.y() - lCenterOfItemY )  )
            # CDConstants.printOut( " "+str( "DiagramItem_init_(): lPointF, lZeroCenteredPointF =", lPointF, lZeroCenteredPointF )+" ", CDConstants.DebugVerbose )
            lZeroCenteredPolygon.append(lZeroCenteredPointF)


        # finally assign the fixed QPolygonF to the QGraphicsPolygonItem:
        self.setPolygon(lZeroCenteredPolygon)
        self.setPos( QtCore.QPointF( lCenterOfItemX, lCenterOfItemY ) )

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        CDConstants.printOut("          polygon (diagramType="+str(self.diagramType)+").boundingRect ="+str(self.polygon().boundingRect()) , CDConstants.DebugVerbose )
        CDConstants.printOut("          item (diagramType="+str(self.diagramType)+").boundingRect ="+str(self.boundingRect()) , CDConstants.DebugVerbose )
        CDConstants.printOut("          item (diagramType="+str(self.diagramType)+").pen ="+str(self.pen()) , CDConstants.DebugVerbose )
        CDConstants.printOut("          item (diagramType="+str(self.diagramType)+").brush ="+str(self.brush()) , CDConstants.DebugVerbose )

    # 2010 - Mitja: we could override paint:
#     def paint(self, pPainter, pStyleOptionGraphicsItem, pWidget=None):
#         super(DiagramItem, self).paint(pPainter, pStyleOptionGraphicsItem, pWidget)
#         lBoundingRect = self.boundingRect()
#         CDConstants.printOut( " "+str( "self.scene().myOutlineResizingItem =", self.scene().myOutlineResizingItem )+" ", CDConstants.DebugTODO )
#         self.scene().myOutlineResizingItem = self
#         CDConstants.printOut( " "+str( "self.scene().myOutlineResizingItem =", self.scene().myOutlineResizingItem )+" ", CDConstants.DebugTODO )
#         # painter->drawRoundedRect(-10, -10, 20, 20, 5, 5);
        CDConstants.printOut("--------------------------------------------- DiagramItem.__init__() end", CDConstants.DebugExcessive )
    # end of    def __init__(self, pDiagramType, contextMenu, parent=None, scene=None)
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # 2011 - Mitja: for each scene item, is this item a region of cells or a single cell?
    # ------------------------------------------------------------------
    def setRegionOrCell(self, pRegionOrCell):

        self.itsaRegionOrCell = pRegionOrCell
       
        #   is this item a region of cells or a single cell?
        if (self.itsaRegionOrCell == CDConstants.ItsaRegionConst):
            # for region items, use darkMagenta pen:
            CDConstants.printOut("setRegionOrCell does CDConstants.ItsaRegionConst", CDConstants.DebugTODO )
            lMyPen = QtGui.QPen(QtCore.Qt.darkMagenta, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        elif (self.itsaRegionOrCell == CDConstants.ItsaCellConst):
            # for cell items, use an orange "#FF9900" or (255, 153, 0) pen:
            CDConstants.printOut("setRegionOrCell does CDConstants.ItsaCellConst", CDConstants.DebugTODO )
            lMyCellOutlineColor = QtGui.QColor(255, 153, 0)
            lMyPen = QtGui.QPen(lMyCellOutlineColor, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        else:
            # for unknown items, use darkRed pen:
            CDConstants.printOut("setRegionOrCell does darkRed for UNKNOWN items", CDConstants.DebugTODO )
            lMyPen = QtGui.QPen(QtCore.Qt.darkRed, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
       
        # 2011 - Mitja: according to Qt's documentation: "QGraphicsItem
        #               does not support use of cosmetic pens with a non-zero width" :
        lMyPen.setCosmetic(True)

        CDConstants.printOut("setRegionOrCell does item.pen().color().rgba() = "+str(self.pen().color().rgba()), CDConstants.DebugTODO )
        self.setPen(lMyPen)
        CDConstants.printOut("setRegionOrCell does item.pen().color().rgba() = "+str(self.pen().color().rgba()), CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def setRegionID(self, pID):
        self.regionID = pID
        CDConstants.printOut("DiagramItem.setRegionID("+str(self.regionID)+")", CDConstants.DebugTODO )

    # ------------------------------------------------------------------
    def getRegionID(self):
        # CDConstants.printOut( " "+str( "DiagramItem.getRegionID() =",self.regionID )+" ", CDConstants.DebugTODO )
        return (self.regionID)

    # ------------------------------------------------------------------
    def saveAndClearPen(self):
        # 2011 - Mitja: store a backup copy of the item's pen:
        self.bakPen = QtGui.QPen(self.pen())
        # 2011 - Mitja: create and assign an invisible pen:
        lMyPen = QtGui.QPen(QtCore.Qt.transparent, 0.0)
        lMyPen.setCosmetic(True)
        self.setPen(lMyPen)

    # ------------------------------------------------------------------
    def restorePen(self):
        # 2011 - Mitja: restore the backup copy to the item's active pen:
        self.setPen(self.bakPen)

    # ------------------------------------------------------------------
    def saveAndClearBrush(self):
        # 2011 - Mitja: store a backup copy of the item's Brush:
        self.bakBrush = QtGui.QBrush(self.brush())
        # 2011 - Mitja: create and assign a plain black Brush:
        lMyBrush = QtGui.QBrush(QtGui.QColor(QtCore.Qt.black))
        self.setBrush(lMyBrush)

    # ------------------------------------------------------------------
    def restoreBrush(self):
        # 2011 - Mitja: restore the backup copy to the item's active Brush:
        self.setBrush(self.bakBrush)

    # ------------------------------------------------------------------
    def removeArrow(self, arrow):
        try:
            self.arrows.remove(arrow)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    def removeArrows(self):
        for arrow in self.arrows[:]:
            arrow.startItem().removeArrow(arrow)
            arrow.endItem().removeArrow(arrow)
            self.scene().removeItem(arrow)

    # ------------------------------------------------------------------
    def addArrow(self, arrow):
        self.arrows.append(arrow)


    # ------------------------------------------------------------------
    # 2010 - Mitja: renamed this method from "image" to "pixmapForIconFromPolygon"
    #    since it returns a pixmap, NOT an image!!!
    # ------------------------------------------------------------------
    def pixmapForIconFromPolygon(self, pText=None):
        pixmap = QtGui.QPixmap(210, 210)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 8.0))
        painter.translate(105, 105)

        #  self.setFillRule(QtCore.Qt.WindingFill) from Qt documentation:
        # Specifies that the region is filled using the non zero winding rule.
        # With this rule, we determine whether a point is inside the shape by
        # using the following method. Draw a horizontal line from the point to a
        # location outside the shape. Determine whether the direction of the line
        # at each intersection point is up or down. The winding number is
        # determined by summing the direction of each intersection. If the number
        # is non zero, the point is inside the shape. This fill mode can also in
        # most cases be considered as the intersection of closed shapes.

        painter.drawPolygon(self.polygon(), QtCore.Qt.WindingFill)
        
        if (pText != None):
#            painter.translate(-110, 100)
            painter.setFont(QtGui.QFont("Helvetica", 72))
            painter.drawText(QtCore.QRectF(-105,30,210,100), QtCore.Qt.AlignCenter, pText)
        
        painter.end()

        return pixmap


    # ------------------------------------------------------------------
    def contextMenuEvent(self, event):
        # 2010 - Mitja: nothing to be done as "context menu event" here at the moment:
        # self.scene().clearSelection()
        # self.setSelected(True)
        # self.myContextMenu.exec_(event.screenPos())
        pass

    # ------------------------------------------------------------------
    def itemChange(self, change, value):
        # CDConstants.printOut( " "+str( "itemChange() itemChange() itemChange() itemChange() itemChange() itemChange() change, value =", change, value )+" ", CDConstants.DebugTODO )
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            for arrow in self.arrows:
                arrow.updatePosition()

        return value

    # ---------------------------------------------------------
    # end of class DiagramItem(QtGui.QGraphicsPolygonItem)
    # ---------------------------------------------------------

# ------------------------------------------------------------
# 2010 - Mitja: add code for handling insertion of pixmap items:
#   this class is UNUSED and can be safely removed in production code.
# ------------------------------------------------------------
class DiagramPixmapItem(QtGui.QGraphicsPixmapItem):

    # ---------------------------------------------------------
    def __init__(self, pPixmap, pContextMenu, pParent=None, pScene=None):
        super(DiagramPixmapItem, self).__init__(pParent, pScene)

        self.arrows = []

        self.itemPixmap = pPixmap
        self.contextMenu = pContextMenu

#         path = QtGui.QPainterPath()
#         self.myPolygon = QtGui.QPolygonF([
#                 QtCore.QPointF(-120, -80), QtCore.QPointF(-70, 80),
#                 QtCore.QPointF(120, 80), QtCore.QPointF(70, -80),
#                 QtCore.QPointF(-120, -80)])

        # 2010 - Mitja: create a pixmap item here:
        self.setPixmap(self.itemPixmap)

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

    # ---------------------------------------------------------
    def removeArrow(self, arrow):
        try:
            self.arrows.remove(arrow)
        except ValueError:
            pass

    # ---------------------------------------------------------
    def removeArrows(self):
        for arrow in self.arrows[:]:
            arrow.startItem().removeArrow(arrow)
            arrow.endItem().removeArrow(arrow)
            self.scene().removeItem(arrow)

    # ---------------------------------------------------------
    def addArrow(self, arrow):
        self.arrows.append(arrow)

    # ---------------------------------------------------------
    def pixmapForIconFromPolygon(self):

        # 1. get a copy of the QGraphicsPixmapItem's pixmap:
        lOriginalPixmap = QtGui.QPixmap.fromImage(self.pixmap().toImage())

        # 3. create an empty pixmap where to store the composed image:
        lResultPixmap = QtGui.QPixmap(250, 250)
        # 4. create a QPainter to perform the overlay operation:
        lPainter = QtGui.QPainter(lResultPixmap)
        # 5. do the overlay:
        lPainter.setCompositionMode(lPainter.CompositionMode_Source)
        lPainter.fillRect(lResultPixmap.rect(), QtCore.Qt.transparent)
        lPainter.setCompositionMode(lPainter.CompositionMode_SourceOver)
        lPainter.drawPixmap(QtCore.QPoint(0,0), lOriginalPixmap)
        lPainter.end()           
       
        return lResultPixmap


    # ---------------------------------------------------------
    def contextMenuEvent(self, event):
        self.scene().clearSelection()
        self.setSelected(True)
        self.myContextMenu.exec_(event.screenPos())

    # ---------------------------------------------------------
    def itemChange(self, change, value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            for arrow in self.arrows:
                arrow.updatePosition()

        return value
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # end of class DiagramPixmapItem(QtGui.QGraphicsPixmapItem)
    # ---------------------------------------------------------


# ------------------------------------------------------------
# ------------------------------------------------------------
class DiagramScene(QtGui.QGraphicsScene):

    signalThatItemInserted = QtCore.pyqtSignal(DiagramItem)

    signalThatTextInserted = QtCore.pyqtSignal(QtGui.QGraphicsTextItem)

    signalThatItemSelected = QtCore.pyqtSignal(QtGui.QGraphicsItem)

    # 2011 - Mitja:
    signalThatItemResized = QtCore.pyqtSignal(QtCore.QRectF)

    # 2011 - Mitja: add a signal for scene resizing. Has to be handled well!
    signalThatSceneResized = QtCore.pyqtSignal(dict)


    def __init__(self, pEditMenu, pParent=None):
        super(DiagramScene, self).__init__(pParent)

        self.parentWidget = pParent
        self.myItemMenu = pEditMenu
        self.mySceneMode = CDConstants.SceneModeMoveItem
        if self.parentWidget.parentWindow != None:
            self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor")

        # 2010 - Mitja: to resize an item by using the right-click,
        #   add a global variable flag, i.e. direct pointer to the item being resized:
        self.myRightButtonResizingItem = None
        self.myRightButtonResizingClickX = -1.0
        self.myRightButtonResizingClickY = -1.0

        # 2010 - Mitja: to resize an item by using an outline of the item's bounding rectangle,
        #   add a global variable flag, i.e. direct pointer to the item being resized:
        self.myOutlineResizingItem = None
        #   myOutlineResizingVertex can be: "None", "bottomLeft", "bottomRight", "topRight", "topLeft":
        self.myOutlineResizingVertex = "None"
        self.myOutlineResizingCurrentX = -1.0
        self.myOutlineResizingCurrentY = -1.0

        # 2010 - Mitja: add scene units:
        self.mySceneUnits = "Pixel"

        # 2011 - Mitja: add scene depth:
        self.mySceneDepth = 1.0


        # 2011 - Mitja: a reference to a sequence of images:
        self.theImageSequence = None


        # 2011 - Mitja: a reference to an external QPaint-drawing class,
        #   to draw an overlay in drawForeground() :
        self.theImageLayer = None
        # 2011 - Mitja: flag to temporarily completely disable drawing the overlay:
        self.isDrawForegroundEnabled = True
        # 2011 - Mitja: flag to prevent recursive repaints (shouldn't this be handled automatically by Qt?!?):
        self.isDrawForegroundInProgress = False

        # 2011 -
        # regionUseDict = a dict of all region names and their current use,
        #   one for each RGBA color: [name,#regions](color)
        self.regionUseDict = {}


        # 2011 - Mitja: set defaults for each new Item in the Scene:
        self.myItemRegionOrCell = CDConstants.ItsaRegionConst

        self.myItemType = DiagramItem.RectangleConst
        self.line = None
        self.textItem = None
        self.myItemColor = QtCore.Qt.green
        self.myTextColor = QtCore.Qt.red
        self.myLineColor = QtCore.Qt.black
        self.myFont = QtGui.QFont()
       
        # 2011 - Mitja: an ever-incrementing totalItemsCounter, to keep a unique
        #   string as toolTip for each item in the scene.
        self.totalItemsCounter = int(0)
        
        # 2011 Mitja - add scale/zoom parameters stored into our DiagramScene object:
        # This is just so that we can scale the outline's circles/ellipses
        #    when interactively/manually scaling a scene item.
        # The scale factor takes care of the RHS <-> LHS mismatch at its visible end,
        #    by flipping the y coordinate in the QGraphicsView's affine transformations:
        self.myViewScaleZoomX =  1.0
        self.myViewScaleZoomY = -1.0



    # ------------------------------------------------------------
    # 2011 - Mitja: set the depth value for scene, to keep for completeness:
    # ------------------------------------------------------------
    def setDepth(self, pValue = 1.0):
        # 2011 - Mitja - add scene depth:
        self.mySceneDepth = pValue

    # ------------------------------------------------------------
    # 2011 - Mitja: set the depth value for scene, to keep for completeness:
    # ------------------------------------------------------------
    def depth(self):
        # 2011 - Mitja - add scene depth:
        return self.mySceneDepth


    # ------------------------------------------------------------
    # 2011 - Mitja: set the reference for overlay drawing in drawForeground() :
    # ------------------------------------------------------------
    def setImageLayer(self, pImageLayer = None):
        if isinstance( pImageLayer, CDImageLayer ) == True:
            self.theImageLayer = pImageLayer



    # ------------------------------------------------------------
    # 2011 - Mitja: set the reference for a sequence of images:
    # ------------------------------------------------------------
    def setImageSequence(self, pImageSequence = None):
        if isinstance( pImageSequence, CDImageSequence ) == True:
            self.theImageSequence = pImageSequence



    # ------------------------------------------------------------
    # 2011 - Mitja: set flag to temporarily completely disable drawing the overlay:
    # ------------------------------------------------------------
    def setDrawForegroundEnabled(self, pYesOrNo = True):
        self.isDrawForegroundEnabled = pYesOrNo
        CDConstants.printOut("DiagramScene.setDrawForegroundEnabled( pYesOrNo=="+str(pYesOrNo)+" )", CDConstants.DebugTODO )



    # ------------------------------------------------------------
    # 2012 - Mitja: retrieve the flag to temporarily completely disable drawing the overlay:
    # ------------------------------------------------------------
    def getDrawForegroundEnabled(self):
        CDConstants.printOut("DiagramScene.getDrawForegroundEnabled() returning " +str(self.isDrawForegroundEnabled), CDConstants.DebugTODO )
        return self.isDrawForegroundEnabled



    # ------------------------------------------------------------
    # 2010 - Mitja - we reimplement drawForeground() because
    #   we need to draw an outline of the selected item on the top (foreground) of the scene
    # ------------------------------------------------------------
    def drawForeground(self, pPainter, pRect):
        super(DiagramScene, self).drawForeground(pPainter, pRect)

        # check the flag that temporarily completely disables drawing the overlay:
        if (self.isDrawForegroundEnabled == False):
            return

        # check the flag that signals drawForeground() in progress...
        #   ....to prevent recursive repaints (shouldn't this be handled automatically by Qt?!?) :
        if (self.isDrawForegroundInProgress == True):
            return
        else:
            # set the flag that signals drawForeground() in progress:
            self.isDrawForegroundInProgress = True


        # 2011 - Mitja - use this overlay routine to draw a single instance of one cluster
        #   as foreground to the graphics scene, in "SceneModeEditCluster"
        if (self.mySceneMode == CDConstants.SceneModeEditCluster):

            # store the current pen & brush & background & bg mode to restore them later:
            lTmpPen = pPainter.pen()
            lTmpBrush = pPainter.brush()
            lTmpBackground = pPainter.background()
            lTmpBackgroundMode = pPainter.backgroundMode()

            # TODO:  replace with a new paint call for scene edit clusters!

            # this used to call the theImageSequence's paintEvent handler directly: self.theImageSequence.paintEvent(pPainter)
            #  but direct paintEvent calls are BAD! instead we now call our separate paint routine:
            # self.theImageSequence.paintTheImageSequence(pPainter)

            # restore the painter's pen & background to what they were before this function:
            pPainter.setPen(lTmpPen)
            pPainter.setBrush(lTmpBrush)
            pPainter.setBackground(lTmpBackground)
            pPainter.setBackgroundMode(lTmpBackgroundMode)


        # 2011 - Mitja - use this overlay routine to draw selected content from an image sequence
        #   as foreground to the graphics scene, in "SceneModeImageSequence"
        if (self.mySceneMode == CDConstants.SceneModeImageSequence) and \
            (self.theImageSequence != None):

            # store the current pen & brush & background & bg mode to restore them later:
            lTmpPen = pPainter.pen()
            lTmpBrush = pPainter.brush()
            lTmpBackground = pPainter.background()
            lTmpBackgroundMode = pPainter.backgroundMode()
           
            # this used to call the theImageSequence's paintEvent handler directly: self.theImageSequence.paintEvent(pPainter)
            #  but direct paintEvent calls are BAD! instead we now call our separate paint routine:
            self.theImageSequence.paintTheImageSequence(pPainter)
                       
            # restore the painter's pen & background to what they were before this function:
            pPainter.setPen(lTmpPen)
            pPainter.setBrush(lTmpBrush)
            pPainter.setBackground(lTmpBackground)
            pPainter.setBackgroundMode(lTmpBackgroundMode)



        # 2011 - Mitja - use this overlay routine to draw an image label
        #   as foreground to the graphics scene, in "image layer mode"
        elif (self.mySceneMode == CDConstants.SceneModeImageLayer) and \
            (self.theImageLayer != None):

            # store the current pen & brush & background & bg mode to restore them later:
            lTmpPen = pPainter.pen()
            lTmpBrush = pPainter.brush()
            lTmpBackground = pPainter.background()
            lTmpBackgroundMode = pPainter.backgroundMode()
           
            # this used to call the theImageLayer's paintEvent handler directly:  self.theImageLayer.paintEvent(pPainter)
            #  but direct paintEvent calls are BAD! instead we now call our separate paint routine:
            self.theImageLayer.paintTheImageLayer(pPainter)
                       
            # restore the painter's pen & background to what they were before this function:
            pPainter.setPen(lTmpPen)
            pPainter.setBrush(lTmpBrush)
            pPainter.setBackground(lTmpBackground)
            pPainter.setBackgroundMode(lTmpBackgroundMode)


        # 2010 - Mitja - draw an overlay outline of the item being resized
        #   by dragging one of the outline corner vertices:
        elif self.myOutlineResizingItem != None:
# TODO: find the true bounding rect for the item's polygon WITHOUT its border:            lBoundingRect = self.myOutlineResizingItem.polygon().sceneBoundingRect()
            lBoundingRect = self.myOutlineResizingItem.sceneBoundingRect()
            # store the current pen & brush & background & bg mode to restore them later:
            lTmpPen = pPainter.pen()
            lTmpBrush = pPainter.brush()
            lTmpBackground = pPainter.background()
            lTmpBackgroundMode = pPainter.backgroundMode()

#             lOutlinePen = QtGui.QPen( QtGui.QColor(255, 128, 0) )
#             lOutlinePen.setWidth(4)

            # draw the rectangular outline of the QGraphicsItem which is being resized:
            #   draw in two colors, solid and dotted:
            lOutlineColor = QtGui.QColor(35, 166, 94)
            # we set al QtGui.QPen sizes to 0.0 for cosmetic lines that have to remain of minimal line-width when zooming:
            lOutlinePen = QtGui.QPen(lOutlineColor, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            # lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            lOutlinePen.setCosmetic(True)
            pPainter.setPen(lOutlinePen)
            pPainter.drawRect(lBoundingRect)

            lOutlineColor = QtGui.QColor(219, 230, 249)
            lOutlinePen = QtGui.QPen(lOutlineColor, 0.0, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            lOutlinePen.setCosmetic(True)
            pPainter.setPen(lOutlinePen)
            pPainter.drawRect(lBoundingRect)

            # draw circles at the rectangular outline's vertices:

            pPainter.setBackgroundMode( QtCore.Qt.OpaqueMode )
           
            pPainter.setBrush( QtGui.QBrush( QtGui.QColor(219, 230, 249) ) )
            lOutlineColor = QtGui.QColor(35, 166, 94)
            lOutlinePen = QtGui.QPen(lOutlineColor, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            lOutlinePen.setCosmetic(True)
            pPainter.setPen(lOutlinePen)
            # 2011 Mitja - zoom the ellipses smaller as the scene zooms bigger, and vice versa
            #   so that they'll appear at the same size any time to the user resizing the items.
            pPainter.drawEllipse(lBoundingRect.bottomLeft(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            pPainter.drawEllipse(lBoundingRect.bottomRight(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            pPainter.drawEllipse(lBoundingRect.topRight(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            pPainter.drawEllipse(lBoundingRect.topLeft(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))

            pPainter.setBrush( QtGui.QBrush( QtGui.QColor(255, 255, 0) ) )
            if self.myOutlineResizingVertex == "bottomLeft":
                pPainter.drawEllipse(lBoundingRect.bottomLeft(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            elif self.myOutlineResizingVertex == "bottomRight":
                pPainter.drawEllipse(lBoundingRect.bottomRight(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            elif self.myOutlineResizingVertex == "topRight":
                pPainter.drawEllipse(lBoundingRect.topRight(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
            elif self.myOutlineResizingVertex == "topLeft":
                pPainter.drawEllipse(lBoundingRect.topLeft(), (4 / self.myViewScaleZoomX), (4 / self.myViewScaleZoomY))
               
            # restore the painter's pen & background to what they were before this function:
            pPainter.setPen(lTmpPen)
            pPainter.setBrush(lTmpBrush)
            pPainter.setBackground(lTmpBackground)
            pPainter.setBackgroundMode(lTmpBackgroundMode)
           
            # emit:
            self.signalThatItemResized.emit(lBoundingRect)



        # 2010 - Mitja - draw an overlay outline of the item being resized
        #   by right-clicking (or control-clicking, depending on Qt settings) mouse motion:
        elif self.myRightButtonResizingItem != None:
            lBoundingRect = self.myRightButtonResizingItem.sceneBoundingRect()
            lTmpPen = pPainter.pen()

            #   draw in two colors, solid and dotted:
            lOutlineColor = QtGui.QColor(224, 100, 0)
            # we set al QtGui.QPen sizes to 0.0 for cosmetic lines that have to remain of minimal line-width when zooming:
            lOutlinePen = QtGui.QPen(lOutlineColor, 0.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            # lOutlinePen = QtGui.QPen(lOutlineColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            lOutlinePen.setCosmetic(True)
            pPainter.setPen(lOutlinePen)
            pPainter.drawRect(lBoundingRect)

            lOutlineColor = QtGui.QColor(255, 220, 0)
            lOutlinePen = QtGui.QPen(lOutlineColor, 0.0, QtCore.Qt.DotLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            lOutlinePen.setCosmetic(True)
            pPainter.setPen(lOutlinePen)
            pPainter.drawRect(lBoundingRect)

            pPainter.setPen(lTmpPen)
       
        # clear the flag that signals drawForeground() in progress...
        #   ....to prevent recursive repaints (shouldn't this be handled automatically by Qt?!?) :
        self.isDrawForegroundInProgress = False

    # end of   def drawForeground(self, pPainter, pRect)
    # ------------------------------------------------------------

       
    # ------------------------------------------------------------
    # 2010 - Mitja: we could reimplement drawBackground() because
    #   we don't want a tiled image as background
    #
    #  ...we don't reimplement it for now....
    # ------------------------------------------------------------
    #     def drawBackground(self, pPainter, pRect):
    #         super(DiagramScene, self).drawBackground(pPainter, pRect)




    # ------------------------------------------------------------
    # 2010 - Mitja - we reimplement addItem() because
    #   we want to add a toolTip to each item that's added to the scene,
    # 2010 - Mitja: and because we can have a unique sequential ID for each
    #   scene item this way.
    # ------------------------------------------------------------
    def addItem(self, pGraphicsItem):
        self.totalItemsCounter = self.totalItemsCounter + 1
        pGraphicsItem.setToolTip(str(self.totalItemsCounter))
        pGraphicsItem.setRegionID(self.totalItemsCounter)
        super(DiagramScene, self).addItem(pGraphicsItem)
    # end of def addItem(self)
    # ---------------------------------------------------------



    # ------------------------------------------------------------
    # 2011 - Mitja: add another color as in use by a region in the Cell Scene:
    # ------------------------------------------------------------
    def addToRegionColorsInUse(self, pColor):
        CDConstants.printOut("==================================== addToRegionColorsInUse", CDConstants.DebugTODO )

        # retrieve the region "name(color)" dictionary from the table of regions:
        lRegionsTableDict = self.parentWidget.parentWindow.theTableOfTypes.getRegionsDict()
        lKeys = lRegionsTableDict.keys()
        # lColor is the color for which we have to update the use count:
        lColor = QtGui.QColor(pColor)
        # lCount is where we temporary place the count of regions using lColor:
        lCount = 0

        # find the name corresponding to the added color, and add it to the scene's regionUseDict:
        for i in xrange(len(lRegionsTableDict)) :

            if lRegionsTableDict[lKeys[i]][0].rgba() == lColor.rgba() :

                # check if there is already an entry for this color in the self.regionUseDict :
                if lColor.rgba() not in self.regionUseDict:
                    CDConstants.printOut("NO "+str(lColor.rgba())+" not in "+str(self.regionUseDict), CDConstants.DebugTODO )
                    # add a new entry to the self.regionUseDict local dictionary of used colors :
                    lCount = 1
                else:
                    CDConstants.printOut("YES "+str(lColor.rgba())+" in "+str(self.regionUseDict), CDConstants.DebugTODO )
                    # increment the current entry in the self.regionUseDict local dictionary of used colors:
                    lCount = 1 + self.regionUseDict[ lColor.rgba() ][1]

                self.regionUseDict[ lColor.rgba() ] = [ lRegionsTableDict[lKeys[i]][1], lCount ]
               
                CDConstants.printOut( str(self.regionUseDict[ lColor.rgba() ][0])+" "+str(self.regionUseDict[ lColor.rgba() ][1]) , CDConstants.DebugTODO )

                # signal upstream about the updated usage of this region color:
                self.parentWidget.parentWindow.theTableOfTypes.updateRegionUseOfTableElements(lColor, lCount)

        # if there is at least one color in the regionUseDict table, show the table:
#         if self.regionUseDict :
#             # we can test the dict this way because, according to the Python manual,
#             #   "any empty sequence, for example, '', (), []" is considered false.
#             self.parentWidget.signalVisibilityPIFRegionTable.emit("Show")
        CDConstants.printOut("addToRegionColorsInUse - regionUseDict = "+str(self.regionUseDict) , CDConstants.DebugTODO )
        CDConstants.printOut("==================================== addToRegionColorsInUse", CDConstants.DebugTODO )


    # ------------------------------------------------------------
    # 2011 - Mitja: subtract a color from its use by a region in the Cell Scene:
    # ------------------------------------------------------------
    def subtractFromRegionColorsInUse(self, pColor):
        CDConstants.printOut("==================================== subtractFromRegionColorsInUse", CDConstants.DebugTODO )

        # retrieve the region "name(color)" dictionary from the table of regions:
        lRegionsTableDict = self.parentWidget.parentWindow.theTableOfTypes.getRegionsDict()
        lKeys = lRegionsTableDict.keys()
        # lColor is the color for which we have to update the use count:
        lColor = QtGui.QColor(pColor)
        # lCount is where we temporary place the count of regions using lColor:
        lCount = 0

        # find the name corresponding to the added color, and add it to the scene's regionUseDict:
        for i in xrange(len(lRegionsTableDict)) :
             
            if lRegionsTableDict[lKeys[i]][0].rgba() == lColor.rgba() :

                # check if there is already an entry for this color in the self.regionUseDict :
                if lColor.rgba() not in self.regionUseDict:
                    CDConstants.printOut("NO "+str(lColor.rgba())+" not in "+str(self.regionUseDict) , CDConstants.DebugTODO )
                else:
                    CDConstants.printOut("YES "+str(lColor.rgba())+" in "+str(self.regionUseDict) , CDConstants.DebugTODO )
                    # decrement the current entry in the self.regionUseDict local dictionary of used colors:
                    lCount = -1 + self.regionUseDict[ lColor.rgba() ][1]
                    if (lCount < 1):
                        del self.regionUseDict[ lColor.rgba() ]
                        CDConstants.printOut("self.regionUseDict has now no lColor.rgba() = "+str(lColor.rgba()) , CDConstants.DebugTODO )
                    else:
                        self.regionUseDict[ lColor.rgba() ] = [ lRegionsTableDict[lKeys[i]][1], lCount ]
                        CDConstants.printOut( str(self.regionUseDict[ lColor.rgba() ][0])+" "+str(self.regionUseDict[ lColor.rgba() ][1]) , CDConstants.DebugTODO )

                # signal upstream about the updated usage of this region color:
                self.parentWidget.parentWindow.theTableOfTypes.updateRegionUseOfTableElements(lColor, lCount)

        # if there is at least one color in the regionUseDict table, show the table:
#         if self.regionUseDict :
#             # we can test the dict this way because, according to the Python manual,
#             #   "any empty sequence, for example, '', (), []" is considered false.
#             self.parentWidget.signalVisibilityPIFRegionTable.emit("Show")
        CDConstants.printOut("subtractFromRegionColorsInUse - regionUseDict = "+str(self.regionUseDict) , CDConstants.DebugTODO )
        CDConstants.printOut("==================================== subtractFromRegionColorsInUse", CDConstants.DebugTODO )




    # ------------------------------------------------------------
    # 2011 - Mitja: obtain the dict for all the used region colors
    # ------------------------------------------------------------
    def getRegionColorsInUse(self):
        return self.regionUseDict




    # ------------------------------------------------------------
    # 2011 - Mitja: after obtaining the item's old color (AKA brush)
    #        TODO IS THIS comment meaningful?
    #        TODO: Fix for automatic cell <-> region for TenByTenBox:
    # ------------------------------------------------------------
    def setItemRegionOrCell(self, pRegionOrCell):
        # copy from setItemColor to set the item regionvscell value
        self.myItemRegionOrCell = pRegionOrCell
        if self.isItemChange(DiagramItem):
            item = self.selectedItems()[0]
            item.setRegionOrCell(self.myItemRegionOrCell)


    # ------------------------------------------------------------
    def setItemColor(self, color):
        self.myItemColor = color
        if self.isItemChange(DiagramItem):
            item = self.selectedItems()[0]

            # 2011 - Mitja: after obtaining the item's old color (AKA brush)
            #  also update the scene's regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            lSelectedItemColor = item.brush()
            self.subtractFromRegionColorsInUse(lSelectedItemColor)

            item.setBrush(self.myItemColor)

            # 2011 - Mitja: after setting the lItem's color (AKA brush)
            #  also update the regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            self.addToRegionColorsInUse(self.myItemColor)





    # ------------------------------------------------------------
    def setSequenceColor(self, pColor):

        if isinstance(self.theImageSequence, CDImageSequence) == True:

            # 2011 - Mitja: after obtaining the sequence's old color (AKA brush)
            #  also update the scene's regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            lOldColor = self.theImageSequence.getSequenceCurrentColor()

            self.subtractFromRegionColorsInUse(lOldColor)

            self.theImageSequence.setSequenceCurrentColor(pColor)

            # 2011 - Mitja: after setting the lItem's color (AKA brush)
            #  also update the regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            self.addToRegionColorsInUse(pColor)



    # ------------------------------------------------------------
    def setLineColor(self, color):
        self.myLineColor = color
        if self.isItemChange(Arrow):
            item = self.selectedItems()[0]
            item.setColor(self.myLineColor)
            self.update()

    # ------------------------------------------------------------
    def setTextColor(self, color):
        self.myTextColor = color
        if self.isItemChange(DiagramTextItem):
            item = self.selectedItems()[0]
            item.setDefaultTextColor(self.myTextColor)

    # ------------------------------------------------------------
    def setFont(self, font):
        self.myFont = font
        if self.isItemChange(DiagramTextItem):
            item = self.selectedItems()[0]
            item.setFont(self.myFont)

    # ------------------------------------------------------------
    def setMode(self, mode):
        # 2010 - Mitja: reset resizing unless we are in resizing mode
        #    and remain in resizing mode:
        if (self.mySceneMode == CDConstants.SceneModeResizeItem) and \
            (mode == CDConstants.SceneModeResizeItem):
            pass
        else:
            self.stopOutlineResizing()
        self.mySceneMode = mode
       
        # 2011 - Mitja: adapt window title to current mode:
        if self.parentWidget.parentWindow != None:
            if self.mySceneMode == CDConstants.SceneModeInsertItem:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Insert Item")
            elif self.mySceneMode == CDConstants.SceneModeInsertLine:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Insert Line")
            elif self.mySceneMode == CDConstants.SceneModeInsertText:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Insert Text")
            elif self.mySceneMode == CDConstants.SceneModeMoveItem:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Move Item")
            elif self.mySceneMode == CDConstants.SceneModeInsertPixmap:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Insert Pixmap")
            elif self.mySceneMode == CDConstants.SceneModeResizeItem:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Resize Item")
            elif self.mySceneMode == CDConstants.SceneModeImageLayer:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Image Layer")
            elif self.mySceneMode == CDConstants.SceneModeImageSequence:
                self.parentWidget.parentWindow.setWindowTitle("Cell Scene Editor - Image Sequence")
           
        CDConstants.printOut("DiagramScene: setMode (mode=="+str(mode)+"). ", CDConstants.DebugVerbose )

    # end of  def setMode(self, mode).
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    def getMode(self):
        return self.mySceneMode

    # ------------------------------------------------------------
    def setItemType(self, type):
        self.myItemType = type

    # ------------------------------------------------------------
    def handlerForLostFocus(self, item):
        cursor = item.textCursor()
        cursor.clearSelection()
        item.setTextCursor(cursor)

        if item.toPlainText():
            self.removeItem(item)
            item.deleteLater()


    # ------------------------------------------------------------
    # 2010 - Mitja - we add to mousePressEvent()
    #   to handle outline-resizing of items and right-button resizing of items
    # ------------------------------------------------------------
    def mousePressEvent(self, pMouseEvent, pThePIFItem=None):
        # 2010 - Mitja: add code for handling insertion of Path items,
        #    separately from original drawing code:   

        # 2011 - Mitja: if in SceneModeImageLayer, then skip all QGraphicsScene
        #   event processing, and pass the event to theCDImageLayer's own
        #   mousePressEvent handler... unless it's a Path item's event!
        if (self.mySceneMode == CDConstants.SceneModeImageLayer) and (pThePIFItem==None):
            if isinstance(self.theImageLayer, CDImageLayer) == True:
                self.theImageLayer.mousePressEvent(pMouseEvent)

        elif (pMouseEvent.button() == QtCore.Qt.LeftButton):

            if self.mySceneMode == CDConstants.SceneModeInsertItem:
                self.stopOutlineResizing()
                # CDConstants.printOut( " "+str( "mousePressEvent:", pMouseEvent, pThePIFItem, self.myItemType )+" ", CDConstants.DebugTODO )
                # 2010 - Mitja: add code for handling insertion of Path items:
                if (self.myItemType == DiagramItem.PathConst):
                    if (pThePIFItem == None) :
                        # 2010 - Mitja: if there is no Path yet, make it a simple boring one:
                        # CDConstants.printOut( " "+str( "mousePressEvent, nopath" )+" ", CDConstants.DebugTODO )
                        miBoringPath = QtGui.QPainterPath()
                        miBoringPath.addEllipse(-100.0, -50.0, 200.0, 100.0)
                        lPath = miBoringPath
                    else :
                        # CDConstants.printOut( " "+str( "mousePressEvent, with PATH" )+" ", CDConstants.DebugTODO )
                        lPath = pThePIFItem.path()
                    lItem = DiagramItem(lPath, self.myItemMenu)
                else:                   
                    lItem = DiagramItem(self.myItemType, self.myItemMenu)
                lItem.setBrush(self.myItemColor)

                # 2011 - Mitja: the new item is a region of cells or a single cell:
                lItem.setRegionOrCell(self.myItemRegionOrCell)

                # 2011 - Mitja: after setting the lItem's color (AKA brush)
                #  also update the regionUseDict since it contains the list of all
                #  region colors in use by our scene:
                self.addToRegionColorsInUse(self.myItemColor)

                # 2011 - Mitja: the "addItem()" call is what adds lItem to the scene,
                #   since without the following call the lItem would never appear:
                self.addItem(lItem)

                lItem.setPos(pMouseEvent.scenePos())
                self.signalThatItemInserted.emit(lItem)
                lTmpBoundingRect = lItem.sceneBoundingRect()
                self.signalThatItemResized.emit(lTmpBoundingRect)

            elif self.mySceneMode == CDConstants.SceneModeInsertLine:
                self.stopOutlineResizing()
                self.line = QtGui.QGraphicsLineItem(QtCore.QLineF(pMouseEvent.scenePos(),
                                            pMouseEvent.scenePos()))
                self.line.setPen(QtGui.QPen(self.myLineColor, 2))
                self.addItem(self.line)

            elif self.mySceneMode == CDConstants.SceneModeInsertText:
                self.stopOutlineResizing()
                textItem = DiagramTextItem()
                textItem.setFont(self.myFont)
                textItem.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
                textItem.setZValue(1000.0)
                textItem.signalLostFocus.connect(self.handlerForLostFocus)
                textItem.signalSelectedChange.connect(self.handlerForItemSelected)
                self.addItem(textItem)
                textItem.setDefaultTextColor(self.myTextColor)
                textItem.setPos(pMouseEvent.scenePos())
                self.signalThatTextInserted.emit(textItem)
            # 2010 - Mitja: add code for handling insertion of pixmap items:
            elif self.mySceneMode == CDConstants.SceneModeInsertPixmap:
                self.stopOutlineResizing()
                if (pThePIFItem == None) :
                    # 2010 - Mitja: if there is no pixmap yet, make it a simple boring one:
                    lBoringPixMap = QtGui.QPixmap(128, 128)
                    lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.darkGray) )
                    lPixmap = lBoringPixMap
                else :
                    lPixmap = pThePIFItem.pixmap()
                lItem = DiagramPixmapItem(lPixmap, self.myItemMenu)
                self.addItem(lItem)
            # 2010 - Mitja: add code for handling outline resizing of items:
            elif self.mySceneMode == CDConstants.SceneModeResizeItem:
                # check if there's already a selected resizing item:
                if self.myOutlineResizingItem != None:
                    # if the second outline is already ON, act upon vertex proximity:
                    self.myOutlineResizingVertex = self.isCloseToOutlineVertex( \
                        self.myOutlineResizingItem, pMouseEvent.scenePos() )
                    # check if the mouse click happens near a vertex of the resizing outline:
                    if self.myOutlineResizingVertex != "None":
                        # get ready to resize according to the bounding box's vertex:
                        self.myOutlineResizingCurrentX = pMouseEvent.scenePos().x()
                        self.myOutlineResizingCurrentY = pMouseEvent.scenePos().y()
                        # update the scene to affect redrawing of the selected item's outline
                        self.update()
                        # passing the mouse event upwards would cause the item to be MOVED by the mouse:
                        # super(DiagramScene, self).mousePressEvent(pMouseEvent)
                        # so we have to return from this function BEFORE it reaches the super() part below:
                        return
                # 2010 - Mitja: implement selection of a second outline to resize items:
                lItemAt = self.itemAt(pMouseEvent.scenePos())
                # CDConstants.printOut( " "+str( "lItemAt =", lItemAt )+" ", CDConstants.DebugTODO )
                # if we haven't clicked on a QGraphicsItem, there isn't anything to do here:
                if isinstance( lItemAt, QtGui.QGraphicsItem ) != True:
                    self.stopOutlineResizing()
                else:
                    # if the second outline is ON and the mouse click is not on the
                    #    currently selected item, then deselect the second outline
                    #    but keep processing:
                    if lItemAt != self.myOutlineResizingItem:
                        self.stopOutlineResizing()
                        # we've clicked on a different QGraphicsItem than the one resizing,
                        # so select the newly clicked item for resizing:
                        self.myOutlineResizingItem = lItemAt
                # update the scene, i.e. have the view repaint it:
                # self.update()

            # 2010 - Mitja: pass the mouse event upwards, so that it may be moved:
            super(DiagramScene, self).mousePressEvent(pMouseEvent)

        # 2010 - Mitja: handle the insertion of Path items:
        #    we execute this part only IF the mousePressEvent is generated by our code
        #    (i.e. there is no mouse button pressed: we just called this function directly!)
        elif (pMouseEvent.button() == QtCore.Qt.NoButton):

            lPath = pThePIFItem.path()
            lItem = DiagramItem(lPath, self.myItemMenu)

            # check if pThePIFItem's color has the "magic" RGBA value=0,0,0,0
            #    which means it's from a Freehand or Polygon drawing, and it has
            #    to be assigned the currently selected scene color:
            lRGBAValue = pThePIFItem.brush().color().rgba()
            # CDConstants.printOut( " "+str( "pThePIFItem.brush().color().rgba() =", lRGBAValue )+" ", CDConstants.DebugTODO )
            if (lRGBAValue == 0):
                # we use this code if we want to assign a new color to the arriving pThePIFItem:
                lItem.setBrush(self.myItemColor)
            else:
                lItem.setBrush(pThePIFItem.brush())


            # 2011 - Mitja: the new item is a region of cells or a single cell:
            lRGBAValue = pThePIFItem.pen().color().rgba()
            if ( lRGBAValue == QtGui.QColor(QtCore.Qt.darkMagenta).rgba() ) or \
                ( lRGBAValue == QtGui.QColor(255, 153, 0).rgba() ):
                # new item's pen color corresponding to either region or cell:
                pass
            else:
                # new item's pen color invalid, set it to be the current selection:
                lItem.setRegionOrCell(self.myItemRegionOrCell)

            self.addItem(lItem)

            # 2011 - Mitja: after setting the lItem's color (AKA brush)
            #  also update the regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            self.addToRegionColorsInUse(lItem.brush())

            # scenePos() doesn't seem to exist for pMouseEvent, so this won't work:
            #   lItem.setPos(pMouseEvent.scenePos())
            # but since the Path item we get from picking the image is not 0-centered,
            # its coordinates are already scene coordinates (not good for scaling!)
            # so at the moment we don't need scenePos()
            # TODO: fix image picking so that objects are 0-centered + have an offset

            # first unselect all selected items in the scene:
            for anItem in self.selectedItems():
                anItem.setSelected(False)
            # then select the currently created Path item:
            lItem.setSelected(True)

            # do not emit any signal when creating items from the image layer:
            # self.signalThatItemInserted.emit(lItem)

            #             except:
            #                 # we got exception to setBrush(), therefore pThePIFItem is NOT a QGraphicsPathItem!
            #                 # TODO TODO TODO: MAYBE?
            #                 pass
            #                self.addItem(pThePIFItem)
            #                # pThePIFItem.setPos(pMouseEvent.scenePos())
            #                self.signalThatItemInserted.emit(pThePIFItem)
            # 2010 - Mitja: pass the mouse event upwards:
            # TODO TODO TODO: MAYBE?
            # super(DiagramScene, self).mousePressEvent(pMouseEvent)

        # 2010 - Mitja: implement resizing of items in real-time by right-clicking on them:
        elif (pMouseEvent.button() == QtCore.Qt.RightButton):
            lItemAt = self.itemAt(pMouseEvent.scenePos())
            if isinstance( lItemAt, QtGui.QGraphicsItem ):
                lColor = lItemAt.brush().color().rgba()
                self.myRightButtonResizingItem = lItemAt
                self.myRightButtonResizingClickX = pMouseEvent.scenePos().x()
                self.myRightButtonResizingClickY = pMouseEvent.scenePos().y()
                # CDConstants.printOut( " "+str( "mousePressEvent mousePressEvent mousePressEvent =", pMouseEvent )+" ", CDConstants.DebugTODO )
                # CDConstants.printOut( " "+str( "myRightButtonResizingItem myRightButtonResizingItem myRightButtonResizingItem =", self.myRightButtonResizingItem )+" ", CDConstants.DebugTODO )
                # CDConstants.printOut( " "+str( "lColor lColor lColor =", lColor )+" ", CDConstants.DebugTODO )
                # CDConstants.printOut( " "+str( "x, y =", self.myRightButtonResizingClickX, self.myRightButtonResizingClickY )+" ", CDConstants.DebugTODO )
            else:
                self.myRightButtonResizingItem = None
                self.myRightButtonResizingClickX = -1.0
                self.myRightButtonResizingClickY = -1.0
            # passing the mouse event upwards would cause the item to be MOVED by the mouse:
            # super(DiagramScene, self).mousePressEvent(pMouseEvent)
        else:
            return


    # ------------------------------------------------------------
    # 2010 - Mitja - we add more code to mouseMoveEvent()
    #   to handle outline-resizing of items and right-button resizing of items,
    #   and (2011) SceneModeImageLayer support as well...
    # ------------------------------------------------------------
    def mouseMoveEvent(self, pMouseEvent):

        # 2011 - Mitja: if in SceneModeImageLayer, then skip all QGraphicsScene
        #   event processing, and pass the event to theCDImageLayer's own
        #   mousePressEvent handler:
        if (self.mySceneMode == CDConstants.SceneModeImageLayer):
            if isinstance(self.theImageLayer, CDImageLayer) == True:
                self.theImageLayer.mouseMoveEvent(pMouseEvent)

        # 2010 - Mitja: implement right-clicking on items,
        #   to resize them in real-time:
        elif self.myRightButtonResizingItem != None:
            if (self.myRightButtonResizingClickX >= 0.0):
                prevX = self.myRightButtonResizingClickX
                prevY = self.myRightButtonResizingClickY
                self.myRightButtonResizingClickX = pMouseEvent.scenePos().x()
                self.myRightButtonResizingClickY = pMouseEvent.scenePos().y()
            else:
                prevX = pMouseEvent.scenePos().x()
                prevY = pMouseEvent.scenePos().y()
                self.myRightButtonResizingClickX = pMouseEvent.scenePos().x()
                self.myRightButtonResizingClickY = pMouseEvent.scenePos().y()

            sx = (self.myRightButtonResizingClickX - prevX) / 300.0
            sy = (self.myRightButtonResizingClickY - prevY) / 300.0

            self.myRightButtonResizingItem.myScaleX = self.myRightButtonResizingItem.myScaleX + sx
            self.myRightButtonResizingItem.myScaleY = self.myRightButtonResizingItem.myScaleY - sy

            # calling setScale() directly on a QGraphicsItem WORKS but it can only scale
            #   proportionally, and we want to scale x/y independendently so we don't use it:
            # self.myRightButtonResizingItem.setScale( self.myRightButtonResizingItem.myScaleY )

            # using QTransform, we first create a transformation, then we apply it to the item:
            lTransform = QtGui.QTransform()
            lTransform.scale(  self.myRightButtonResizingItem.myScaleX,  self.myRightButtonResizingItem.myScaleY )
            self.myRightButtonResizingItem.setTransform( lTransform )

            # even though setTransform and setTransformations sound similar, they are two
            #   DIFFERENT transformation mechanisms provided by Qt... confusing or what?!?
            #   using setTransformations does not work this way, so we can't use it here:
            # self.myRightButtonResizingItem.setTransformations( lTransform )

            # even though QTransform and QGraphicsTransform sound similar, they are two
            #   DIFFERENT transformation mechanisms provided by Qt... confusing or what?!?
            #   using QGraphicsTransform does not work this way, so we can't use it here:
            # lQGraphicsScele = QtGui.QGraphicsScale()
            # lQGraphicsScale.setOrigin(QVector3D(QPointF(60,30)))
            # lQGraphicsScele.setXScale(  self.myRightButtonResizingItem.myScaleX  )
            # lQGraphicsScele.setYScale(  self.myRightButtonResizingItem.myScaleY  )
            # self.myRightButtonResizingItem.setTransformations( [ lQGraphicsScele ] )

            # calling scale() directly on a QGraphicsItem existed in Qt 3.3. but it
            #   has been deprecated in Qt 4.x, so we can't use it:
            # self.myRightButtonResizingItem.scale( self.myRightButtonResizingItem.myScaleX,  self.myRightButtonResizingItem.myScaleY )

            # CDConstants.printOut( " "+str( "pMouseEvent.scenePos().x(), self.myRightButtonResizingClickX, sx, self.myRightButtonResizingItem.myScaleX =", pMouseEvent.scenePos().x(), self.myRightButtonResizingClickX, sx, self.myRightButtonResizingItem.myScaleX )+" ", CDConstants.DebugTODO )
            # CDConstants.printOut( " "+str( "pMouseEvent.scenePos().y(), self.myRightButtonResizingClickY, sy, self.myRightButtonResizingItem.myScaleY =", pMouseEvent.scenePos().y(), self.myRightButtonResizingClickY, sy, self.myRightButtonResizingItem.myScaleY )+" ", CDConstants.DebugTODO )
            # CDConstants.printOut( " "+str( "self.myRightButtonResizingItem.transformations() =", self.myRightButtonResizingItem.transformations() )+" ", CDConstants.DebugTODO )

        # 2010 - Mitja: implement a second outline for items, to resize them in real-time:
        elif (self.mySceneMode == CDConstants.SceneModeResizeItem) and (self.myOutlineResizingItem != None):
            # use the mouse relative movement (pointer displacement between events) to
            #   compute the amount of scaling
            if (self.myOutlineResizingCurrentX >= 0.0):
                prevX = self.myOutlineResizingCurrentX
                prevY = self.myOutlineResizingCurrentY
                self.myOutlineResizingCurrentX = pMouseEvent.scenePos().x()
                self.myOutlineResizingCurrentY = pMouseEvent.scenePos().y()
            else:
                prevX = pMouseEvent.scenePos().x()
                prevY = pMouseEvent.scenePos().y()
                self.myOutlineResizingCurrentX = pMouseEvent.scenePos().x()
                self.myOutlineResizingCurrentY = pMouseEvent.scenePos().y()
           
            # we compute scaling using the bounding rectangle and the mouse displacement:
            dXmouse = (self.myOutlineResizingCurrentX - prevX)
            dYmouse = (self.myOutlineResizingCurrentY - prevY)
            lBoundingRect = self.myOutlineResizingItem.sceneBoundingRect()
            bWidthX = lBoundingRect.bottomRight().x() - lBoundingRect.topLeft().x()
            bHeightY = lBoundingRect.topRight().y() - lBoundingRect.bottomLeft().y()
            # CDConstants.printOut( " "+str( " ------------------ lBoundingRect befor scaling =", lBoundingRect )+" ", CDConstants.DebugTODO )

            # A - compute scaling factors along X and Y according to mouse motion:
            if self.myOutlineResizingVertex == "topRight":
                sx = (bWidthX + dXmouse) / bWidthX
                sy = (bHeightY + dYmouse) / bHeightY
            elif self.myOutlineResizingVertex == "topLeft":
                sx = (bWidthX - dXmouse) / bWidthX
                sy = (bHeightY + dYmouse) / bHeightY
            elif self.myOutlineResizingVertex == "bottomRight":
                sx = (bWidthX + dXmouse) / bWidthX
                sy = (bHeightY - dYmouse) / bHeightY
            elif self.myOutlineResizingVertex == "bottomLeft":
                sx = (bWidthX - dXmouse) / bWidthX
                sy = (bHeightY - dYmouse) / bHeightY
            else:
                # if we are not resizing the item by dragging one of its corners,
                # then pass the even upstream and return right away!
                super(DiagramScene, self).mouseMoveEvent(pMouseEvent)
                return

            self.myOutlineResizingItem.myScaleX = self.myOutlineResizingItem.myScaleX * sx
            self.myOutlineResizingItem.myScaleY = self.myOutlineResizingItem.myScaleY * sy
            # limit diminishing resizing values so that they won't make the item disappear:
            if (self.myOutlineResizingItem.myScaleX < 0.005) :
                self.myOutlineResizingItem.myScaleX = 0.005
            if (self.myOutlineResizingItem.myScaleY < 0.005) :
                self.myOutlineResizingItem.myScaleY = 0.005

            # B - apply the scaling transformation to the item:
            # the use of QTransform is as such: first set the transformation, then
            #   apply the transformation to the QGraphicsItem:
            lTransform = QtGui.QTransform()
            lTransform.scale(  self.myOutlineResizingItem.myScaleX,  self.myOutlineResizingItem.myScaleY )
            self.myOutlineResizingItem.setTransform( lTransform )
            # self.update()

            # note: the sceneBoundingRect is NOT updated even after the scaling operation!
            #   apparently, Qt will update the QGraphicsItem's sceneBoundingRect at redraw only.
            #   so the following operations would get the sceneBoundingRect as *before* scaling:
            # lBoundingRect = self.myOutlineResizingItem.sceneBoundingRect()

            # C - now fix the item's position according to the scaling transformation,
            #   since we want the fixed-point of the transformation to be the vertex corner
            #   at the opposite side of the resizing vertex corner (and not the object's center)
            lItemScenePoint = self.myOutlineResizingItem.scenePos()
            if self.myOutlineResizingVertex == "topRight":
                lItemNewPosX = lItemScenePoint.x() - ( (bWidthX - sx * bWidthX) * 0.5 )
                lItemNewPosY = lItemScenePoint.y() - ( (bHeightY - sy * bHeightY) * 0.5 )
            elif self.myOutlineResizingVertex == "topLeft":
                lItemNewPosX = lItemScenePoint.x() + ( (bWidthX - sx * bWidthX) * 0.5 )
                lItemNewPosY = lItemScenePoint.y() - ( (bHeightY - sy * bHeightY) * 0.5 )
            elif self.myOutlineResizingVertex == "bottomRight":
                lItemNewPosX = lItemScenePoint.x() - ( (bWidthX - sx * bWidthX) * 0.5 )
                lItemNewPosY = lItemScenePoint.y() + ( (bHeightY - sy * bHeightY) * 0.5 )
            elif self.myOutlineResizingVertex == "bottomLeft":
                lItemNewPosX = lItemScenePoint.x() + ( (bWidthX - sx * bWidthX) * 0.5 )
                lItemNewPosY = lItemScenePoint.y() + ( (bHeightY - sy * bHeightY) * 0.5 )
            else:
                return

            # one has to be careful with setPos() because Qt does NOT provide a function
            #    to set a QGraphicsItem's position in scene coordinates, only in *parent*
            #    coordinates. So the following setPos() call would FAIL if we had hierarchies
            #    of items. As from the Qt online documentation: "setPos() sets the position of
            #    the item to pos, which is in parent coordinates. For items with no parent,
            #    pos is in scene coordinates."
            lItemScenePoint.setX ( lItemNewPosX )
            lItemScenePoint.setY ( lItemNewPosY )
            self.myOutlineResizingItem.setPos(lItemScenePoint)

            # CDConstants.printOut( " "+str( "pMouseEvent.scenePos().x(), self.myOutlineResizingCurrentX, sx, self.myOutlineResizingItem.myScaleX =", pMouseEvent.scenePos().x(), self.myOutlineResizingCurrentX, sx, self.myOutlineResizingItem.myScaleX )+" ", CDConstants.DebugTODO )
            # CDConstants.printOut( " "+str( "pMouseEvent.scenePos().y(), self.myOutlineResizingCurrentY, sy, self.myOutlineResizingItem.myScaleY =", pMouseEvent.scenePos().y(), self.myOutlineResizingCurrentY, sy, self.myOutlineResizingItem.myScaleY )+" ", CDConstants.DebugTODO )
            # CDConstants.printOut( " "+str( "self.myOutlineResizingItem.transformations() =", self.myOutlineResizingItem.transformations() )+" ", CDConstants.DebugTODO )

        elif self.mySceneMode == CDConstants.SceneModeInsertLine and self.line:
            newLine = QtCore.QLineF(self.line.line().p1(), pMouseEvent.scenePos())
            self.line.setLine(newLine)

        elif self.mySceneMode == CDConstants.SceneModeMoveItem:
            super(DiagramScene, self).mouseMoveEvent(pMouseEvent)
            # signal to update the moved item's x/y/w/h information display:
            lAllSelectedItems = self.selectedItems()
            # make sure that there is at least one selected item:
            if lAllSelectedItems:
                # get the first selected item:
                lFirstSelectedItem = lAllSelectedItems[0]
                lBoundingRect = lFirstSelectedItem.sceneBoundingRect()
                self.signalThatItemResized.emit(lBoundingRect)




    # ------------------------------------------------------------
    # 2010 - Mitja - mouseReleaseEvent() modified to handle both
    #   outline-resizing of items and right-button resizing of items,
    #   and (2011) SceneModeImageLayer support as well...
    # ------------------------------------------------------------
    def mouseReleaseEvent(self, pMouseEvent):

        # 2010 - Mitja: respond to all release events to stop outline resizing of items:
        self.myOutlineResizingVertex = "None"
        # update the scene to affect redrawing of the selected item's outline
        self.update()

        # 2011 - Mitja: if in SceneModeImageLayer, then skip all QGraphicsScene
        #   event processing, and pass the event to theCDImageLayer's own
        #   mousePressEvent handler:
        if (self.mySceneMode == CDConstants.SceneModeImageLayer):
            if isinstance(self.theImageLayer, CDImageLayer) == True:
                self.theImageLayer.mouseReleaseEvent(pMouseEvent)

        # 2010 - Mitja: implement right-clicking on items, to resize them in real-time:
        elif self.myRightButtonResizingItem != None:
            self.myRightButtonResizingItem = None
            self.myRightButtonResizingClickX = -1.0
            self.myRightButtonResizingClickY = -1.0
            # CDConstants.printOut( " "+str( "mouseReleaseEvent ELSE mouseReleaseEvent ELSE mouseReleaseEvent =", pMouseEvent )+" ", CDConstants.DebugTODO )
            # CDConstants.printOut( " "+str( "myRightButtonResizingItem ELSE myRightButtonResizingItem ELSE myRightButtonResizingItem =", self.myRightButtonResizingItem )+" ", CDConstants.DebugTODO )
            # update the scene to remove the foreground outline:
            self.update()


        elif self.line and self.mySceneMode == CDConstants.SceneModeInsertLine:
            startItems = self.items(self.line.line().p1())
            if len(startItems) and startItems[0] == self.line:
                startItems.pop(0)
            endItems = self.items(self.line.line().p2())
            if len(endItems) and endItems[0] == self.line:
                endItems.pop(0)

            self.removeItem(self.line)
            self.line = None

            if len(startItems) and len(endItems) and \
                    isinstance(startItems[0], DiagramItem) and \
                    isinstance(endItems[0], DiagramItem) and \
                    startItems[0] != endItems[0]:
                startItem = startItems[0]
                endItem = endItems[0]
                arrow = Arrow(startItem, endItem)
                arrow.setColor(self.myLineColor)
                startItem.addArrow(arrow)
                endItem.addArrow(arrow)
                arrow.setZValue(-1000.0)
                self.addItem(arrow)
                arrow.updatePosition()

        self.line = None

        #
        # 2010 - Mitja: update the scene's rectangle manually, so that it expands with all the scene elements:
        #
        #         lNewSceneRect = self.itemsBoundingRect()
        #         if lNewSceneRect.isNull():
        #             # if there are no elements, make the scene at least as large as the background image:
        #             self.setSceneRect(self.theImageFromFile.rect())
        #         else:
        #             # make the scene large enough to include both the elements as well as the background image dimensions:
        #             self.setSceneRect(lNewSceneRect.united(self.theImageFromFile.rect()))
        #
        # does not seem to work properly.... the scene resizes in a weird way!?!
        #

        super(DiagramScene, self).mouseReleaseEvent(pMouseEvent)




    # ------------------------------------------------------------
    # 2012 - Mitja - keyReleaseEvent() added to handle pressing the <esc> key
    #   and passing it to theImageLayer when in SceneModeImageLayer mode.
    # ------------------------------------------------------------
    def keyReleaseEvent(self, pMouseEvent):

        # 2012 - Mitja: if in SceneModeImageLayer, then skip all QGraphicsScene
        #   event processing, and pass the event to theCDImageLayer's own
        #   keyReleaseEvent handler:
        if (self.mySceneMode == CDConstants.SceneModeImageLayer):
            if isinstance(self.theImageLayer, CDImageLayer) == True:
                self.theImageLayer.keyReleaseEvent(pMouseEvent)

        super(DiagramScene, self).keyReleaseEvent(pMouseEvent)




    # ------------------------------------------------------------
    def isItemChange(self, type):
        for item in self.selectedItems():
            if isinstance(item, type):
                return True
        return False




    # ------------------------------------------------------------
    # 2010 - Mitja: reset variables for resizing an item by using an outline:
    def stopOutlineResizing(self):
        self.myOutlineResizingItem = None
        #   myOutlineResizingVertex can be: "None", "bottomLeft", "bottomRight", "topRight", "topLeft":
        self.myOutlineResizingVertex = "None"
        self.myOutlineResizingCurrentX = -1.0
        self.myOutlineResizingCurrentY = -1.0
        self.update()
       
       

    # ------------------------------------------------------------
    # check if any of the vertices in the item's bounding rectangle is close to the mouse position,
    #   and if so return its value - the input parameters are the item and the mouse's scene pos:
    # ------------------------------------------------------------
    def isCloseToOutlineVertex(self, pMyOutlineResizingItem, pMousePosInScene):
        lBoundingRect = self.myOutlineResizingItem.sceneBoundingRect()
        if self.vvIsCloseDistance(lBoundingRect.bottomLeft(), pMousePosInScene):
            # CDConstants.printOut( " "+str( "bottomLeft" )+" ", CDConstants.DebugTODO )
            return "bottomLeft"
        elif self.vvIsCloseDistance(lBoundingRect.bottomRight(), pMousePosInScene):
            # CDConstants.printOut( " "+str( "bottomRight" )+" ", CDConstants.DebugTODO )
            return "bottomRight"
        elif self.vvIsCloseDistance(lBoundingRect.topRight(), pMousePosInScene):
            # CDConstants.printOut( " "+str( "topRight" )+" ", CDConstants.DebugTODO )
            return "topRight"
        elif self.vvIsCloseDistance(lBoundingRect.topLeft(), pMousePosInScene):
            # CDConstants.printOut( " "+str( "topLeft" )+" ", CDConstants.DebugTODO )
            return "topLeft"
        else:
            # CDConstants.printOut( " "+str( "None" )+" ", CDConstants.DebugTODO )
            return "None"

    # ------------------------------------------------------------
    # provide vertex-to-vertex distance calculation:
    # ------------------------------------------------------------
    def vvIsCloseDistance(self, pV1, pV2):
        x1 = pV1.x()
        y1 = pV1.y()
        x2 = pV2.x()
        y2 = pV2.y()
        # CDConstants.printOut( " "+str( "x1, y1,  x2, y2,  dist = ",  x1, y1,  x2, y2, math.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) ) )+" ", CDConstants.DebugTODO )
        if math.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) ) < 6.0:
            return True
        else:
            return False





# ------------------------------------------------------------
# 2010 - Mitja - move from a QMainWindow class to a QWidget panel-type class
# ------------------------------------------------------------
# note: this class emits a signal: "signalVisibilityPIFRegionTable"
# ------------------------------------------------------------
# class DiagramSceneMainWindow(QtGui.QMainWindow):
class CDDiagramSceneMainWidget(QtGui.QWidget):

    mysignal = QtCore.pyqtSignal(str)

    signalVisibilityPIFRegionTable = QtCore.pyqtSignal(str)

    InsertTextButton = 10

    # 2010 - Mitja - since this class is not a QMainWindow class anymore,
    #   we need to be able to access our "parent" class (a QMainWindow!)
    #   to place menus and toolbars onto it.
    def __init__(self, pParentWindow = None):
        super(CDDiagramSceneMainWidget, self).__init__(pParentWindow)

        self.parentWindow = pParentWindow


        if self.parentWindow == None:
            #  -> this is now called diagramSceneCreateActions() in ControllerMainWindow:
            # self.createActions()
            pass
        else:
            self.createSceneEditActions()

            # 2010 - Mitja: is there any real need for menus,
            #    when all actions are reachable from toolbars?
            self.createMenus()
           

        # 2011 - Mitja: create the QGraphicsScene as main PIFF editing place:
        self.scene = DiagramScene(self.editMenu, self)
        # add a file associated to the scene:
        self.curFile = ''
        # 2010 - Mitja:
        # self.scene.setSceneRect(QtCore.QRectF(0, 0, 5000, 5000))
        self.scene.setSceneRect(QtCore.QRectF(0, 0, 240, 180))


        # -------------------------------------------------------------------

        # 2011 - Mitja: create a Control Panel with buttons & sliders:
        #
        # panel for PIFF parameters, with the CDDiagramSceneMainWidget as its parent
        #    (that's why we pass "self" as parameter to the dialog panel) :
        self.windowPIFControlPanel = CDControlPanel(self)

        # 2011 - Mitja: we now place all toolbox items inside the main Control Panel.

        # -----------------------------------

        # 2011 - Mitja: to control the "layer selection" for Cell Scene mode,
        #   we add a set of radio-buttons to the Control Panel:

        self.theControlsForLayerSelection = CDControlLayerSelection()

        # explicitly connect the "signalLayersSelectionModeHasChanged()"
        #   signal from the theControlsForLayerSelection object,
        #   to our "slot" method responding to radio button changes:
        self.theControlsForLayerSelection.signalLayersSelectionModeHasChanged.connect( \
            self.handleLayersSelectionModeHasChanged )

        # place the layer selection buttons in the control panel:
        self.windowPIFControlPanel.setControlsForLayerSelection( \
            self.theControlsForLayerSelection)

        # -----------------------------------

        # 2011 - Mitja: to control Cell Scene "scale/zoom" factor,
        #   we add a combobox (pop-up menu) to the Control Panel:

        self.theSceneScaleZoomControl = CDControlSceneScaleZoom()

        # explicitly connect the "signalScaleZoomHasChanged()"
        #   signal from the theSceneScaleZoomControl object,
        #   to our "slot" method responding to radio button changes:
        self.theSceneScaleZoomControl.signalScaleZoomHasChanged.connect( \
            self.handleSceneScaleZoomChanged )

        # place the layer selection buttons in the control panel:
        self.windowPIFControlPanel.setControlsForSceneScaleZoom( \
            self.theSceneScaleZoomControl)

        # -----------------------------------

        # 2011 - Mitja: to control the "drawing toggle" for regions vs. cells,
        #   we add radio-buttons to the Control Panel:

        self.theToggleforRegionOrCellDrawing = CDControlRegionOrCell()

        # explicitly connect the "signalSetRegionOrCell()" signal from the
        #   theToggleforRegionOrCellDrawing object, to our "slot" method
        #   so that it will respond to any change in radio button choices:

        answer = self.theToggleforRegionOrCellDrawing.signalSetRegionOrCell.connect( \
            self.handleToggleRegionOrCellDrawingChanged)

        # place the drawing toggle GUI in the control panel:
        self.windowPIFControlPanel.setControlsForDrawingRegionOrCellToggle( \
            self.theToggleforRegionOrCellDrawing)

        # -----------------------------------

        #   Here we provide a logical button group for Region Shapes button widgets
        #   that are going to be placed inside the main Control Panel:
        self.theButtonGroupForRegionShapes = self.createButtonGroupForRegionShapes()
        #   Here we provide a logical button group for Background button widgets
        #   that are going to be placed inside the main Control Panel:
        self.theButtonGroupForBackgrounds = self.createButtonGroupForBackgrounds()

        self.windowPIFControlPanel.setButtonGroupForRegionShapes( \
            self.theButtonGroupForRegionShapes )

        self.windowPIFControlPanel.setButtonGroupForBackgrounds( \
            self.theButtonGroupForBackgrounds )

        # create four icons for the basic four region DiagramItem buttons:
        lItem = DiagramItem(DiagramItem.RectangleConst, self.editMenu)
        lIcon = QtGui.QIcon( lItem.pixmapForIconFromPolygon() )
        self.windowPIFControlPanel.setWidgetIcon(DiagramItem.RectangleConst, lIcon)

        lItem = DiagramItem(DiagramItem.TenByTenBoxConst, self.editMenu)
        # u"\u00D7" # <-- the multiplication sign as unicode
        lIcon = QtGui.QIcon( lItem.pixmapForIconFromPolygon("10" + u"\u00D7" + "10") )
        self.windowPIFControlPanel.setWidgetIcon(DiagramItem.TenByTenBoxConst, lIcon)

        lItem = DiagramItem(DiagramItem.TwoByTwoBoxConst, self.editMenu)
        # u"\u00D7" # <-- the multiplication sign as unicode
        lIcon = QtGui.QIcon( lItem.pixmapForIconFromPolygon("2 " + u"\u00D7" + " 2") )
        self.windowPIFControlPanel.setWidgetIcon(DiagramItem.TwoByTwoBoxConst, lIcon)


        # the "PathConst" is not there since it's generated by CDControlPanel internally:
        # lItem = DiagramItem(DiagramItem.PathConst, self.editMenu)
        # lIcon = QtGui.QIcon( lItem.pixmapForIconFromPolygon() )
        # self.windowPIFControlPanel.setWidgetIcon(DiagramItem.PathConst, lIcon)
        lItem = 0

        # -----------------------------------

        # 2010 - Mitja: add code for new backgrounds:
        lBoringPixMap = QtGui.QPixmap(240, 180)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.white) )
        self.theImageFromFile = QtGui.QImage(lBoringPixMap)
        self.theImageNameFromFile = "BlankBackground"

        # -----------------------------------

        # 2011 - Mitja: to control the "picking mode" for the input image,
        #   we add a set of radio-buttons and a slider to the Control Panel:

        self.theControlsForInputImagePicking = CDControlInputImage()
        # explicitly connect the "inputImagePickingModeChangedSignal()" signal from the
        #   theControlsForInputImagePicking object, to our "slot" method
        #   so that it will respond to any change in radio button choices:
        answer = self.connect(self.theControlsForInputImagePicking, \
                              QtCore.SIGNAL("inputImagePickingModeChangedSignal()"), \
                              self.handleInputImagePickingModeChanged )
        # explicitly connect the "inputImageOpacityChangedSignal()" signal from the
        #   theControlsForInputImagePicking object, to our "slot" method
        #   so that it will respond to any change in slider values:
        answer = self.connect(self.theControlsForInputImagePicking, \
                              QtCore.SIGNAL("inputImageOpacityChangedSignal()"), \
                              self.handleImageOpacityChanged )
        # explicitly connect the "fuzzyPickTresholdChangedSignal()" signal from the
        #   theControlsForInputImagePicking object, to our "slot" method
        #   so that it will respond to any change in slider values:
        answer = self.connect(self.theControlsForInputImagePicking, \
                              QtCore.SIGNAL("fuzzyPickTresholdChangedSignal()"), \
                              self.handleFuzzyPickThresholdChanged )

        # explicitly connect the "signalImageScaleZoomHasChanged()"
        #   signal from the theControlsForInputImagePicking object,
        #   to our "slot" method responding to radio button changes:
        self.theControlsForInputImagePicking.signalImageScaleZoomHasChanged.connect( \
            self.handleImageScaleZoomChanged )


        # 2011 - Mitja: since the self.theControlsForInputImagePicking widget is used
        #   to control QGraphicsScene's theCDImageLayer's behavior, it's initially not enabled:
        self.theControlsForInputImagePicking.setEnabled(False)
       
        self.windowPIFControlPanel.setControlsForInputImagePicking( \
            self.theControlsForInputImagePicking)




        # -----------------------------------

        # 2011 - Mitja: to access the "image sequence",
        #   we add a set of buttons and sliders to the Control Panel:

        self.theControlsForImageSequence = CDControlImageSequence()

        # explicitly connect the "signalSelectedImageInSequenceHasChanged()" signal from the
        #   theControlsForImageSequence object, to our "slot" method
        #   so that it will respond to any change in slider values:
        self.theControlsForImageSequence.signalSelectedImageInSequenceHasChanged.connect( \
            self.handleSelectedImageWithinSequenceChanged )


        # explicitly connect the "signalImageSequenceProcessingModeHasChanged()"
        #   signal from the theControlsForImageSequence object,
        #   to our "slot" method responding to radio button changes:
        self.theControlsForImageSequence.signalImageSequenceProcessingModeHasChanged.connect( \
            self.handleAreaOrEdgeModeHasChanged )


        # explicitly connect the "signalSetCurrentTypeColor()" signal from the
        #   theControlsForImageSequence object, to our "slot" method:

        answer = self.theControlsForImageSequence.signalSetCurrentTypeColor.connect( \
            self.handleTypesColorEvent)

        # explicitly connect the "signalForPIFFTableToggle()" signal from the
        #   theControlsForImageSequence object, to our "slot" method:
        answer = self.theControlsForImageSequence.signalForPIFFTableToggle.connect( \
            self.handlePIFRegionTableButton)


        # 2011 - Mitja: since the self.theControlsForImageSequence widget is used
        #   to control an Image Sequence, it's initially not enabled:
        self.theControlsForImageSequence.setEnabled(False)
       
        self.windowPIFControlPanel.setControlsForImageSequence( \
            self.theControlsForImageSequence)










        # -----------------------------------

        # 2011 - Mitja: to access the "image sequence",
        #   we add a set of buttons and sliders to the Control Panel:

        self.theControlsForClusters = CDControlClusters()
# 
#         # explicitly connect the "signalSelectedImageInSequenceHasChanged()" signal from the
#         #   theControlsForClusters object, to our "slot" method
#         #   so that it will respond to any change in slider values:
#         self.theControlsForClusters.signalSelectedImageInSequenceHasChanged.connect( \
#             self.handleSelectedImageWithinSequenceChanged )
# 
# 
#         # explicitly connect the "signalImageSequenceProcessingModeHasChanged()"
#         #   signal from the theControlsForClusters object,
#         #   to our "slot" method responding to radio button changes:
#         self.theControlsForClusters.signalImageSequenceProcessingModeHasChanged.connect( \
#             self.handleAreaOrEdgeModeHasChanged )
# 
#         # explicitly connect the "signalSetCurrentTypeColor()" signal from the
#         #   theControlsForClusters object, to our "slot" method:
# 
#         answer = self.theControlsForClusters.signalSetCurrentTypeColor.connect( \
#             self.handleTypesColorEvent)
# 
#         # explicitly connect the "signalForPIFFTableToggle()" signal from the
#         #   theControlsForClusters object, to our "slot" method:
#         answer = self.theControlsForClusters.signalForPIFFTableToggle.connect( \
#             self.handlePIFRegionTableButton)
# 
# 
#         # 2011 - Mitja: since the self.theControlsForClusters widget is used
#         #   to control an Image Sequence, it's initially not enabled:
#         self.theControlsForClusters.setEnabled(False)
       
        self.windowPIFControlPanel.setControlsForClusters( \
            self.theControlsForClusters)




        # ------------------------------------------------------------
        # 2010 - Mitja: "Scene Item Edit" controls,
        #   containing basic scene editing actions:
        #
        self.theControlsForSceneItemEdit = PIFControlSceneItemEdit()
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.cutAction)
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.copyAction)
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.pasteAction)
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.deleteAction)
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.toFrontAction)
        self.theControlsForSceneItemEdit.addActionToControlsForSceneItemEdit(self.sendBackAction)
        self.theControlsForSceneItemEdit.populateControlsForSceneItemEdit()

        self.windowPIFControlPanel.setControlsForSceneItemEdit( \
            self.theControlsForSceneItemEdit)



        # ------------------------------------------------------------
        # 2010 - Mitja: controls for setting types of regions and cells:
        #
        self.theControlsForTypes = CDControlTypes()

        # explicitly connect the "signalSetCurrentTypeColor()" signal from the
        #   theControlsForTypes object, to our "slot" method:

        answer = self.theControlsForTypes.signalSetCurrentTypeColor.connect( \
            self.handleTypesColorEvent)

        # explicitly connect the "signalForPIFFTableToggle()" signal from the
        #   theControlsForTypes object, to our "slot" method:
        answer = self.theControlsForTypes.signalForPIFFTableToggle.connect( \
            self.handlePIFRegionTableButton)

        # add the Table of Types window show/hide toggle action to the windowMenu in the main menu bar:
        self.theControlsForTypes.setMenuForTableAction(self.windowMenu)

        self.windowPIFControlPanel.setControlsForTypes( self.theControlsForTypes)
       

        # ------------------------------------------------------------

        # populate the PIFF control panel toolbar with its content, i.e.
        #   now that we passed all the required data, generate the Control Panel GUI:
        self.windowPIFControlPanel.populateControlPanel()

        # and finally show the PIFF control panel:
        self.windowPIFControlPanel.show()

        # ----------------------

        #         self.createToolbars()

        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setMargin(2)
        self.layout().setAlignment(QtCore.Qt.AlignCenter)

        self.view = QtGui.QGraphicsView(self.scene)
        # our QGraphicsView will not accept partial viewport updates:
        self.view.setViewportUpdateMode(QtGui.QGraphicsView.FullViewportUpdate)       
        # when the view is resized, the view leaves the scene's *position* unchanged
        self.view.setResizeAnchor(QtGui.QGraphicsView.NoAnchor)
        # when the whole scene is visible in the view, where is it aligned?
        self.view.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)

        # take care of the RHS <-> LHS mismatch at its visible end,
        #   by flipping the y coordinate in the QGraphicsView's affine transformations:
        self.view.scale(1.0, -1.0)

        self.layout().addWidget(self.view)
       
        # 2011 - Mitja: add a separate image layer object
        #   to paint piff input images on the top of the QGraphicsScene:
        self.theCDImageLayer = CDImageLayer(self)
        self.scene.setImageLayer(self.theCDImageLayer)


        # 2011 - Mitja: add a separate object to handle a sequence of images:
        self.theCDImageSequence = CDImageSequence(self)
        self.scene.setImageSequence(self.theCDImageSequence)


        self.connectSignals()

        self.setWindowTitle("Cell Scene Region Editor")
        self.show()



    # ------------------------------------------------------------------
    # 2011 - Mitja: assign CellDraw preferences object:
    # ------------------------------------------------------------------
    def setPreferencesObject(self, pCDPreferences=None):
        self.cdPreferences = pCDPreferences
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDDiagramSceneMainWidget.cdPreferences is now =" + str(self.cdPreferences), CDConstants.DebugVerbose )


    # ------------------------------------------------------------
    def handlerForButtonGroupBackgroundsClicked(self, pButton):
        theBgButtons = self.theButtonGroupForBackgrounds.buttons()
        for myButton in theBgButtons:
            if myButton != pButton:
                myButton.setChecked(False)
            # CDConstants.printOut( " "+str( "handlerForButtonGroupBackgroundsClicked() myButton =",myButton,\ )+" ", CDConstants.DebugTODO )
            #     " isChecked() =", myButton.isChecked()

        text = pButton.text()

        self.updateSceneBackgroundImage(text)




    # ------------------------------------------------------------
    # 2010 - Mitja - separated updateSceneBackgroundImage() from
    #    handlerForButtonGroupBackgroundsClicked(), so that the following functionality is reachable
    #    also when there is no toolbox click, but a change either in the scene dimensions
    #    or in the background image source itself.
    # ------------------------------------------------------------
    def updateSceneBackgroundImage(self, pText):
    # ------------------------------------------------------------
        # create a pixmap that's as least as wide/tall as the current scene,
        #   and at least as wide as to fit the background image's size, positioned at 0,0,
        #   and at least 2000x2000 in size not to have tiling of the background image:
        lTempLargestQPixmapRect = QtCore.QRect(0, 0, 2000, 2000)
        lBackgroundRect = QtCore.QRectF(       \
            self.scene.sceneRect().united(     \
              QtCore.QRectF(self.theImageFromFile.rect().united(lTempLargestQPixmapRect) )    \
            )     \
        )
        lPixmap = QtGui.QPixmap(lBackgroundRect.width(), lBackgroundRect.height())
        # lPixmap.fill( QtGui.QPalette.color(QtGui.QPalette(), QtGui.QPalette.Base) )
        lPixmap.fill( QtGui.QColor(QtCore.Qt.gray) )
        lPainter = QtGui.QPainter(lPixmap)

        # create another pixmap of the same size as the graphics scene rectangle, and fill it with the chosen background pattern:
        lTmpPixmap  = QtGui.QPixmap( self.scene.width(), self.scene.height() )
        lTmpPixmap.fill( QtGui.QColor(QtCore.Qt.white) )
        lTmpPainter = QtGui.QPainter(lTmpPixmap)

        if pText == "Blue Grid":
            lTmpPainter.fillRect( self.scene.sceneRect(), QtGui.QBrush(QtGui.QPixmap(':/icons/background1.png')) )
            # self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/icons/background1.png')))
        elif pText == "White Grid":
            lTmpPainter.fillRect( self.scene.sceneRect(), QtGui.QBrush(QtGui.QPixmap(':/icons/background2.png')) )
            # self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/icons/background2.png')))
        elif pText == "Gray Grid":
            lTmpPainter.fillRect( self.scene.sceneRect(), QtGui.QBrush(QtGui.QPixmap(':/icons/background3.png')) )
            # self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/icons/background3.png')))
        elif pText == "No Grid":
            lTmpPainter.fillRect( self.scene.sceneRect(), QtGui.QBrush(QtGui.QPixmap(':/icons/background4.png')) )
            # self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QPixmap(':/icons/background4.png')))
        # 2010 - Mitja: add code for new backgrounds:
        else:
            # lPixmap.fill( QtGui.QPalette.color(QtGui.QPalette(), QtGui.QPalette.Base) )
            # lPainter.drawImage(QtCore.QPoint(0,0), self.theImageFromFile)

            # immediately transform the starting/loaded QImage: invert Y values, from RHS to LHS:
            lWidth = self.theImageFromFile.width()
            lHeight = self.theImageFromFile.height()

            # take care of the RHS <-> LHS mismatch at its visible end,
            #   by flipping the y coordinate in the QPainter's affine transformations:       
            lTmpPainter.translate(0.0, float(lHeight))
            lTmpPainter.scale(1.0, -1.0)
   
            # access the QLabel's pixmap to draw it explicitly, using QPainter's scaling:
            lTmpPainter.drawImage(QtCore.QPoint(0,0), self.theImageFromFile)

        lTmpPainter.end()
        lPainter.drawPixmap(QtCore.QPoint(0,0), lTmpPixmap)
        lPainter.end()

        # 2010 - Mitja: update the background image from file:
        # self.theImageFromFile = QtGui.QImage(lPixmap)
        # 2010 - Mitja: if we wanted to update the scene size to fit to the background image, we'd do this:
        # self.scene.setSceneRect(lBackgroundRect)

        self.scene.setBackgroundBrush(QtGui.QBrush(lPixmap))

        self.scene.update()
        self.view.update()
        # CDConstants.printOut( " "+str( "___ - DEBUG ----- CDDiagramSceneMainWidget: updateSceneBackgroundImage() lBackgroundRect =", lBackgroundRect, "done." )+" ", CDConstants.DebugTODO )
        # end of def updateSceneBackgroundImage(self, pText)
        # ------------------------------------------------------------


    # ------------------------------------------------------------
    # 2011 - Mitja: add code for new items being either a cell or a region:
    # ------------------------------------------------------------
    def handlerForButtonGroupRegionShapesClicked(self, id):
        buttons = self.theButtonGroupForRegionShapes.buttons()
        for button in buttons:
            if self.theButtonGroupForRegionShapes.button(id) != button:
                button.setChecked(False)

        if id == self.InsertTextButton:
            self.scene.setMode(CDConstants.SceneModeInsertText)
            # 2011 - Mitja: disable SceneModeImageLayer-specific buttons:
            self.theControlsForInputImagePicking.setEnabled(False)
        else:
            self.scene.setItemType(id)
            self.scene.setMode(CDConstants.SceneModeInsertItem)
            # 2011 - Mitja: disable SceneModeImageLayer-specific buttons:
            self.theControlsForInputImagePicking.setEnabled(False)




    # ------------------------------------------------------------
    # 2010 - Mitja: add code for toggling on/off the Table of Types Window:
    # ------------------------------------------------------------
    def handlePIFRegionTableButton(self, pString):
        # propagate the signal upstream, for example to parent objects:
        self.signalVisibilityPIFRegionTable.emit(pString)
#         CDConstants.printOut( " "+str( "handlePIFRegionTableButton" )+" ", CDConstants.DebugTODO )
        #         TODO add show/hide of table when there are no rows
        #         TODO add showing rows in the table when a new region is added
#         CDConstants.printOut( " "+str( "handlePIFRegionTableButton" )+" ", CDConstants.DebugTODO )
        pass



    # ------------------------------------------------------------
    # 2010 - Mitja: add code for cut/copy/paste of scene items:
    # ------------------------------------------------------------
    def cutItem(self):
        # pass a True parameter to copyItem(), signalling that we
        #   are about to delete the original, so copyItem() knows
        #   that the call has originated from a cutItem():
        self.copyItem(True)
        self.deleteItem()


    # ------------------------------------------------------------
    # 2010 - Mitja: add code for cut/copy/paste of scene items:
    # ------------------------------------------------------------
    def copyItem(self, pItemHasBeenCut = False):
        # the optional pItemHasBeenCut parameter is to be set to True
        #   if the call has originated from a cutItem(), i.e. the original
        #   item has now been deleted, so we keep its scene position unmodified.
        #   Otherwise (for copy-paste) it'll be shifted slightly, so that the user
        #   will see that it's a pasted item.

        lAllSelectedItems = self.scene.selectedItems()
        # make sure that there is at least one selected item:
        if not lAllSelectedItems:
            # CDConstants.printOut( " "+str( "#=#=#=#=#=#=#=# copyItem() lAllSelectedItems =", lAllSelectedItems )+" ", CDConstants.DebugTODO )
            return

        # we copy the first selected item:
        lFirstSelectedItem = lAllSelectedItems[0]
       
        # get all the necessary information from the item:
        lSelectedItemPosition = lFirstSelectedItem.scenePos()
        lSelectedItemHasBeenCut = pItemHasBeenCut
        lSelectedItemColor = lFirstSelectedItem.brush().color()
        lSelectedItemRegionOrCell = lFirstSelectedItem.itsaRegionOrCell
        lSelectedItemType = lFirstSelectedItem.diagramType
        lSelectedItemScaleX = lFirstSelectedItem.myScaleX
        lSelectedItemScaleY = lFirstSelectedItem.myScaleY
        lSelectedItemPolygon = lFirstSelectedItem.polygon()

        lPointListX = []       
        lPointListY = []       
        for lPointF in lSelectedItemPolygon:
            # CDConstants.printOut( " "+str( "lPointF = ", lPointF )+" ", CDConstants.DebugTODO )
            lPointListX.append(lPointF.x())
            lPointListY.append(lPointF.y())

        # CDConstants.printOut( " "+str( "COPY: lSelectedItemPosition =", lSelectedItemPosition )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemHasBeenCut =", lSelectedItemHasBeenCut )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemColor =", lSelectedItemColor )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemRegionOrCell = lSelectedItemRegionOrCell )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemType =", lSelectedItemType )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemScaleX =", lSelectedItemScaleX )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lSelectedItemScaleY =", lSelectedItemScaleY )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lPointListX =", lPointListX )+" ", CDConstants.DebugTODO )
        # CDConstants.printOut( " "+str( "COPY: lPointListY =", lPointListY )+" ", CDConstants.DebugTODO )

        # create an empty byte array:
        lItemDataByteArray = QByteArray()

        # open a data stream that can write to the new byte array:
        lDataStream = QtCore.QDataStream(lItemDataByteArray, QtCore.QIODevice.WriteOnly)

        # write into the datastream:
        lDataStream.writeQVariant(lSelectedItemPosition)
        lDataStream.writeQVariant(lSelectedItemHasBeenCut)
        lDataStream.writeQVariant(lSelectedItemColor)
        lDataStream.writeQVariant(lSelectedItemRegionOrCell)
        lDataStream.writeQVariant(lSelectedItemType)
        lDataStream.writeQVariant(lSelectedItemScaleX)
        lDataStream.writeQVariant(lSelectedItemScaleY)
        lDataStream.writeQVariant(lPointListX)
        lDataStream.writeQVariant(lPointListY)

        # place the byte array into a mime data container:
        lMimeData = QtCore.QMimeData()
        lMimeData.setData('application/x-pif-region-item', lItemDataByteArray)

        # 2010 - Mitja: this is our clipboard for cut/copy/paste operations,
        #   it provides access to the single QClipboard object in the application.
        # As from Qt documentation:
        # "Note: The QApplication object should already be constructed before accessing the clipboard."

        # place the MIME data into the application clipboard:
        lClipboard = QtGui.QApplication.clipboard()
        lClipboard.setMimeData(lMimeData)

        # fix for yet another PyQt bug, according to:
        #   http://www.mail-archive.com/pyqt@riverbankcomputing.com/msg17328.html
        # we have to manually emit a clipboard-related event, even though we
        #   just explicitly set its MIME data above.
        lEvent = QtCore.QEvent(QtCore.QEvent.Clipboard)
        QtGui.QApplication.sendEvent(lClipboard, lEvent)


    # ------------------------------------------------------------
    # 2010 - Mitja: add code for cut/copy/paste of scene items:
    # ------------------------------------------------------------
    def pasteItem(self):
        # get the application's clipboard:
        lClipboard = QtGui.QApplication.clipboard()
        if lClipboard == None:
            return

        # get the mime data from the application clipboard:
        lMimeData = lClipboard.mimeData()

        # if the MIME type is the one we specify, let's use the data:
        if lMimeData.hasFormat('application/x-pif-region-item'):
            # data() returns the a byte array containing the data stored in the clipboard,
            #   in the format described by the specified MIME type:
            lItemDataByteArray = lMimeData.data('application/x-pif-region-item')
            # open a data stream that can read from that byte array:
            lDataStream = QtCore.QDataStream(lItemDataByteArray, QtCore.QIODevice.ReadOnly)

            # according to Qt documentation:
            # "because QVariant is part of the QtCore library, it cannot provide conversion
            #  functions to data types defined in QtGui, such as QColor, QImage, and QPixmap.
            #  In other words, there is no toColor() function. Instead, you can use the
            #  QVariant::value() or the qVariantValue() template function"
            # But the value() template trick doesn't work with PyQt, and unfortunately
            #   there is no PyQt documentation to explain why, nor show any workarounds.

            # read from the datastream:
            lSelectedItemPosition = lDataStream.readQVariant().toPointF()
            lSelectedItemHasBeenCut = lDataStream.readQVariant().toBool()
            lSelectedItemColor = QtGui.QColor(lDataStream.readQVariant())           
            # a bit counterintuitively, the toInt() and toFloat() functions DON'T return
            #   ints or floats... they return couples of values. So we grab the first ones:
            lSelectedItemRegionOrCell = lDataStream.readQVariant().toInt()[0]
            lSelectedItemType = lDataStream.readQVariant().toInt()[0]
            lSelectedItemScaleX = lDataStream.readQVariant().toFloat()[0]
            lSelectedItemScaleY = lDataStream.readQVariant().toFloat()[0]
           
            lListOfPointsX = lDataStream.readQVariant().toList()
            lListOfPointsY = lDataStream.readQVariant().toList()
            CDConstants.printOut("PASTE: lListOfPointsX =", lListOfPointsX , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lListOfPointsY =", lListOfPointsY , CDConstants.DebugTODO )
            lLengthX = len(lListOfPointsX)
            lLengthY = len(lListOfPointsY)
            if lLengthX == lLengthY:
                pass
            else:
                CDConstants.printOut("PASTE: received INCONSISTENT paste data! Can't paste.", CDConstants.DebugTODO )
                return


            lSelectedItemPolygon = QtGui.QPolygonF()
            # CDConstants.printOut( " "+str( "PASTE: lSelectedItemPolygon =", lSelectedItemPolygon )+" ", CDConstants.DebugTODO )


            for i in xrange(lLengthX):
                # CDConstants.printOut( " "+str( "PASTE: lListOfPointsX[i].toFloat() = ", lListOfPointsX[i].toFloat() )+" ", CDConstants.DebugTODO )
                # CDConstants.printOut( " "+str( "PASTE: lListOfPointsY[i].toFloat() = ", lListOfPointsX[i].toFloat() )+" ", CDConstants.DebugTODO )
                lPointF = QtCore.QPointF(lListOfPointsX[i].toFloat()[0], lListOfPointsY[i].toFloat()[0])
                # CDConstants.printOut( " "+str( "PASTE: lPointF =", lPointF )+" ", CDConstants.DebugTODO )
                lSelectedItemPolygon.append(lPointF)


            CDConstants.printOut("PASTE: lSelectedItemPosition =", lSelectedItemPosition , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemHasBeenCut =", lSelectedItemHasBeenCut , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemColor =", lSelectedItemColor , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemRegionOrCell =", lSelectedItemRegionOrCell , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemType =", lSelectedItemType , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemScaleX =", lSelectedItemScaleX , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemScaleY =", lSelectedItemScaleY , CDConstants.DebugTODO )
            CDConstants.printOut("PASTE: lSelectedItemPolygon =", lSelectedItemPolygon , CDConstants.DebugTODO )

            # convert the polygon we've constructed from pasted QPointF values
            #   into a QPainterPath which can be digested by the DiagramItem constructor:
            lPath = QtGui.QPainterPath()
            lPath.addPolygon(lSelectedItemPolygon)
            lTheNewItem = DiagramItem(lPath, self.scene.myItemMenu)
            # set the new item's color i.e. brush value:
            lTheNewItem.setBrush(lSelectedItemColor)
            # 2011 - Mitja: the new item is a region of cells or a single cell:
            lTheNewItem.setRegionOrCell(lSelectedItemRegionOrCell)

            # finally, place the newly built item, i.e. "paste" it into the Cell Scene:
            self.scene.addItem(lTheNewItem)

            # to provide the default behavior of having the new item selected,
            #   first unselect all selected items in the scene:
            for anItem in self.scene.selectedItems():
                anItem.setSelected(False)
            #   then select the newly created item:
            lTheNewItem.setSelected(True)

            # emit our own signal to the handler which does other GUI adjustments
            #   whenever a new signal is inserted:
            self.scene.signalThatItemInserted.emit(lTheNewItem)

            # position the new item in the scene:
            if lSelectedItemHasBeenCut == False:
                # shift the copied item's position slightly if it wasn't "cut",
                #   so that when it's pasted back it is distinguishable from the original:
                lSelectedItemPosition.setX ( lSelectedItemPosition.x() + 13.0 )
                lSelectedItemPosition.setY ( lSelectedItemPosition.y() + 13.0 )
            lTheNewItem.setPos(lSelectedItemPosition)
           
            # set the same size transformation to the pasted item as it had from the copied/cut one:
            lTheNewItem.myScaleX = lSelectedItemScaleX
            lTheNewItem.myScaleY = lSelectedItemScaleY
            # using QTransform, we first create a transformation, then we apply it to the item:
            lTransform = QtGui.QTransform()
            lTransform.scale(  lTheNewItem.myScaleX,  lTheNewItem.myScaleY )
            lTheNewItem.setTransform( lTransform )
           
            # 2011 - Mitja: after setting the pasted item's color (AKA brush)
            #  also update the scene's regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            self.scene.addToRegionColorsInUse(lSelectedItemColor)

           


    # ------------------------------------------------------------
    def deleteItem(self):
        for item in self.scene.selectedItems():
            if isinstance(item, DiagramItem):
                item.removeArrows()

            # 2011 - Mitja: after obtaining the deleted item's color (AKA brush)
            #  also update the scene's regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            lSelectedItemColor = item.brush()
            self.scene.subtractFromRegionColorsInUse(lSelectedItemColor)

            self.scene.removeItem(item)
           
        # 2010 - Mitja - clear any outlines which may be in use for resizing:
        self.scene.stopOutlineResizing()
       
        # 2011 - Mitja - clear any x/y/w/h information about current item:
        lBoundingRect = QtCore.QRectF(0.0, 0.0, 0.0, 0.0)
        self.scene.signalThatItemResized.emit(lBoundingRect)
       


    # ------------------------------------------------------------------
    # 2010 Mitja - slot method handling "signalLayersSelectionModeHasChanged"
    #    events (AKA signals) arriving from  theControlsForLayerSelection:
    # ------------------------------------------------------------------
    def handleLayersSelectionModeHasChanged(self, pNewMode):

        CDConstants.printOut( "___ starting handleLayersSelectionModeHasChanged( pNewMode == "+str(pNewMode)+" ) ....", CDConstants.DebugTODO )
        # we used to query which button is checked explicitly, but we now
        #   receive that information as parameter directly from the signal:
        # lCheckedId = self.theControlsForLayerSelection.getCheckedButtonId()
        self.scene.setMode(pNewMode)

        # 2011 - Mitja: enable/disable SceneModeImageLayer-specific buttons:
        if self.scene.getMode() == CDConstants.SceneModeImageLayer:
            self.theControlsForInputImagePicking.setEnabled(True)
        else:
            self.theControlsForInputImagePicking.setEnabled(False)

        # 2011 - Mitja: enable/disable SceneModeImageSequence-specific controls:
        if self.scene.getMode() == CDConstants.SceneModeImageSequence:
            self.theControlsForImageSequence.setEnabled(True)
            # signal upstream about the updated usage of this region color:
            self.parentWindow.theTableOfTypes.updateTableOfTypesForImageSequenceOn()
            CDConstants.printOut( "        in handleLayersSelectionModeHasChanged() -- self.parentWindow.theTableOfTypes.updateTableOfTypesForImageSequenceOn() called.", CDConstants.DebugTODO )
        else:
            self.theControlsForImageSequence.setEnabled(False)
            # signal upstream about the updated usage of this region color:
            self.parentWindow.theTableOfTypes.updateTableOfTypesForImageSequenceOff()
            CDConstants.printOut( "        in handleLayersSelectionModeHasChanged() -- self.parentWindow.theTableOfTypes.updateTableOfTypesForImageSequenceOff() called.", CDConstants.DebugTODO )

        CDConstants.printOut( "___ ending handleLayersSelectionModeHasChanged( pNewMode == "+str(pNewMode)+" ) ....done.", CDConstants.DebugTODO )

    # ------------------------------------------------------------
    def bringToFront(self):
        if not self.scene.selectedItems():
            return

        selectedItem = self.scene.selectedItems()[0]
        overlapItems = selectedItem.collidingItems()

        zValue = 0
        for item in overlapItems:
            if (item.zValue() >= zValue and isinstance(item, DiagramItem)):
                zValue = item.zValue() + 0.1
        selectedItem.setZValue(zValue)

    # ------------------------------------------------------------
    def sendToBack(self):
        if not self.scene.selectedItems():
            return

        selectedItem = self.scene.selectedItems()[0]
        overlapItems = selectedItem.collidingItems()

        zValue = 0
        for item in overlapItems:
            if (item.zValue() <= zValue and isinstance(item, DiagramItem)):
                zValue = item.zValue() - 0.1
        selectedItem.setZValue(zValue)

    # ------------------------------------------------------------
    def handlerForItemInserted(self, item):
        self.theControlsForLayerSelection.setCheckedButton( \
            CDConstants.SceneModeMoveItem, True )
        lCheckedId = self.theControlsForLayerSelection.getCheckedButtonId()
        self.scene.setMode(lCheckedId)

        # 2011 - Mitja: disable SceneModeImageLayer-specific buttons:
        self.theControlsForInputImagePicking.setEnabled(False)

        # 2011 - Mitja: disable SceneModeImageSequence-specific buttons:
        self.theControlsForImageSequence.setEnabled(False)

        # 2010 - Mitja: add code for handling insertion of pixmap items:
        try:
            self.theButtonGroupForRegionShapes.button(item.diagramType).setChecked(False)
        except:
            CDConstants.printOut("EXCEPTION EXCEPTION EXCEPTION item item item item =", item , CDConstants.DebugTODO )
            pass



    # ------------------------------------------------------------
    def handlerForMouseMoved(self, pDict):
        lDict = dict(pDict)
        self.theControlsForInputImagePicking.setFreehandXLabel(lDict[0])
        self.theControlsForInputImagePicking.setFreehandYLabel(lDict[1])
        self.theControlsForInputImagePicking.setFreehandColorLabel( \
                QtGui.QColor( lDict[2], lDict[3], lDict[4] )      )



    # ------------------------------------------------------------
    def handlerForItemResized(self, pRect):
        lRect = QtCore.QRectF(pRect)
#         CDConstants.printOut( " "+str( lRect )+" ", CDConstants.DebugTODO )
        self.windowPIFControlPanel.setResizingItemXLabel(str(int(lRect.x())))
        self.windowPIFControlPanel.setResizingItemYLabel(str(int(lRect.y())))
        self.windowPIFControlPanel.setResizingItemWidthLabel(str(int(lRect.width())))
        self.windowPIFControlPanel.setResizingItemHeightLabel(str(int(lRect.height())))
#         self.scene.setMode(self.theControlsForLayerSelection.getCheckedButtonId())
#         self.theButtonGroupForRegionShapes.button(self.InsertTextButton).setChecked(False)
#         self.theControlsForInputImagePicking.setEnabled(False)


    # ------------------------------------------------------------
    # 2011 - Mitja: moving to a MVC design,
    #   this connect signal handler should go to the Controller object!
    # ------------------------------------------------------------
    def handlerForSceneResized(self, pDict):
        lDict = dict(pDict)
        print
        CDConstants.printOut("    TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )
        CDConstants.printOut(str( lDict ) , CDConstants.DebugTODO )
        CDConstants.printOut("    TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )
        print
        self.windowPIFControlPanel.setSceneWidthLabel(lDict[0])
        self.windowPIFControlPanel.setSceneHeightLabel(lDict[1])
        self.windowPIFControlPanel.setSceneDepthLabel(lDict[2])
        self.windowPIFControlPanel.setSceneUnitsLabel(lDict[3])


    # ------------------------------------------------------------
    def handlerForTextInserted(self, item):
        self.theButtonGroupForRegionShapes.button(self.InsertTextButton).setChecked(False)
        lCheckedId = self.theControlsForLayerSelection.getCheckedButtonId()
        self.scene.setMode(lCheckedId)
        # 2011 - Mitja: disable SceneModeImageLayer-specific buttons:
        self.theControlsForInputImagePicking.setEnabled(False)
        # 2011 - Mitja: disable SceneModeImageSequence-specific buttons:
        self.theControlsForInputImagePicking.setEnabled(False)

    # ------------------------------------------------------------
    def currentFontChanged(self, font):
        self.handleFontChange()

    # ------------------------------------------------------------
    def fontSizeChanged(self, font):
        self.handleFontChange()


    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle
    #    "current index changed" events (AKA signals) arriving from
    #    the object theSceneScaleZoomControl
    # ------------------------------------------------------------------
    def handleSceneScaleZoomChanged(self, pScale):
        lNewScale = pScale.left(pScale.indexOf("%")).toDouble()[0] / 100.0
        oldMatrix = self.view.matrix()
        self.view.resetMatrix()
        self.view.translate(oldMatrix.dx(), oldMatrix.dy())
        # take care of the RHS <-> LHS mismatch at its visible end,
        #   by flipping the y coordinate in the QGraphicsView's affine transformations:
        self.view.scale(lNewScale, -lNewScale)
        # 2011 Mitja - add scale/zoom parameters stored into our DiagramScene object:
        self.scene.myViewScaleZoomX = lNewScale
        self.scene.myViewScaleZoomY = lNewScale
        CDConstants.printOut( "___ - DEBUG ----- CDDiagramSceneMainWidget.handleSceneScaleZoomChanged() - self.scene.myViewScaleZoomX = " + \
            str(self.scene.myViewScaleZoomX) + " self.scene.myViewScaleZoomY = " + str(self.scene.myViewScaleZoomY)  , CDConstants.DebugVerbose )



    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle
    #    "current index changed" events (AKA signals) arriving from
    #    the object theControlsForInputImagePicking
    # ------------------------------------------------------------------
    def handleImageScaleZoomChanged(self, pScale):
        lNewScale = pScale.left(pScale.indexOf("%")).toDouble()[0] / 100.0

        if isinstance(self.scene.theImageLayer, CDImageLayer) == True:
            self.scene.theImageLayer.setScaleZoom( lNewScale )

        self.scene.update()

        CDConstants.printOut("      self.theCDImageLayer.scaleFactor = "+str(self.theCDImageLayer.scaleFactor) , CDConstants.DebugTODO )



    # ------------------------------------------------------------
    def textColorChanged(self):
        self.textAction = self.sender()
        self.fontColorToolButton.setIcon(self.createColorToolButtonIcon(
                    ':/icons/textpointer.png',
                    QtGui.QColor(self.textAction.data())))
        self.textButtonTriggered()


    # ------------------------------------------------------------
    def lineColorChanged(self):
        self.lineAction = self.sender()
        self.lineColorToolButton.setIcon(self.createColorToolButtonIcon(
                    ':/icons/linecolor.png',
                    QtGui.QColor(self.lineAction.data())))
        self.lineButtonTriggered()

    # ------------------------------------------------------------
    def textButtonTriggered(self):
        self.scene.setTextColor(QtGui.QColor(self.textAction.data()))

    # ------------------------------------------------------------
    def lineButtonTriggered(self):
        self.scene.setLineColor(QtGui.QColor(self.lineAction.data()))

    # ------------------------------------------------------------
    def handleFontChange(self):
        font = self.fontCombo.currentFont()
        font.setPointSize(self.fontSizeCombo.currentText().toInt()[0])
        if self.boldAction.isChecked():
            font.setWeight(QtGui.QFont.Bold)
        else:
            font.setWeight(QtGui.QFont.Normal)
        font.setItalic(self.italicAction.isChecked())
        font.setUnderline(self.underlineAction.isChecked())

        self.scene.setFont(font)

    # ------------------------------------------------------------
    def handlerForItemSelected(self, item):
        font = item.font()
        color = item.defaultTextColor()
        self.fontCombo.setCurrentFont(font)
        self.fontSizeCombo.setEditText(str(font.pointSize()))
        self.boldAction.setChecked(font.weight() == QtGui.QFont.Bold)
        self.italicAction.setChecked(font.italic())
        self.underlineAction.setChecked(font.underline())





    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "current type color" events
    #    (AKA signals) arriving from the object theControlsForTypes
    # ------------------------------------------------------------
    def handleTypesColorEvent(self, pColor):
        # CDConstants.printOut( " "+str( "handleTypesColorEvent received pColor =", pColor )+" ", CDConstants.DebugTODO )
        
        # this signal has to be handled differently according to the current mode:

        # 2011 - Mitja - use this overlay routine to draw selected content from an image sequence
        #   as foreground to the graphics scene, in "SceneModeImageSequence"
        if (self.scene.mySceneMode == CDConstants.SceneModeImageSequence):
            self.scene.setSequenceColor(QtGui.QColor(pColor))
        else:
            self.scene.setItemColor(QtGui.QColor(pColor))



    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "button change" events
    #    (AKA signals) arriving from the object theToggleforRegionOrCellDrawing
    # ------------------------------------------------------------------
    def handleToggleRegionOrCellDrawingChanged(self, pRegionOrCell):
        # this is a SLOT function for the signal "signalSetRegionOrCell()"
        #   from theToggleforRegionOrCellDrawing
        #
        # here we retrieve the updated value from radio buttons and update
        #   the global keeping track of the drawing mode:
        #     0 = Cell Draw = CDConstants.ItsaCellConst
        #     1 = Region Draw = CDConstants.ItsaRegionConst

        self.scene.setItemRegionOrCell(pRegionOrCell)

        CDConstants.printOut( "      self.handleToggleRegionOrCellDrawingChanged: "+str(pRegionOrCell) , CDConstants.DebugExcessive )





    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "button change" events
    #    (AKA signals) arriving from the object theControlsForInputImagePicking
    # ------------------------------------------------------------------
    def handleInputImagePickingModeChanged(self):
        # SLOT function for the signal "inputImagePickingModeChangedSignal()"
        #   from theControlsForInputImagePicking
        #
        # here we retrieve the updated values from radio buttons and update
        # the global keeping track of what's been changed:
        #    0 = Color Pick = CDConstants.ImageModePickColor
        #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
        #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
        #    3 = Extract Cells = CDConstants.ImageModeExtractCells
        self.theCDImageLayer.inputImagePickingMode = \
          self.theControlsForInputImagePicking.theInputImagePickingMode

        # if inputImagePickingMode is 3, i.e. Polygon Draw,
        #     then we need to track *passive* mouse motion;
        # same for mode 0, i.e. Color Pick, and mode 3, i.e. Extract Cells
        #
        # otherwise we only track *active* mouse motion (which is the default for QWidget),
        #     i.e. when at least one mouse button is pressed while the mouse is moved:
        if (self.theCDImageLayer.inputImagePickingMode == CDConstants.ImageModeDrawPolygon):
            self.theCDImageLayer.setMouseTracking(True)
        elif (self.theCDImageLayer.inputImagePickingMode == CDConstants.ImageModePickColor):
            self.theCDImageLayer.setMouseTracking(True)
        elif (self.theCDImageLayer.inputImagePickingMode == CDConstants.ImageModeExtractCells):
            self.theCDImageLayer.setMouseTracking(True)
        else:
            self.theCDImageLayer.setMouseTracking(False)

        self.scene.update()

        CDConstants.printOut("      self.theCDImageLayer.inputImagePickingMode = "+str(self.theCDImageLayer.inputImagePickingMode) , CDConstants.DebugTODO )







    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "slider change" events
    #    (AKA signals) arriving from the object theControlsForInputImagePicking
    # ------------------------------------------------------------------
    def handleImageOpacityChanged(self):
        # SLOT function for the signal "inputImageOpacityChangedSignal()"
        #   from theControlsForInputImagePicking
        #
        # here we retrieve the updated value from the opacity slider and update
        # from the class global keeping track of the required opacity:
        #      0 = minimum = the image is completely transparent (invisible)
        #    100 = minimum = the image is completely opaque

        if isinstance(self.scene.theImageLayer, CDImageLayer) == True:
            self.scene.theImageLayer.setImageOpacity( \
              self.theControlsForInputImagePicking.theImageOpacity )

        self.scene.update()

        CDConstants.printOut("      self.theCDImageLayer.imageOpacity = "+str(self.theCDImageLayer.imageOpacity) , CDConstants.DebugTODO )




    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "slider change" events
    #    (AKA signals) arriving from the object theControlsForInputImagePicking
    # ------------------------------------------------------------------
    def handleFuzzyPickThresholdChanged(self):
        # SLOT function for the signal "fuzzyPickTresholdChangedSignal()"
        #   from theControlsForInputImagePicking
        #
        # here we retrieve the updated value from the opacity slider and update
        # from the class global keeping track of the fuzzy pick treshold:
        #      0 = minimum = pick only the seed color
        #    100 = minimum = pick everything in the image

        if isinstance(self.scene.theImageLayer, CDImageLayer) == True:
            self.scene.theImageLayer.setFuzzyPickTreshold( \
              self.theControlsForInputImagePicking.theFuzzyPickTreshold )

        self.scene.update()

        CDConstants.printOut("      self.theCDImageLayer.fuzzyPickTreshold ="+str(self.theCDImageLayer.fuzzyPickTreshold) , CDConstants.DebugTODO )



    # ---------------------------------------------------------
    # 2011 - Mitja: handleMousePressedInImageLayerSignal() is called every time
    #   the theCDImageLayer object emits a "mousePressedInImageLayerSignal()" signal,
    #   (defined in the CDImageLayer class's mousePressEvent() function),
    # ---------------------------------------------------------
    def handleMousePressedInImageLayerSignal(self):

        # 2011 - Mitja: only do something about mouse clicks within the theCDImageLayer
        #   if the theCDImageLayer contains an image loaded from a file.
        # Otherwise do nothing:
        if self.theCDImageLayer.imageLoadedFromFile == False:
            CDConstants.printOut("2011 DEBUG: ---- ---- CDDiagramSceneMainWidget ---- ---- handleMousePressedInImageLayerSignal() has no input image. Returning.", CDConstants.DebugTODO )
            return

        # 2011 - Mitja: obtain the x,y coordinates where the mouse was clicked
        #   within the theCDImageLayer:
        x = self.theCDImageLayer.myMouseX
        y = self.theCDImageLayer.myMouseY
        # 2011 - Mitja: obtain the color of the pixel at coordinates i,j as
        #   QRgb type in the format #AARRGGBB (equivalent to an unsigned int) :
        lRGBAColorAtClickedPixel = self.theCDImageLayer.processedImage.pixel(x,y)

        # 2011 - Mitja: if asked to ignore white or black, ignore clicks on white or black:
        if  (self.parentWindow.ignoreWhiteRegionsForPIF == True) and (lRGBAColorAtClickedPixel == QtGui.QColor(QtCore.Qt.white).rgba()) :
            CDConstants.printOut("2011 DEBUG: ---- ---- CDDiagramSceneMainWidget ---- ---- handleMousePressedInImageLayerSignal() clicked on white. Returning.", CDConstants.DebugTODO )
            return # do nothing
        if (self.parentWindow.ignoreBlackRegionsForPIF == True) and (lRGBAColorAtClickedPixel == QtGui.QColor(QtCore.Qt.black).rgba()) :
            CDConstants.printOut("2011 DEBUG: ---- ---- CDDiagramSceneMainWidget ---- ---- handleMousePressedInImageLayerSignal() clicked on black. Returning.", CDConstants.DebugTODO )
            return # do nothing

        # 2011 - Mitja: now find if the picked color is actually in the list of scene colors,
        #   or if we need to compute the closest one:
        if lRGBAColorAtClickedPixel in self.parentWindow.colorDict:
            lRegionName = self.parentWindow.colorDict[lRGBAColorAtClickedPixel]
            lRGBAClosestColor = lRGBAColorAtClickedPixel
        else:
            lColorDistance = 9999.9
            lRGBAClosestColor = 0
            for lRegionColor in self.parentWindow.colorIds:
                r1 = QtGui.QColor(lRGBAColorAtClickedPixel).redF()
                g1 = QtGui.QColor(lRGBAColorAtClickedPixel).greenF()
                b1 = QtGui.QColor(lRGBAColorAtClickedPixel).blueF()
                r2 = QtGui.QColor(lRegionColor).redF()
                g2 = QtGui.QColor(lRegionColor).greenF()
                b2 = QtGui.QColor(lRegionColor).blueF()
                d =   ((r2-r1)*0.30) * ((r2-r1)*0.30) \
                    + ((g2-g1)*0.59) * ((g2-g1)*0.59) \
                    + ((b2-b1)*0.11) * ((b2-b1)*0.11)
                CDConstants.printOut("r1, g1, b1, r2, g2, b2, d, lColorDistance, lRGBAClosestColor ="+str(r1)+", "+str(g1)+", "+str(b1)+", "+str(r2)+", "+str(g2)+", "+str(b2)+", "+str(d)+", "+str(lColorDistance)+", "+str(lRGBAClosestColor) , CDConstants.DebugTODO )
                if (lColorDistance > d) :
                    lRGBAClosestColor = lRegionColor
                    lColorDistance = d

        CDConstants.printOut("Inserting", CDConstants.DebugTODO )
        CDConstants.printOut(str( lRGBAColorAtClickedPixel) , CDConstants.DebugTODO )
        CDConstants.printOut(str( lRGBAClosestColor) , CDConstants.DebugTODO )
        CDConstants.printOut(str( self.parentWindow.comboDict[lRGBAClosestColor] ) , CDConstants.DebugTODO )

        lClosestColor = QtGui.QColor(lRGBAClosestColor)
        lClickedPixelColor = QtGui.QColor(lRGBAColorAtClickedPixel)

       
        # 2011 - Mitja: this is how we'd get from key to region name:
        # key = (int)(self.parentWindow.ui.colorId2Text.text())
        # name =  self.parentWindow.colorDict[key]
       
        # 2011 - Mitja: add functionality for picking a color region:       
        if (self.parentWindow.pickColorRegion == True) :

            # 1. get a *pixmap copy* of the QImage from theCDImageLayer:
            lOriginalPixmap = QtGui.QPixmap( QtGui.QPixmap.fromImage(self.theCDImageLayer.processedImage) )
            # 2. create a "mask" QBitmap object from lOriginalPixmap's pixels NOT in the clicked color:
            lTheMaskBitmap = lOriginalPixmap.createMaskFromColor(lClickedPixelColor,  QtCore.Qt.MaskOutColor)
            # 3. apply lTheMaskBitmap to lOriginalPixmap, so that it eliminates all other color pixels from it:
            # ------> BUT the setMask() function applied to lOriginalPixmap will ALSO mask the theCDImageLayer... WHY???
            # lOriginalPixmap.setMask( lTheMaskBitmap )

            # 3. create an empty pixmap where to store the composed image:
            lResultPixmap = QtGui.QPixmap(self.theCDImageLayer.width, self.theCDImageLayer.height)
            # 4. create a QPainter to perform the overlay operation:
            lPainter = QtGui.QPainter(lResultPixmap)
            # 5. do the overlay:
            lPainter.setCompositionMode(lPainter.CompositionMode_Source)
            lPainter.fillRect(lResultPixmap.rect(), QtCore.Qt.transparent)
            lPainter.setCompositionMode(lPainter.CompositionMode_SourceOver)
            lPainter.drawPixmap(QtCore.QPoint(0,0), lOriginalPixmap)
            lPainter.end()           

            # don't copy the input image back to theCDImageLayer:
            # self.theCDImageLayer.setPixmap(lResultPixmap)
            # self.scene.update()
           
            # now obtain a QRegion from the pixmap:
            theRegionFromBitmap = QtGui.QRegion(lTheMaskBitmap)
            # create a QGraphicsItem from a QPainterPath created from the pixmap:
            thePainterPath = QtGui.QPainterPath()
            thePainterPath.addRegion(theRegionFromBitmap)
            theGraphicsPathItem = QtGui.QGraphicsPathItem(thePainterPath)

            # create a QGraphicsItem from the pixmap:
            theGraphicsPixmapItem = QtGui.QGraphicsPixmapItem(lOriginalPixmap)

            if (self.parentWindow.pickColorAsPath == False) :
                theGraphicsItem = theGraphicsPixmapItem
                # this always remains True since we only pick colors as paths (polygons) now:
                # self.parentWindow.pickColorAsPath = True
            else :
                theGraphicsItem = theGraphicsPathItem
                # this always remains True since we only pick colors as paths (polygons) now:
                # self.parentWindow.pickColorAsPath = False

            # set the graphics item's color to be the one we computed above
            #   (i.e. either it's already the same as in our list of scene colors,
            #    or we computed the closest one) :
            theGraphicsItem.setBrush(lClosestColor)

            # pass the QGraphicsItem to the external QGraphicsScene window:
            self.scene.mousePressEvent( \
                QtGui.QMouseEvent( QtCore.QEvent.GraphicsSceneMousePress, \
                                   QtCore.QPoint(x,y), \
                                   QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier), \
                theGraphicsItem )


            # 2011 - Mitja: restore the original QImage loaded from a file
            # back into the processed QImage, undoing all color processing.
            self.theCDImageLayer.setToProcessedImage()









    # ------------------------------------------------------------
    # 2011 - Mitja: moving to a MVC design, TODO:
    #   this connect signal handler should go to the Controller object!
    #
    # handleImageSequenceResized() responds to signalThatImageSequenceResized changes
    # ------------------------------------------------------------
    def handleImageSequenceResized(self, pDict):
        lDict = dict(pDict)
        CDConstants.printOut("CDDiagramSceneMainWidget - handleImageSequenceResized() - " + \
            str( lDict ), CDConstants.DebugTODO )
        self.windowPIFControlPanel.setImageSequenceWidthLabel(lDict[0])
        self.windowPIFControlPanel.setImageSequenceHeightLabel(lDict[1])
        self.windowPIFControlPanel.setImageSequenceDepthLabel(lDict[2])
        self.windowPIFControlPanel.setImageSequenceImageUnitsLabel(lDict[3])





    # ------------------------------------------------------------
    # 2011 - Mitja: moving to a MVC design, TODO:
    #   this connect signal handler should go to the Controller object!
    #
    # handleImageSequenceIndexSet() responds to signalThatCurrentIndexSet changes
    # ------------------------------------------------------------
    def handleImageSequenceIndexSet(self, pDict):
        lDict = dict(pDict)
        self.windowPIFControlPanel.setImageSequenceCurrentIndex(lDict[0])
        self.windowPIFControlPanel.setImageSequenceCurrentFilename(lDict[1])




    # ------------------------------------------------------------------
    # 2011 Mitja - slot method handling "signalSelectedImageInSequenceHasChanged" events
    #    (AKA signals) arriving from the object theControlsForImageSequence
    # ------------------------------------------------------------------
    def handleSelectedImageWithinSequenceChanged(self, pSelectedImage):

        if isinstance(self.scene.theImageSequence, CDImageSequence) == True:
            self.scene.theImageSequence.setCurrentIndexInSequence( pSelectedImage )

        self.scene.update()

        CDConstants.printOut( "      handleSelectedImageWithinSequenceChanged() - self.scene.theImageSequence.getCurrentIndex() = "+str(self.scene.theImageSequence.getCurrentIndex()) , CDConstants.DebugExcessive )


    # ------------------------------------------------------------------
    # 2011 Mitja - slot method handling "signalImageSequenceProcessingModeHasChanged" events
    #    (AKA signals) arriving from the object theControlsForImageSequence
    # ------------------------------------------------------------------
    def handleAreaOrEdgeModeHasChanged(self, pMode):
    
        CDConstants.printOut( "      handleAreaOrEdgeModeHasChanged() str(type(pMode))==["+str(type(pMode))+"]" , CDConstants.DebugTODO )
        CDConstants.printOut( "             str(type(pMode).__name__)==["+str(type(pMode).__name__)+"]" , CDConstants.DebugTODO )
        CDConstants.printOut( "             str(pMode)==["+str(pMode)+"]" , CDConstants.DebugTODO )
        # bin() does not exist in Python 2.5:
        if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
            CDConstants.printOut( "             str(bin(int(pMode)))==["+str(bin(int(pMode)))+"]" , CDConstants.DebugTODO )
        else:
            CDConstants.printOut( "             str(int(pMode))==["+str(int(pMode))+"]" , CDConstants.DebugTODO )

        if ( isinstance(self.scene.theImageSequence, CDImageSequence) == True ):
            # go and tell the image sequence in what mode it is now:
            self.scene.theImageSequence.assignAllProcessingModesForImageSequenceToPIFF( pMode )
            self.scene.update()

# 
# TODO:
# 
# Traceback (most recent call last):
#   File "/Volumes/30G/Users/mitja/Desktop/CellDraw/CellDraw/1.5.1/src/cdDiagramScene.py", line 3026, in handleAreaOrEdgeModeHasChanged
#     CDConstants.printOut( "      handleAreaOrEdgeModeHasChanged - self.scene.theImageSequence.assignAllProcessingModesForImageSequenceToPIFF( "+str(bin(pMode))+") " , CDConstants.DebugExcessive )
# TypeError: bin(QTextStream): argument 1 has unexpected type 'int'
# 
# 

            # bin() does not exist in Python 2.5:
            if ((sys.version_info[0] >= 2) and (sys.version_info[1] >= 6)) :
                CDConstants.printOut( "      handleAreaOrEdgeModeHasChanged() called self.scene.theImageSequence.assignAllProcessingModesForImageSequenceToPIFF( "+str(bin(pMode))+") complete." , CDConstants.DebugExcessive )
            else:
                CDConstants.printOut( "      handleAreaOrEdgeModeHasChanged() called self.scene.theImageSequence.assignAllProcessingModesForImageSequenceToPIFF( "+str(pMode)+") complete." , CDConstants.DebugExcessive )










    # ------------------------------------------------------------
    # 2011 - Mitja: add code for loading an entire scene from a saved .pifScene file:
    # ( file handling in PyQt inspired by:
    #   "mi_pyqt/examples/mainwindows/application/application.py" )
    # ------------------------------------------------------------


    # ------------------------------------------------------------
    def newSceneFile(self):
#         lUserChoice = QtGui.QMessageBox.warning(self, "CellDraw",
#                 "Do you want to delete your current Cell Scene, Types, and Regions?",
#                 QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard |
#                 QtGui.QMessageBox.Cancel)
        lNewSceneMessageBox = QtGui.QMessageBox(self)
        lNewSceneMessageBox.setWindowModality(QtCore.Qt.WindowModal)
        lNewSceneMessageBox.setIcon(QtGui.QMessageBox.Warning)
        # the "setText" sets the main large CDConstants.printOut( " "+str( text in the dialog box: )+" ", CDConstants.DebugTODO )
        lNewSceneMessageBox.setText("The current Cell Scene, Types and Regions will be discarded.")
        # the "setInformativeText" sets a smaller CDConstants.printOut( " "+str( text, below the main large print text in the dialog box: )+" ", CDConstants.DebugTODO )
        lNewSceneMessageBox.setInformativeText("Do you want to save your changes?")
        lNewSceneMessageBox.setStandardButtons(QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard |
               QtGui.QMessageBox.Cancel)
        lNewSceneMessageBox.setDefaultButton(QtGui.QMessageBox.Cancel)
        lUserChoice = lNewSceneMessageBox.exec_()

        if lUserChoice == QtGui.QMessageBox.Save:
            QtGui.QMessageBox.warning(self, "CellDraw", \
                    "Automatic cc3s SAVE yet to be implemented...\n" + \
                    "Please save manually, then try New Scene again.")
            return False
        elif lUserChoice == QtGui.QMessageBox.Cancel:
            # if the user hits "esc" or presses the "Cancel" button,
            #  there's nothing to be done:
            return False
        elif lUserChoice == QtGui.QMessageBox.Discard:
            # ------------------------------------------------------------
            # bring up the New Scene assistant AKA wizard
            lAssistant = CDSceneAssistant()
        
            # read some persistent-value globals from the preferences file on disk, if it already exists.
            self.cdPreferences.readPreferencesFromDisk()
            self.cdPreferences.readCC3DPreferencesFromDisk()
            lTheColorTable = self.cdPreferences.populateCellColors()
            lTheColorDict = self.cdPreferences.getCellColorsDict()
        
            lAssistant.setPreferencesObject(self.cdPreferences)
        
            lAssistant.addPage(lAssistant.createIntroPage(lTheColorTable))
            lAssistant.addPage(lAssistant.createCellTypePage(lTheColorDict))
            lAssistant.addPage(lAssistant.createRegionTypePage())
        
            lAssistant.show()
            lAssistant.raise_()
            lAssistant.exec_()
            
            # ------------------------------------------------------------
            
        else:
            lCriticalErrorWarning = QtGui.QMessageBox.critical( self, \
            "CellDraw", \
            "Critical eror -4554073\n\n." + \
            "Please contact your system administrator or the source where you obtained CellDraw." )
            sys.exit()

        CDConstants.printOut( "___ - DEBUG ----- CDDiagramSceneMainWidget: newSceneFile(): done.", CDConstants.DebugExcessive )
    #
    # end of  def newSceneFile(self)
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    def openSceneFile(self, pFileName):
        # TODO: for smoother user feedback, implement a "modified" bit
        #   to only allow saving when a scene actually contains unsaved content.
        # self.scene.setModified(False)
#         if self.maybeSave():
#             fileName = QtGui.QFileDialog.getOpenFileName(self)
#             if fileName:
#                 self.loadScenePIFDataFromFile(fileName)
        self.loadScenePIFDataFromFile(pFileName)

    # ------------------------------------------------------------
    def loadScenePIFDataFromFile(self, pFileName):
        CDConstants.printOut(  "2011 DEBUG: loadScenePIFDataFromFile("+str(pFileName)+") starting.", CDConstants.DebugExcessive )
        lQFileForReading = QtCore.QFile(pFileName)
        if not lQFileForReading.open(QtCore.QFile.ReadOnly):
            QtGui.QMessageBox.warning(self, "CellDraw",
                    "Cannot read file:\n%s\n%s." % (pFileName, lQFileForReading.errorString()))
            return False

        lDataStream = QtCore.QDataStream(lQFileForReading)


        lHowManyItems = lDataStream.readQVariant().toInt()[0]

        for i in xrange(lHowManyItems) :

            # read from the datastream:
            lSelectedItemPosition = lDataStream.readQVariant().toPointF()
            lSelectedItemColor = QtGui.QColor(lDataStream.readQVariant())           
            # a bit counterintuitively, the toInt() and toFloat() functions DON'T return
            #   ints or floats... they return couples of values. So we grab the first ones:
            lSelectedItemRegionOrCell = lDataStream.readQVariant().toInt()[0]
            lSelectedItemType = lDataStream.readQVariant().toInt()[0]
            lSelectedItemScaleX = lDataStream.readQVariant().toFloat()[0]
            lSelectedItemScaleY = lDataStream.readQVariant().toFloat()[0]
            lSelectedItemZValue = lDataStream.readQVariant().toFloat()[0]
           
            lListOfPointsX = lDataStream.readQVariant().toList()
            lListOfPointsY = lDataStream.readQVariant().toList()
#             CDConstants.printOut( " "+str( "LOAD: lListOfPointsX =", lListOfPointsX )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lListOfPointsY =", lListOfPointsY )+" ", CDConstants.DebugTODO )
            lLengthX = len(lListOfPointsX)
            lLengthY = len(lListOfPointsY)
            if lLengthX == lLengthY:
                pass
            else:
                CDConstants.printOut("LOAD: received INCONSISTENT paste data! Can't paste.", CDConstants.DebugTODO )
                return


            lSelectedItemPolygon = QtGui.QPolygonF()
            # CDConstants.printOut( " "+str( "LOAD: lSelectedItemPolygon =", lSelectedItemPolygon )+" ", CDConstants.DebugTODO )


            for i in xrange(lLengthX):
                # CDConstants.printOut( " "+str( "LOAD: lListOfPointsX[i].toFloat() = ", lListOfPointsX[i].toFloat() )+" ", CDConstants.DebugTODO )
                # CDConstants.printOut( " "+str( "LOAD: lListOfPointsY[i].toFloat() = ", lListOfPointsX[i].toFloat() )+" ", CDConstants.DebugTODO )
                lPointF = QtCore.QPointF(lListOfPointsX[i].toFloat()[0], lListOfPointsY[i].toFloat()[0])
                # CDConstants.printOut( " "+str( "LOAD: lPointF =", lPointF )+" ", CDConstants.DebugTODO )
                lSelectedItemPolygon.append(lPointF)


#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemPosition =", lSelectedItemPosition )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemColor =", lSelectedItemColor )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemRegionOrCell =", lSelectedItemRegionOrCell )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemType =", lSelectedItemType )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemScaleX =", lSelectedItemScaleX )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemScaleY =", lSelectedItemScaleY )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemZValue =", lSelectedItemZValue )+" ", CDConstants.DebugTODO )
#             CDConstants.printOut( " "+str( "LOAD: lSelectedItemPolygon =", lSelectedItemPolygon )+" ", CDConstants.DebugTODO )

            # convert the polygon we've constructed from pasted QPointF values
            #   into a QPainterPath which can be digested by the DiagramItem constructor:
            lPath = QtGui.QPainterPath()
            lPath.addPolygon(lSelectedItemPolygon)
            lTheNewItem = DiagramItem(lPath, self.scene.myItemMenu)
            # set the new item's color i.e. brush value:
            lTheNewItem.setBrush(lSelectedItemColor)

            # 2011 - Mitja: the new item is a region of cells or a single cell:
            lTheNewItem.setRegionOrCell(lSelectedItemRegionOrCell)

            # finally, place the newly built item, i.e. "paste" it into the Cell Scene:
            self.scene.addItem(lTheNewItem)

            # to provide the default behavior of having the new item selected,
            #   first unselect all selected items in the scene:
            for anItem in self.scene.selectedItems():
                anItem.setSelected(False)
            #   then select the newly created item:
            lTheNewItem.setSelected(True)

            # emit our own signal to the handler which does other GUI adjustments
            #   whenever a new signal is inserted:
            self.scene.signalThatItemInserted.emit(lTheNewItem)

            # position the new item in the scene:
            lTheNewItem.setPos(lSelectedItemPosition)
            lTheNewItem.setZValue(lSelectedItemZValue)
           
            # set the same size transformation to the pasted item as it had from the copied/cut one:
            lTheNewItem.myScaleX = lSelectedItemScaleX
            lTheNewItem.myScaleY = lSelectedItemScaleY
            # using QTransform, we first create a transformation, then we apply it to the item:
            lTransform = QtGui.QTransform()
            lTransform.scale(  lTheNewItem.myScaleX,  lTheNewItem.myScaleY )
            lTheNewItem.setTransform( lTransform )
           
            # 2011 - Mitja: after setting the pasted item's color (AKA brush)
            #  also update the scene's regionUseDict since it contains the list of all
            #  region colors in use by our scene:
            self.scene.addToRegionColorsInUse(lSelectedItemColor)


        lQFileForReading.close()
        self.setCurrentFile(pFileName)
        self.parentWindow.setStatusTip("Cell Scene read from file %s" % pFileName)
        return True


    # ------------------------------------------------------------
    # 2011 - Mitja: add code for saving an entire scene to a .pifScene file:
    # ( file handling in PyQt inspired by:
    #   "mi_pyqt/examples/mainwindows/application/application.py" )
    # ------------------------------------------------------------

    # TODO: for smoother user feedback, implement a "modified" bit
    #   to only allow saving when a scene actually contains unsaved content.
    # self.scene.setModified(False)
#     def maybeSave(self):
#         if self.textEdit.document().isModified():
#             ret = QtGui.QMessageBox.warning(self, "CellDraw",
#                     "Do you want to save your current Scene?",
#                     QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard |
#                     QtGui.QMessageBox.Cancel)
#             if ret == QtGui.QMessageBox.Save:
#                 return self.save()
#             elif ret == QtGui.QMessageBox.Cancel:
#                 return False
#         return True

    # ------------------------------------------------------------
    def setCurrentFile(self, pFileName):
        self.curFile = pFileName
        # TODO: for smoother user feedback, implement a "modified" bit
        #   to only allow saving when a scene actually contains unsaved content.
        # self.scene.setModified(False)
        self.setWindowModified(False)

        if self.curFile:
            shownName = self.strippedName(self.curFile)
        else:
            shownName = 'untitled.pifScene'

        self.setWindowTitle("%s[*] - CellDraw" % shownName)

    def strippedName(self, fullFileName):
        return QtCore.QFileInfo(fullFileName).fileName()

    # ------------------------------------------------------------
    def saveSceneFile(self):
        if self.curFile:
            lFileSaved = self.saveScenePIFDataToFile(self.curFile)
        else:
            lFileSaved = self.saveAs()
        return lFileSaved

    # ------------------------------------------------------------
    def saveAs(self):

        # 2011 - Mitja: setup local variables for file saving:
        lToBeSavedFileExtension = QtCore.QString("pifScene")
        lToBeSavedInitialPath = QtCore.QDir.currentPath() + self.tr("/untitled.") + lToBeSavedFileExtension

        lFileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("CellDraw - Save Scene As"),
                               lToBeSavedInitialPath,
                               self.tr("%1 files (*.%2);;All files (*)")
                                   .arg(lToBeSavedFileExtension)
                                   .arg(lToBeSavedFileExtension))

        if lFileName:
            lFileSaved = self.saveScenePIFDataToFile(lFileName)
        else:
            lFileSaved = False
        return lFileSaved

    # ------------------------------------------------------------
    def saveScenePIFDataToFile(self, pFileName):
        lAllItems = self.scene.items()
        lHowManyItems = len(lAllItems)

        CDConstants.printOut( str(lHowManyItems) + ", " + str(lAllItems) + " ", CDConstants.DebugTODO )

        # make sure that there is at least one item in the scene:
        if not lAllItems:
            CDConstants.printOut("#=#=#=#=#=#=#=# saveScenePIFDataToFile() lAllItems =", lAllItems , CDConstants.DebugTODO )
            print
            CDConstants.printOut("there is NOTHING in your Cell Scene!!!", CDConstants.DebugTODO )
            print
            return


        lQFileForWriting = QtCore.QFile(pFileName)
        if not lQFileForWriting.open(QtCore.QFile.WriteOnly):
            QtGui.QMessageBox.warning(self, "CellDraw",
                    "Cannot write file %s.\nError: %s." % (pFileName, lQFileForWriting.errorString()))
            return False

        # if we wrote into memory first, we'd create an empty byte array:
        # lItemDataByteArray = QByteArray()

        # open a data stream that can write *not* to the new byte array,
        #   but to the file we just opened:
        # lDataStream = QtCore.QDataStream(lItemDataByteArray, QtCore.QIODevice.WriteOnly)
        lDataStream = QtCore.QDataStream(lQFileForWriting)
       
        # first, write how many items we have,
        #   for easier loading when reopening the file:
        lDataStream.writeQVariant(lHowManyItems)

        # extract all the necessary data for *each* scene item individually:
        for lAnItem in lAllItems:
            # get all the necessary information from the item:
            lSelectedItemPosition = lAnItem.scenePos()
            lSelectedItemColor = lAnItem.brush().color()
            lSelectedItemRegionOrCell = lAnItem.itsaRegionOrCell
            lSelectedItemType = lAnItem.diagramType
            lSelectedItemScaleX = lAnItem.myScaleX
            lSelectedItemScaleY = lAnItem.myScaleY
            lSelectedItemZValue = lAnItem.zValue()
            lSelectedItemPolygon = lAnItem.polygon()
            lPointListX = []       
            lPointListY = []       
            for lPointF in lSelectedItemPolygon:
                lPointListX.append(lPointF.x())
                lPointListY.append(lPointF.y())

            # write into the QDataStream just opened above:
            # the order in which data is written is important!!!
            lDataStream.writeQVariant(lSelectedItemPosition)
            lDataStream.writeQVariant(lSelectedItemColor)
            lDataStream.writeQVariant(lSelectedItemRegionOrCell)
            lDataStream.writeQVariant(lSelectedItemType)
            lDataStream.writeQVariant(lSelectedItemScaleX)
            lDataStream.writeQVariant(lSelectedItemScaleY)
            lDataStream.writeQVariant(lSelectedItemZValue)
            lDataStream.writeQVariant(lPointListX)
            lDataStream.writeQVariant(lPointListY)

        # we could place the byte array into a mime data container?
        # lMimeData = QtCore.QMimeData()
        # lMimeData.setData('application/x-pif-scene', lItemDataByteArray)

        lQFileForWriting.close()
        self.setCurrentFile(pFileName)
        self.parentWindow.setStatusTip("Cell Scene written to file %s" % pFileName)
        return True


    # ------------------------------------------------------------
    def about(self):
       
        lAboutString = "CellDraw 1.5.1<br><br>An editing and conversion software tool for PIFF files, as used in CompuCell3D simulations.<br><br>CellDraw can be useful for creating PIFF files containing a high number of cells and cell types, either by drawing a scene containing cell regions in a paint program, and then discretize the drawing into a PIFF file, or by drawing the cell scenario directly in CellDraw.<br><br>More information at:<br><a href=\"http://www.compucell3d.org/\">http://www.compucell3d.org/</a>"

        lVersionString = "<br><br><small>Support library information:<br>Python runtime version: %s<br>Qt runtime version: %s<br>Qt compile-time version: %s<br>PyQt version: %s (%s = 0x%06x)</small>" % \
            ( str(sys.version_info[0])+"."+str(sys.version_info[1])+"."+str(sys.version_info[2])+" | "+str(sys.version_info[3])+" | "+str(sys.version_info[4]) , \
            QtCore.QT_VERSION_STR, QtCore.qVersion(), PyQt4.QtCore.PYQT_VERSION_STR, PyQt4.QtCore.PYQT_VERSION, PyQt4.QtCore.PYQT_VERSION)

        QtGui.QMessageBox.about(self, "About CellDraw", lAboutString+lVersionString)


    # ------------------------------------------------------------
    # 2011 - Mitja: connectSignals() contains all signal<->handler(AKA slot)
    #   connections for the CDDiagramSceneMainWidget class:
    # ------------------------------------------------------------
    def connectSignals(self):

        # connect signals in other objects to this class' methods:
        self.scene.signalThatItemInserted.connect(self.handlerForItemInserted)
        self.scene.signalThatTextInserted.connect(self.handlerForTextInserted)
        self.scene.signalThatItemSelected.connect(self.handlerForItemSelected)

        self.scene.signalThatItemResized.connect(self.handlerForItemResized)

        self.scene.signalThatSceneResized.connect(self.handlerForSceneResized)


        self.scene.theImageLayer.signalThatMouseMoved.connect(self.handlerForMouseMoved)


        # signal<->handler(AKA slot) connections, connect signals to this class' methods:
        self.theCDImageSequence.signalThatImageSequenceResized.connect( \
            self.handleImageSequenceResized )

        # signal<->handler(AKA slot) connections, connect signals to this class' methods:
        self.theCDImageSequence.signalThatCurrentIndexSet.connect( \
            self.handleImageSequenceIndexSet )



        # 2011 - Mitja: connect the sender (i.e. self.theCDImageLayer) object's signal
        #  (i.e. "mousePressedInImageLayerSignal()" in CDImageLayer's mousePressEvent() ),
        #  to the receiver fuction (i.e. handleMousePressedInImageLayerSignal() here.) :
        self.connect( self.theCDImageLayer,  \
            QtCore.SIGNAL("mousePressedInImageLayerSignal()"), \
            self.handleMousePressedInImageLayerSignal )

        # you can use this syntax instead of the 'old' one:
        self.mysignal.connect(self.myslot)

        # but this will also work
        self.connect(self, QtCore.SIGNAL('mysignal(QString)'), self.myslot)

        self.mysignal.emit("hello")



    # ------------------------------------------------------------
    # 2011 - Mitja: myslot is a test handler (AKA slot) for signals:
    # ------------------------------------------------------------
    def myslot(self, param):
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )
        CDConstants.printOut("received %s" % param , CDConstants.DebugTODO )





    # ------------------------------------------------------------


    # ------------------------------------------------------------
    def createButtonGroupForRegionShapes(self):
        lButtonGroup = QtGui.QButtonGroup()
        lButtonGroup.setExclusive(False)
        lButtonGroup.buttonClicked[int].connect(self.handlerForButtonGroupRegionShapesClicked)
        return lButtonGroup


    # ------------------------------------------------------------
    def createButtonGroupForBackgrounds(self):
        lButtonGroup = QtGui.QButtonGroup()
        lButtonGroup.buttonClicked.connect(self.handlerForButtonGroupBackgroundsClicked)
        return lButtonGroup









    # ------------------------------------------------------------
    def createSceneEditActions(self):

        # Note: PyQt 4.8.6 seems to have problems with assigning the proper key shortcuts
        #   using mnemonics such as:
        #      shortcut=QtGui.QKeySequence.Cut
        #      shortcut=QtGui.QKeySequence.Copy
        #      shortcut=QtGui.QKeySequence.Paste
        # so we have to set the shortcuts explicitly to "Ctrl+key" ...
   
        self.cutAction = QtGui.QAction(
                QtGui.QIcon(':/icons/cutRegion.png'), "Cut", self,
                shortcut="Ctrl+X", statusTip="Cut a region from the Cell Scene",
                triggered=self.cutItem)

        self.copyAction = QtGui.QAction(
                QtGui.QIcon(':/icons/copyRegion.png'), "Copy", self,
                shortcut="Ctrl+C", statusTip="Copy a region from the Cell Scene",
                triggered=self.copyItem)

        self.pasteAction = QtGui.QAction(
                QtGui.QIcon(':/icons/pasteRegion.png'), "Paste", self,
                shortcut="Ctrl+V", statusTip="Paste a region to the Cell Scene",
                triggered=self.pasteItem)

        # Another cross-platform Qt messy implementation: according to Qt documentation,
        #   the "Delete" shortcut maps to the "Del" button on Mac OS X.
        #   But on Mac OS X there is NO "Del" button: there are the "Delete" button
        #   and the "forward delete" button. To obtain correct behavior, we therefore have
        #   to use the "Backspace" Qt shortcut, which maps to the "Delete" button on Mac OS X.
        #   Qt continuously exhibits poorly implemented cross-platform functionalities.
        if sys.platform=='darwin':
            self.deleteAction = QtGui.QAction(QtGui.QIcon(':/icons/deleteRegion.png'),
                    "&Delete", self, shortcut="Backspace",
                    statusTip="Delete a region from the Cell Scene",
                    triggered=self.deleteItem)
        else:
            self.deleteAction = QtGui.QAction(QtGui.QIcon(':/icons/deleteRegion.png'),
                    "&Delete", self, shortcut="Delete",
                    statusTip="Delete a region from the Cell Scene",
                    triggered=self.deleteItem)

        self.toFrontAction = QtGui.QAction(
                QtGui.QIcon(':/icons/bringtofront.png'), "Bring to &Front",
                self, shortcut="Ctrl+F", statusTip="Bring a region to the front of the Cell Scene",
                triggered=self.bringToFront)

        self.sendBackAction = QtGui.QAction(
                QtGui.QIcon(':/icons/sendtoback.png'), "Send to &Back", self,
                shortcut="Ctrl+B", statusTip="Send a region to the back of the Cell Scene",
                triggered=self.sendToBack)


        # 2010 - Mitja: one more cross-platform Qt messup:
        #   since when is Ctrl-X a standard cross-platform shortcut for quitting an application???
        # 2010 - Mitja: GUI simplification: hide all that's not necessary to Cell Scene editing:
        #   in this case don't create the exitAction, it's redundant:
        # self.exitAction = QtGui.QAction("E&xit", self, shortcut="Ctrl+X",
        #         statusTip="Quit Scenediagram example", triggered=self.close)

        self.boldAction = QtGui.QAction(QtGui.QIcon(':/icons/bold.png'),
                "Bold", self, checkable=True, shortcut="Ctrl+B",
                triggered=self.handleFontChange)

        self.italicAction = QtGui.QAction(QtGui.QIcon(':/icons/italic.png'),
                "Italic", self, checkable=True, shortcut="Ctrl+I",
                triggered=self.handleFontChange)

        self.underlineAction = QtGui.QAction(
                QtGui.QIcon(':/icons/underline.png'), "Underline", self,
                checkable=True, shortcut="Ctrl+U",
                triggered=self.handleFontChange)

        self.aboutAction = QtGui.QAction("About", self, \
                shortcut="Ctrl+E", \
                triggered=self.about)

        CDConstants.printOut("___ - DEBUG ----- CDDiagramSceneMainWidget: createSceneEditActions() done.", CDConstants.DebugTODO )

        # end of def createSceneEditActions(self)
        # ------------------------------------------------------------
       
       

    # ------------------------------------------------------------
    # 2010 - Mitja: is there any real need for menus,
    #     when all actions are reachable from toolbars?
    # ------------------------------------------------------------
    def createMenus(self):
        # 2010 - Mitja: GUI simplification: hide all that's not necessary to Cell Scene editing:
        #   in this case don't add the fileMenu to the main menubar, it's redundant:
        # self.fileMenu = self.parentWindow.menuBar().addMenu("&File")
        # self.fileMenu.addAction(self.exitAction)
        CDConstants.printOut(str( self.parentWindow ), CDConstants.DebugTODO )
        self.editMenu = self.parentWindow.menuBar().addMenu("Edit")
        self.editMenu.addAction(self.cutAction)
        self.editMenu.addAction(self.copyAction)
        self.editMenu.addAction(self.pasteAction)
        self.editMenu.addAction(self.deleteAction)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.toFrontAction)
        self.editMenu.addAction(self.sendBackAction)

        self.windowMenu = self.parentWindow.menuBar().addMenu("Window")

        self.aboutMenu = self.parentWindow.menuBar().addMenu("Help")
        self.aboutMenu.addAction(self.aboutAction)

    # ------------------------------------------------------------
#    def createToolbars(self):

        # 2010 - Mitja: dummy "Toolbar" toolbar, it contains nothing, but
        #    it seems to be necessary to bypass a Qt bug which prevents the first toolbar
        #    added to a parent window from showing up there.
        #
        # self.dummyToolBar = self.parentWindow.addToolBar("Toolbar")


        # ------------------------------------------------------------
        # 2010 - Mitja: "Color" toolbar, it's built from 3 pop-up menu buttons,
        #    one each for font color, item fill color, and line color.
        #    Menu defaults are set here and for consistency they *must* coincide
        #      with the defaults set in the DiagramScene class globals:
        #    But really only use the item fill color, and we disable the other two:
        #
        #
#         self.fontColorToolButton = QtGui.QToolButton()
#         self.fontColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
#         self.fontColorToolButton.setMenu(
#                 self.createColorMenu(self.textColorChanged, QtCore.Qt.red))
#         self.textAction = self.fontColorToolButton.menu().defaultAction()
#         self.fontColorToolButton.setIcon(
#                 self.createColorToolButtonIcon(':/icons/textpointer.png',
#                         QtCore.Qt.red))
#         self.fontColorToolButton.setAutoFillBackground(True)
#         self.fontColorToolButton.clicked.connect(self.textButtonTriggered)
#         #
#         self.lineColorToolButton = QtGui.QToolButton()
#         self.lineColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
#         self.lineColorToolButton.setMenu(
#                 self.createColorMenu(self.lineColorChanged, QtCore.Qt.black))
#         self.lineAction = self.lineColorToolButton.menu().defaultAction()
#         self.lineColorToolButton.setIcon(
#                 self.createColorToolButtonIcon(':/icons/linecolor.png',
#                         QtCore.Qt.black))
#         self.lineColorToolButton.clicked.connect(self.lineButtonTriggered)

#        self.colorToolBar = self.parentWindow.addToolBar("Color")
        # 2010 - Mitja: GUI simplification: hide all that's not necessary to Cell Scene editing:
        #   in this case don't add to the colorToolbar the fontColorToolButton and lineColorToolButton button-popup-menus:
        # self.colorToolBar.addWidget(self.fontColorToolButton)
        # self.colorToolBar.addWidget(self.lineColorToolButton)
        # self.colorToolBar.addWidget(self.fillColorToolButton)
#        self.colorToolBar.addAction(self.pifTableAction)


        # ------------------------------------------------------------
        # 2010 - Mitja: "Font" toolbar:
        #   (we don't currently use this toolbar in the Cell Scene editor)
        # 
#         self.fontCombo = QtGui.QFontComboBox()
#         self.fontCombo.currentFontChanged.connect(self.currentFontChanged)
#         #
#         self.fontSizeCombo = QtGui.QComboBox()
#         self.fontSizeCombo.setEditable(True)
#         for i in range(8, 30, 2):
#             self.fontSizeCombo.addItem(str(i))
#         validator = QtGui.QIntValidator(2, 64, self)
#         self.fontSizeCombo.setValidator(validator)
#         self.fontSizeCombo.currentIndexChanged.connect(self.fontSizeChanged)
        #
        # 2010 - Mitja: GUI simplification: hide all that's not necessary to Cell Scene editing:
        #   in this case don't add to the parentWindow the textToolBar for handling text fonts:
        # self.textToolBar = self.parentWindow.addToolBar("Font")
        # just create the QToolBar and hide it, in case not having it would bring havoc to the application!
#         self.textToolBar = QtGui.QToolBar("Font")
#         self.textToolBar.addWidget(self.fontCombo)
#         self.textToolBar.addWidget(self.fontSizeCombo)
#         self.textToolBar.addAction(self.boldAction)
#         self.textToolBar.addAction(self.italicAction)
#         self.textToolBar.addAction(self.underlineAction)
#         self.textToolBar.hide()



    # ------------------------------------------------------------
    # 2010 - Mitja: add code for new backgrounds:
    # ------------------------------------------------------------
    def updateBackgroundImage(self, pText, pImage):
        # update globals in the cdDiagramScene:
        self.theImageFromFile = pImage
        self.theBackgroundNameFromFile = pText
        # update the appearance of the Control Panel buttons for background selection:
        self.windowPIFControlPanel.updateBackgroundImageButtons(pText, pImage)
               
       

    # ------------------------------------------------------------
    # 2010 - Mitja: add code for handling changing size of graphics scene rectangle:
    # ------------------------------------------------------------
    def updateSceneRectSize(self):
       
        # emit a signal to update scene size GUI controls:
       
        lDict = { \
            0: str(int(self.scene.width())), \
            1: str(int(self.scene.height())), \
            #  the depth() function is not part of QGraphicsScene, we add it for completeness:
            2: str(int(self.scene.depth())), \
            3: str(self.scene.mySceneUnits) \
            }


        # this crashes on Linux!!!"
        CDConstants.printOut(" TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )
        # although version PIF Generator 1.2.1  works?!?! And it also displays the table correctly!!!!
        self.scene.signalThatSceneResized.emit(lDict)
        # version Ubuntu 10.04 LTS the Lucid Linux released April 2010
        # default Python is 2.6.5
        # has installed Qt 4.6.2
        # has installed PyQt 263938
        CDConstants.printOut(" TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO ", CDConstants.DebugTODO )


        # self.sceneWidthLabel.setText(str(int(self.scene.width())))
        # self.sceneHeightLabel.setText(str(int(self.scene.height())))
        # self.sceneDepthLabel.setText(str(int(self.scene.depth())))
        # self.sceneUnitsLabel.setText(str(self.scene.mySceneUnits))

        lText = ""
        buttons = self.theButtonGroupForBackgrounds.buttons()
        for myButton in buttons:
            if myButton.isChecked() == True:
                lText = myButton.text()
                # CDConstants.printOut( " "+str( "myButton.text() =", myButton.text() )+" ", CDConstants.DebugTODO )
               
        self.updateSceneBackgroundImage(lText)

#         if self.theImageNameFromFile is "BlankBackground":
#             lBoringPixMap = QtGui.QPixmap(int(self.scene.width()), int(self.scene.height()))
#             lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.white) )
#             self.theImageFromFile = QtGui.QImage(lBoringPixMap)
#             self.updateBackgroundImage(self.theImageNameFromFile, self.theImageFromFile)
           



# 2011 - Mitja: uncomment createPixmapCellWidget to use pixmap items in Cell Scene:
#
#     # ------------------------------------------------------------
#     # 2010 - Mitja: add code for handling insertion of pixmap items:
#     # ------------------------------------------------------------
#     def createPixmapCellWidget(self, pText) : #  was: , pPixmap):
#
#         # 2010 - Mitja: if there is no pixmap yet, make it a simple boring one:
#         lBoringPixMap = QtGui.QPixmap(128, 128)
#         lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.darkGray) )
#         lPixmap = lBoringPixMap
#         item = DiagramPixmapItem(lPixmap, self.editMenu)
#         icon = QtGui.QIcon(item.pixmapForIconFromPolygon())
#
#         button = QtGui.QToolButton()
#         button.setIcon(icon)
#         button.setIconSize(QtCore.QSize(50, 50))
#         button.setCheckable(True)
#         self.theButtonGroupForRegionShapes.addButton(button)
#
#         layout = QtGui.QGridLayout()
#         layout.setMargin(2)
#         layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
#         layout.addWidget(QtGui.QLabel(pText), 1, 0, QtCore.Qt.AlignCenter)
#
#         widget = QtGui.QWidget()
#         widget.setLayout(layout)
#
#         return widget



    def createColorToolButtonIcon(self, imageFile, color):
        pixmap = QtGui.QPixmap(50, 80)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        lImage = QtGui.QPixmap(imageFile)
        target = QtCore.QRect(0, 0, 50, 60)
        source = QtCore.QRect(0, 0, 42, 42)
        painter.fillRect(QtCore.QRect(0, 60, 50, 80), color)
        painter.drawPixmap(target, lImage, source)
        painter.end()

        return QtGui.QIcon(pixmap)




# ------------------------------------------------------------
# ------------------------------------------------------------
if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)


    mainWindow = QtGui.QMainWindow()
    mainWindow.setGeometry(100, 100, 900, 500)

    mainWidget = CDDiagramSceneMainWidget(mainWindow)
    # mainWidget.setGeometry(100, 100, 900, 500)
   
    mainWindow.setCentralWidget(mainWidget)

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    mainWindow.raise_()
    mainWindow.show()

    sys.exit(app.exec_())

# ------------------------------------------------------------
# ------------------------------------------------------------

# Local Variables:
# coding: US-ASCII
# End:
