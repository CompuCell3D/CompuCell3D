#!/usr/bin/env python


# ======================================================================
# all application-global constants needed in CellDraw code are
#    placed here, for easier access from various classes
# ======================================================================


# 2010 - Mitja:
import inspect # for debugging functions, remove in final version
# debugging functions, remove in final version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]


from PyQt4 import QtCore


# ----------------------------------------------------------------------
# 2011- Mitja: CDConstants doesn't do much except declare values
# ------------------------------------------------------------
class CDConstants(QtCore.QObject):
    # --------------------------------------------------------


    # 2010 - Mitja: mode for editing PIFF scene Items:
    #     (originally it was defined in the DiagramScene class as:
    #      InsertItem, InsertLine, InsertText, MoveItem  = range(4) )
    SceneModeInsertItem, SceneModeInsertLine, SceneModeInsertText, \
        SceneModeMoveItem, SceneModeInsertPixmap, SceneModeResizeItem, \
        SceneModeImageLayer, SceneModeImageSequence, SceneModeEditCluster = range(9)


    # 2010 - Mitja: mode for saving PIFF files, different generating methods:
    PIFFSaveWithFixedRaster, PIFFSaveWithOneRasterPerRegion, \
        PIFFSaveWithPotts = range(3)


    # 2011 - Mitja: toggle between drawing cells vs. regions,
    #    with a constant keeping track of what is to be used:
    #    0 = Cell Draw = CDConstants.ItsaCellConst
    #    1 = Region Draw = CDConstants.ItsaRegionConst
    #    2 = Cluster Draw = CDConstants.ItsaClusterConst
    ItsaCellConst, ItsaRegionConst, ItsaClusterConst = range(3)


    # 2011 - Mitja: mode for drawing color regions and cells:
    #    0 = Color Pick = CDConstants.ImageModePickColor
    #    1 = Freehand Draw = CDConstants.ImageModeDrawFreehand
    #    2 = Polygon Draw = CDConstants.ImageModeDrawPolygon
    #    3 = Extract Cells = CDConstants.ImageModeExtractCells
    ImageModePickColor, ImageModeDrawFreehand, ImageModeDrawPolygon, ImageModeExtractCells  = range(4)


    # 2011 - Mitja: bit-flag modes for generating PIFF from displayed imported image sequence:
    #    0 = Use Discretized Images to B/W = CDConstants.ImageSequenceUseDiscretizedToBWMode
    #    1 = Region 2D Edge = CDConstants.ImageSequenceUse2DEdges
    #    2 = Region 3D Contours = CDConstants.ImageSequenceUse3DContours
    #    3 = Region 3D Volume = CDConstants.ImageSequenceUse3DVolume
    #    4 = Region Cell Seeds = CDConstants.ImageSequenceUseAreaSeeds
    ImageSequenceUseDiscretizedToBWMode, ImageSequenceUse2DEdges, ImageSequenceUse3DContours, \
        ImageSequenceUse3DVolume, ImageSequenceUseAreaSeeds = range(5)



    # 2012 - Mitja: bit-flag modes for NumPy image processing:
    #    0 = Use Discretized Images to B/W = CDConstants.ImageNPUseDiscretizedToBWMode
    #    1 = Region 2D Edge = CDConstants.ImageNPUseEdge
    #    2 = Region 3D Contours = CDConstants.ImageNPUse3DContours
    #    3 = Region 3D Volume = CDConstants.ImageNPUse3DVolume
    #    4 = Region 2D Cell Contours = CDConstants.ImageNPUseAreaCells
    ImageNPUseDiscretizedToBWMode, ImageNPUseEdge, ImageNPUse3DContours, \
        ImageNPUse3DVolume, ImageNPUseAreaCells = range(5)


    TypesColors = [QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.red, \
                   QtCore.Qt.darkYellow, QtCore.Qt.lightGray, QtCore.Qt.magenta, \
                   QtCore.Qt.darkBlue, QtCore.Qt.cyan, QtCore.Qt.darkGreen, QtCore.Qt.white]
    TypesColorNames = ["green", "blue", "red", "darkYellow", "lightGray", \
                  "magenta", "darkBlue", "cyan", "darkGreen", "white"]


    # 2011 - Mitja: debugging level:
    #    0 = DebugImportant = no debug info to stderr, only crucially important messages!
    #    1 = DebugSparse = some debug info
    #    2 = DebugVerbose = print out a lot of information, such as global variables' values
    #    3 = DebugExcessive = print out an excessive amount of a lot of information, such as reaching begin/end of functions, etc.
    #    4 = DebugAll = just dump everything you can
    #    5 = DebugTODO = this constant and all references to it need to be TODO removed from release version!!!
    DebugImportant, DebugSparse, DebugVerbose, DebugExcessive, DebugAll, DebugTODO = range(6)


    # and the static class variable for debugging level:
    globalDebugLevel = DebugTODO


    # 2011 - Mitja: scene bundle:
    SceneBundleFileExtension = "cc3s"
    SceneBundleResDirName = "Resources"
    PIFSceneFileExtension = "pifScene"
    PIFFFileExtension = "piff"
    
    # 2011 - Mitja: QSettings file application and organization names:
    #
    # the new CellDraw way - from Oct 2011 or so:
    PrefsCellDrawFormat2011 = QtCore.QSettings.IniFormat
    PrefsCellDrawScope2011 = QtCore.QSettings.UserScope
    PrefsCellDrawOrganization2011 = "Biocomplexity"
    PrefsCellDrawApplication2011 = "CellDraw_defaults"
    PrefsCellDrawApplicatioVersion2011 = "1.5.2"
    PrefsCellDrawOrganizationDomain2011 = "compucell.org"
    #
    # the new CC3D way - from Oct 2011 or so:
    PrefsCC3DFormat2011 = QtCore.QSettings.IniFormat
    PrefsCC3DScope2011 = QtCore.QSettings.UserScope
    PrefsCC3DOrganization2011 = "Biocomplexity"
    PrefsCC3DApplication2011 = "cc3d_default"
    # the old CC3D way:
    PrefsCC3DFormatOld = QtCore.QSettings.NativeFormat
    PrefsCC3DScopeOld = QtCore.QSettings.UserScope
    PrefsCC3DOrganizationOld = "Biocomplexity"
    PrefsCC3DApplicationOld = "PyQtPlayerNew"


    # --------------------------------------------------------
    def __init__(self, pParent=None):

        QtCore.QObject.__init__(self, pParent)

        self.printOut( "___ - DEBUG ----- CDConstants: __init__( pParent=="+str(pParent)+" )" , CDConstants.DebugExcessive )

    # end of __init__()
    # --------------------------------------------------------


    # --------------------------------------------------------
    # the printOut() function is a static method,
    #   so it doesn't have 'self' as first parameter:
    # --------------------------------------------------------
    def printOut(pString=None, pDebugLevel=DebugAll):
        # example from somewhere
        # print "hello %c[1mworld%c[0m" % (27, 27)

        # only print something if pDebugLevel is low enough,
        #   i.e. at least as low as the "CDConstants.globalDebugLevel",
        #   as the to have priority for printing:
        if pDebugLevel > CDConstants.globalDebugLevel:
            return

        # "color" sequences:
        lRed     = "%c[0;40;31m" % (27)
        lYellow  = "%c[0;40;33m" % (27)
        lCyan    = "%c[0;40;36m" % (27)
        lMagenta = "%c[0;40;35m" % (27)
        lBlue    = "%c[0;40;34m" % (27)
        lGray    = "%c[0;40;37m" % (27)

        # "reset" sequence:
        lEndEscSeq = "%c[0m" % (27)

        if pDebugLevel == CDConstants.DebugImportant:
            lStartEscSeq = lRed           # red
        elif pDebugLevel == CDConstants.DebugSparse:
            lStartEscSeq = lYellow        # yellow
        elif pDebugLevel == CDConstants.DebugVerbose:
            lStartEscSeq = lCyan          # cyan
        elif pDebugLevel == CDConstants.DebugExcessive:
            lStartEscSeq = lMagenta       # magenta
        elif pDebugLevel == CDConstants.DebugTODO:
            lStartEscSeq = lGray          # gray
        else:
            lStartEscSeq = lBlue          # lBlue

        print lStartEscSeq + pString + lEndEscSeq
    # end of printOut()
    # we need this line to make a method static:
    printOut = staticmethod(printOut)
    # --------------------------------------------------------









# 2012 - Mitja: advanced debugging tools,
#   work only on Posix-compliant systems so far (no MS-Windows)
#   commented out for now:
# 
# 
# # --------------------------------------------------------
# # the dumpstacks() function is a static method?
# #   so it doesn't have 'self' as first parameter?
# # --------------------------------------------------------
# import threading
# import sys
# import traceback
# # 2011 - Mitja:
# def dumpstacks(signal, frame):
#     CDConstants.printOut("|dumpstacks begin-------------------------------------------------------------|", CDConstants.DebugTODO )
#     id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
#     code = []
#     for threadId, stack in sys._current_frames().items():
#         code.append("\n# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
#         for filename, lineno, name, line in traceback.extract_stack(stack):
#             code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
#             if line:
#                 code.append("  %s" % (line.strip()))
#     CDConstants.printOut(str("\n".join(code)) , CDConstants.DebugTODO )
#     CDConstants.printOut("|dumpstacks end---------------------------------------------------------------|", CDConstants.DebugTODO )
# 
# 
# import signal
# signal.signal(signal.SIGUSR1, dumpstacks)
# CDConstants.printOut( "___ - DEBUG ----- CDConstants: signal.signal(signal.SIGUSR1, dumpstacks) executed. " , CDConstants.DebugTODO )
# 
# Local Variables:
# coding: US-ASCII
# End:
