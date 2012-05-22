#!/usr/bin/env python
#
# CDSceneBundle - handle saving/reading bundle-style scene files for CellDraw - Mitja 2011-2012
#
# ------------------------------------------------------------

import sys     # for handling platform-specific system queries
import os      # for path and split functions

from PyQt4 import QtXml   # to read/write simple XML files

from PyQt4 import QtGui # from PyQt4.QtGui import *
from PyQt4 import QtCore # from PyQt4.QtCore import *

# from PyQt4 import QtCore, QtGui


# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# 2011 - Mitja: import classes generating bound signals
#        (from objects i.e. class instances, not from classes)
#        that change global preferences:
#
# to refer to the signalPIFFGenerationModeHasChanged signal
#   defined in CDControlPIFFGenerationMode we need the specific instance:
# .... no need to do that?
# from cdTypesEditor import CDTypesEditor



# --------------------------------------------------------------------------
# 2010- Mitja: this is the main class for CellDraw bundle-style scene files
# --------------------------------------------------------------------------
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("cdPreferencesChangedSignal()"))
#
class CDSceneBundle(QtGui.QDialog):

    # ---------------------------------------------------------
    def __init__(self, pParent=None):
    # ---------------------------------------------------------
        # it is compulsory to call the parent's __init__ class right away:
        super(CDSceneBundle, self).__init__(pParent)

        self.hide()
        self.theMainWindow = pParent

        #  __init__ (1) - set up the default file extension & resource directory:
        # ---------------------------------------------------------
        #
        # self.SceneBundleMainFileExtension = CDConstants.SceneBundleFileExtension
        # self.SceneBundleResourceDirectoryName = CDConstants.SceneBundleResDirName

        #  __init__ (2) - set up the default scene bundle paths:
        # ---------------------------------------------------------
        #
        self.thePathToSceneBundleDir = ""
#         self.thePathToPIFScene = os.path.join( \
#             self.thePathToSceneBundleDir, CDConstants.SceneBundleResDirName)
        self.theSceneBundleMainFileName = ""        
        self.thePIFSceneFileName = ""
        self.thePIFFFileName = ""
        self.theResDir = ""

        #  __init__ (3) - create a QTreeWidget to show the contents of a scene bundle file:
        # ---------------------------------------------------------
        #
        self.treeWidget = QtGui.QTreeWidget()
        self.treeWidget.header().setResizeMode(QtGui.QHeaderView.Stretch)
        self.treeWidget.setHeaderLabels(("File Type", "Location"))
        
        self.setWindowTitle("CellDraw Scene Bundle Contents")
        self.setMinimumSize(480, 320)

        self.mainPanelLayout = QtGui.QVBoxLayout()
        self.mainPanelLayout.setMargin(2)
        self.mainPanelLayout.setSpacing(4)
        self.mainPanelLayout.setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setLayout(self.mainPanelLayout)

        miDialogsWindowFlags = QtCore.Qt.WindowFlags()
        miDialogsWindowFlags = QtCore.Qt.Window
        # this strange combination of WindowFlags makes all 3 buttons inactive (close/min/max) :
        miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
        miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint

        self.setWindowFlags(miDialogsWindowFlags)
        self.setAttribute( QtCore.Qt.ApplicationModal | \
            QtCore.Qt.WA_MacNoClickThrough | \
            QtCore.Qt.WA_MacVariableSize | \
            QtCore.Qt.WA_MacOpaqueSizeGrip )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.mainPanelLayout.addWidget(self.treeWidget)

        self.buttonOK = QtGui.QPushButton("OK")
        self.buttonCancel = QtGui.QPushButton("Cancel")
        self.buttonBox = QtGui.QDialogButtonBox()
        self.buttonBox.addButton(self.buttonOK, QtGui.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(self.buttonCancel, QtGui.QDialogButtonBox.RejectRole)
        # connects signals from buttons to "slot" methods:
        self.buttonBox.accepted.connect(self.handleButtonOK)
        self.buttonBox.rejected.connect(self.handleButtonCancel)
        self.mainPanelLayout.addWidget(self.buttonBox)
        self.close()
        
        CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: __init__(): done" , CDConstants.DebugExcessive )

    # end of __init__()
    # --------------------------------------------------------



    # ---------------------------------------------------------
    # openSceneBundleFile() sets the base path for bundle-style scene files:
    # ---------------------------------------------------------    
    def openSceneBundleFile(self, pPathToSceneBundleMainFile=None):
        if (pPathToSceneBundleMainFile == None) or (not (os.path.isfile(pPathToSceneBundleMainFile))):
            CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: openSceneBundleFile() can not find "+str(pPathToSceneBundleMainFile)+" path." , CDConstants.DebugSparse )
            return
        else:

            self.treeWidget.clear()

            lXMLHandler = CC3SHandler(self)
            lXMLReader = QtXml.QXmlSimpleReader()
            lXMLReader.setContentHandler(lXMLHandler)
            lXMLReader.setErrorHandler(lXMLHandler)

            lFile = QtCore.QFile(pPathToSceneBundleMainFile)
            if not lFile.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text):
                QtGui.QMessageBox.warning(self, "CC3D Scene Bundle",
                        "Cannot read file %s:\n%s." % (pPathToSceneBundleMainFile, lFile.errorString()))
                return False
    
            xmlInputSource = QtXml.QXmlInputSource(lFile)
            if lXMLReader.parse(xmlInputSource):
                
                self.thePathToSceneBundleDir,self.theSceneBundleMainFileName = os.path.split(str(pPathToSceneBundleMainFile))

                CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: openSceneBundleFile() parsed file "+ \
                    str(pPathToSceneBundleMainFile) + " self.thePathToSceneBundleDir==" + \
                    self.thePathToSceneBundleDir + " self.theSceneBundleMainFileName==" + \
                    self.theSceneBundleMainFileName , CDConstants.DebugAll )
                
                # parsing went fine, show the user what we got:
                self.show()
                self.treeWidget.clearFocus()

                return True
            else:
                QtGui.QMessageBox.warning(self, "CC3D Scene Bundle",
                        "Cannot parse file %s" % pPathToSceneBundleMainFile)
                return False


    # ------------------------------------------------------------------
    # handle the OK Button:
    # ------------------------------------------------------------------
    def handleButtonOK(self):
        CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: handleButtonOK(): done" , CDConstants.DebugExcessive )
        self.close()

    # ------------------------------------------------------------------
    # handle the Cancel Button:
    # ------------------------------------------------------------------
    def handleButtonCancel(self):
        CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: handleButtonCancel(): done" , CDConstants.DebugExcessive )
        self.close()

    # ---------------------------------------------------------
    # the reject() function is called when the user presses the <ESC> keyboard key.
    #   We override the default and implicitly include the equivalent action of
    #   <esc> being the same as clicking the "Cancel" button, as in well-respected GUI paradigms:
    # ---------------------------------------------------------
    def reject(self):
        self.buttonCancel.click()
        super(CDSceneBundle, self).reject()
        CDConstants.printOut("___ - DEBUG ----- CDSceneBundle: reject(): done", CDConstants.DebugExcessive)


    # ---------------------------------------------------------
    # the reject() function is called when the user presses the <ESC> keyboard key.
    #   We override the default and implicitly include the equivalent action of
    #   <esc> being the same as clicking the "Cancel" button, as in well-respected GUI paradigms:
    # ---------------------------------------------------------
    def accept(self):
        self.buttonOK.click()
        super(CDSceneBundle, self).accept()
        CDConstants.printOut("___ - DEBUG ----- CDSceneBundle: accept(): done", CDConstants.DebugExcessive)





    def getSceneFileName(self):
        lTmpPathOnly = os.path.join(self.thePathToSceneBundleDir, self.theResDir)
        lTmpPathAndFileName = os.path.join(lTmpPathOnly, self.thePIFSceneFileName)
        CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: getSceneFileName() lTmpPathAndFileName=="+lTmpPathAndFileName , CDConstants.DebugVerbose )
        return lTmpPathAndFileName


    def getPIFFFileName(self):
        lTmpPathOnly = os.path.join(self.thePathToSceneBundleDir, self.theResDir)
        lTmpPathAndFileName = os.path.join(lTmpPathOnly, self.thePIFFFileName)
        CDConstants.printOut( "___ - DEBUG ----- CDSceneBundle: getPIFFFileName() lTmpPathAndFileName=="+lTmpPathAndFileName , CDConstants.DebugVerbose )
        return lTmpPathAndFileName



# --------------------------------------------------------------------------
# the CC3SHandler class implements a simple XML parser
#   as from the PyQt4 example "examples/xml/saxbookmarks/saxbookmarks.py"
# it requires QtXml...
# --------------------------------------------------------------------------

class CC3SHandler(QtXml.QXmlDefaultHandler):

    # ---------------------------------------------------------
    def __init__(self, pParentClass):
        super(CC3SHandler, self).__init__()

        self.theParentWindget = pParentClass
        self.treeWidget = self.theParentWindget.treeWidget
        self.folderIcon = QtGui.QIcon()
        self.bookmarkIcon = QtGui.QIcon()
        self.currentText = ''
        self.errorStr = ''

        self.item = None
        self.metCc3sTag = False

        # set standard icons for folders and files:
        style = self.treeWidget.style()
        self.folderIcon.addPixmap(style.standardPixmap(QtGui.QStyle.SP_DirClosedIcon),
                QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.folderIcon.addPixmap(style.standardPixmap(QtGui.QStyle.SP_DirOpenIcon),
                QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.bookmarkIcon.addPixmap(style.standardPixmap(QtGui.QStyle.SP_FileIcon))

    # ---------------------------------------------------------
    def startElement(self, namespaceURI, localName, qName, attributes):
    
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: startElement( namespaceURI=="+str(namespaceURI)+", localName=="+str(localName)+", qName=="+str(qName)+", attributes=="+str(attributes)+" )" , CDConstants.DebugExcessive )

        if not self.metCc3sTag and qName != CDConstants.SceneBundleFileExtension:
            self.errorStr = "The file is not a cc3s file."
            return False

        # check that the XML is of the type of document we want:
        if qName == CDConstants.SceneBundleFileExtension:
            version = attributes.value('version')
            if version and version != '1.0':
                self.errorStr = "The file is not a cc3s version 1.0 file."
                return False
            self.metCc3sTag = True
        elif qName == 'ResDir':
            self.item = self.createChildItem(qName)
            self.item.setFlags(self.item.flags() | QtCore.Qt.ItemIsEditable)
            self.item.setIcon(0, self.folderIcon)
            self.item.setText(0, "DirName")
            self.item.setText(1, attributes.value("DirName"))
            # assign the resources directory name:
            self.theParentWindget.theResDir = str(attributes.value("DirName"))
            self.treeWidget.setItemExpanded(self.item, True)
        elif qName == 'PIFScene':
            self.item = self.createChildItem(qName)
            self.item.setFlags(self.item.flags() | QtCore.Qt.ItemIsEditable)
            self.item.setIcon(0, self.bookmarkIcon)
            self.item.setText(0, attributes.value("Type"))
            self.item.setText(1, "Unknown Filename")
        elif qName == 'PIFF':
            self.item = self.createChildItem(qName)
            self.item.setFlags(self.item.flags() | QtCore.Qt.ItemIsEditable)
            self.item.setIcon(0, self.bookmarkIcon)
            self.item.setText(0, attributes.value("Type"))
            self.item.setText(1, "Unknown Filename")
#         elif qName == 'separator':
#             self.item = self.createChildItem(qName)
#             self.item.setFlags(self.item.flags() & ~QtCore.Qt.ItemIsSelectable)
#             self.item.setText(0, 30 * "\xb7")

        self.currentText = ''
        return True

    # ---------------------------------------------------------
    def endElement(self, namespaceURI, localName, qName):
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: endElement( namespaceURI=="+str(namespaceURI)+", localName=="+str(localName)+", qName=="+str(qName)+" ), self.currentText=="+str(self.currentText) , CDConstants.DebugExcessive )

#         if qName == 'title':
#             if self.item:
#                 self.item.setText(0, self.currentText)
        if qName == "PIFScene":
            if self.item:
                self.item.setText(1, str(self.currentText))
            # we got the filename for the Scene file:
            self.theParentWindget.thePIFSceneFileName = str(self.currentText)
            self.item = self.item.parent()
        elif qName == "PIFF":
            if self.item:
                self.item.setText(1, str(self.currentText))
            # we got the filename for the PIFF file:
            self.theParentWindget.thePIFFFileName = str(self.currentText)
            self.item = self.item.parent()
        elif qName == "ResDir":
            self.item = self.item.parent()

        return True

    # ---------------------------------------------------------
    def characters(self, txt):
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: characters( txt=="+str(txt)+" )" , CDConstants.DebugExcessive )

        self.currentText += txt
        return True

    # ---------------------------------------------------------
    def fatalError(self, exception):
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: fatalError( exception=="+str(exception)+" )" , CDConstants.DebugExcessive )

        QtGui.QMessageBox.information(self.treeWidget.window(),
                "SAX Bookmarks",
                "Parse error at line %d, column %d:\n%s" % (exception.lineNumber(), exception.columnNumber(), exception.message()))
        return False

    # ---------------------------------------------------------
    def errorString(self):
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: errorString()" , CDConstants.DebugExcessive )
        return self.errorStr

    # ---------------------------------------------------------
    # can we use QTreeWidgetItem just to store data?
    # ---------------------------------------------------------
    def createChildItem(self, tagName):
        CDConstants.printOut( "___ - DEBUG ----- CDConstants: createChildItem()" , CDConstants.DebugExcessive )
        if self.item:
            childItem = QtGui.QTreeWidgetItem(self.item)
        else:
            childItem = QtGui.QTreeWidgetItem(self.treeWidget)

        childItem.setData(0, QtCore.Qt.UserRole, tagName)
        return childItem

# Local Variables:
# coding: US-ASCII
# End:
