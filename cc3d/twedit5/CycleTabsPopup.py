from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, setDebugging

MAC = "qt_mac_set_native_menubar" in dir()



class CycleTabsPopup(QLabel):

    # signals

    def __init__(self, parent=None):

        super(CycleTabsPopup, self).__init__(parent)

        self.editorWindow = parent

        self.setText("THIS IS CYCLE WINDOWS TAB")

        # self.__text = unicode(text)

        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)

        # palette = QPalette()

        palette = self.palette()

        palette.setColor(self.backgroundRole(), QColor('#F5F6CE'))

        self.setPalette(palette)

        font = self.font()

        font.setPointSize(9)

        # on mac base size font has to be gibber than on linux or windows - otherwise letters are too small

        if sys.platform.startswith('darwin'):
            font.setPointSize(11)

        self.setFont(font)

        # self.setBackgroundRole(QtPalette.Light)

        # self.setWindowFlags(Qt.FramelessWindowHint)

        self.setTextFormat(Qt.RichText)

        # self.setupUi(self)        

        self.ctrlTabShortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)

        self.CtrlKeyEquivalent = Qt.Key_Control

        if sys.platform.startswith("darwin"):
            self.ctrlTabShortcut = QShortcut(QKeySequence("Alt+Tab"), self)

            self.CtrlKeyEquivalent = Qt.Key_Alt

        self.ctrlTabShortcut.activated.connect(self.cycleTabs)

        self.highlightedItem = ''

        self.openFileNames = None

        self.cycleTabFilesList = None

        dbgMsg("self.editorWindow.pos()=", str(self.editorWindow.pos()) + "\n\n\n\n")

    def initializeContent(self, _cycleTabFilesList):

        labelContent = ''

        i = 0

        self.cycleTabFilesList = _cycleTabFilesList

        self.openFileNames = _cycleTabFilesList.tabList

        # special case for just one document

        if len(self.openFileNames) == 1:
            labelContent += "<b>" + self.openFileNames[0][0] + "</b><br>"

            self.highlightedItem = self.openFileNames[0]

            self.setText(labelContent)

            return

        for fileName in self.openFileNames:

            #             dbgMsg("fileName=",fileName)

            if i == 1:

                labelContent += "<b>" + fileName[0] + "</b><br>"

                self.highlightedItem = fileName

                # dbgMsg("label content=",labelContent)



            else:

                labelContent += fileName[0] + "<br>"

                # if firstName=='':

            i += 1

        self.setText(labelContent)

        # dbgMsg("dir(self)=",dir(self))

        # # self.setSelection(0,len(firstName))

    def cycleTabs(self):

        dbgMsg("QLabel cycleTabs")

        # highlightedItem is a list - [fileName,editor]

        highlightTextFlag = False

        if self.highlightedItem == self.openFileNames[-1]:  # we start highlighting cycle from the begining

            highlightTextFlag = True

        highlightingChanged = False

        if self.openFileNames:

            labelContent = ''

            for fileName in self.openFileNames:

                if highlightTextFlag:

                    dbgMsg("GOT HIGHLIGHT TEXT FLAG")

                    labelContent += "<b>" + fileName[0] + "</b><br>"

                    highlightTextFlag = False

                    self.highlightedItem = fileName

                    highlightingChanged = True

                    # dbgMsg("label content=",labelContent)



                else:

                    labelContent += fileName[0] + "<br>"

                    # if firstName=='':

                if self.highlightedItem[0] == fileName[0] and not highlightingChanged:
                    highlightTextFlag = True

            self.setText(labelContent)

    def keyPressEvent(self, event):

        if event.key() == self.CtrlKeyEquivalent:
            dbgMsg("CTRL key pressed")

            self.ctrlPressed = True

            # if event.modifiers()==Qt.CtrlModifier:

            # dbgMsg("CTRL key pressed")

            self.ctrlPressed = True

            # if event.key():

            # dbgMsg("TAB PRESSED ",event.key()        )

    def keyReleaseEvent(self, event):

        if event.key() == self.CtrlKeyEquivalent:
            dbgMsg("CTRL RELEASED in QTextEdit")

            self.ctrlPressed = False

            self.close()

            # make lastly selected tab current

            self.cycleTabFilesList.makeItemCurrent(self.highlightedItem)

            self.openFileNames = None

            self.cycleTabFilesList = None

            # if self.cycleTabsFlag: # release cycleTabs flag and switch to new tab

            # self.cycleTabsFlag=False

            # self.tabFileNamesDict=None

            # self.cycleTabsPopup.close()

            # self.cycleTabsPopup=None


class CycleTabFileList:

    def __init__(self, _editorWindow):

        self.editorWindow = _editorWindow

        self.tabList = []

        self.tabDict = {}  # this is auxiliary container used for faster searches

        # self.initializeTabFileList()

    def initializeTabFileList(self):

        if not len(self.tabList):  # initialize from scratch if the list is empty - this is done on startup

            dbgMsg("INITIALIZE FILE TAB LIST\n\n\n\n")

            openFileDict = {}

            for tabWidget in self.editorWindow.panels:

                for i in range(tabWidget.count()):

                    editor = tabWidget.widget(i)

                    if editor == tabWidget.currentWidget():

                        if self.editorWindow.getEditorFileName(editor) != "":

                            self.tabList.insert(0, [self.editorWindow.getEditorFileName(editor), editor])

                        else:

                            documentName = tabWidget.tabText(tabWidget.indexOf(editor))

                            self.tabList.insert(0, [documentName, editor])

                        self.tabDict[editor] = self.tabList[0][0]

                    else:

                        if self.editorWindow.getEditorFileName(editor) != "":

                            self.tabList.append([self.editorWindow.getEditorFileName(editor), editor])

                        else:

                            documentName = tabWidget.tabText(tabWidget.indexOf(editor))

                            self.tabList.append([documentName, editor])

                            # self.tabList.append([self.editorWindow.fileDict[editor][0],editor])

                        self.tabDict[editor] = self.tabList[-1][0]

                        # storing items in the tabDict. tab dict will be used to compare if new items have been added to self.editorWindow.fileDict

                        # self.tabDict[editor]=self.editorWindow.fileDict[editor][0]

                        # self.tabDict[editor]=self.tabList[-1]

        else:

            self.refresh()

    def refresh(self):

        dbgMsg("REFRESH FILE TAB LIST\n\n\n\n")

        for tabWidget in self.editorWindow.panels:

            for i in range(tabWidget.count()):

                editor = tabWidget.widget(i)

                try:

                    self.tabDict[editor]

                    # checking if file name has not changed

                    documentName = self.editorWindow.getEditorFileName(editor)

                    if documentName == "":
                        documentName = tabWidget.tabText(tabWidget.indexOf(editor))

                    if documentName != self.tabDict[editor]:

                        dbgMsg("fileName has changed in the open tab")

                        # linear search for item entry with matchin editor entry

                        for i in range(len(self.tabList)):

                            if self.tabList[i][1] == editor:
                                self.tabList[i][0] = documentName

                                self.tabDict[editor] = self.tabList[i][0]

                                break

                                # sys.exit()

                except KeyError as e:

                    # found new editor window

                    dbgMsg("# found new editor window")

                    if self.editorWindow.getEditorFileName(editor) != "":  # if the name of the item is non empty

                        self.insertNewItem([self.editorWindow.getEditorFileName(editor), editor])

                        # self.tabDict[editor]=self.tabList.insert[1][0] # new item was inserted at position 1

                        # self.tabDict[editor]=self.editorWindow.fileDict[editor][0]

                    else:  # otherwise get tab text

                        documentName = tabWidget.tabText(tabWidget.indexOf(editor))

                        self.insertNewItem([documentName, editor])

                        # self.tabDict[editor]=self.tabList.insert[1][0]# new item was inserted at position 1

                        # self.tabDict[editor]=documentName 

        # check if a tab has been deleted

        editorItemsToBeDeleted = []

        for editor in list(self.tabDict.keys()):

            # print '\t\t editor=',editor,' FNAME=',self.editorWindow.getEditorFileName(editor),' exists=',self.editorWindow.checkIfEditorExists(editor)

            if not self.editorWindow.checkIfEditorExists(editor):
                # this editor has been deleted from tab widget

                editorItemsToBeDeleted.append(editor)

                # try:

                # self.editorWindow.fileDict[editor]

                # except KeyError,e:

                # # this editor has been deleted from tab widget

                # editorItemsToBeDeleted.append(editor) 

        # print '*******editorItemsToBeDeleted=',editorItemsToBeDeleted

        for editorItem in editorItemsToBeDeleted:

            del self.tabDict[editorItem]

            # linear search for editor entry in the tabList

            for i in range(len(self.tabList)):

                if self.tabList[i][1] == editorItem:
                    del self.tabList[i]

                    break

        activeTab = self.editorWindow.getActivePanel()

        # make sure current item is listed at the top

        currentEditor = self.editorWindow.getActiveEditor()

        # print 'currentEditor=',currentEditor,' fileName=',self.editorWindow.getEditorFileName(currentEditor)

        currentItem = [self.tabDict[currentEditor], currentEditor]

        self.makeItemCurrent(currentItem)

    def insertNewItem(self, _item):

        if (len(self.tabList) >= 1):
            self.tabList.insert(1, _item)

            self.tabDict[_item[1]] = _item[0]

    def makeItemCurrent(self, _item):

        _currentEditor = _item[1]

        _currentEditor.panel.setCurrentWidget(_currentEditor)

        _currentEditor.setFocus(Qt.MouseFocusReason)

        # liear search for item in the tabList

        index = 0

        for i in range(len(self.tabList)):

            if self.tabList[i] == _item:
                index = i

                break

        # putting current item at the top of the cycle list        

        del self.tabList[index]

        self.tabList.insert(0, _item)
