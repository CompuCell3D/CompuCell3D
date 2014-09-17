"""
    TO DO:
    * Keyboard events - Del
    * New Simulation wizard
    * resource properties display
    * 
"""

"""
Module used to link Twedit++ with CompuCell3D.
"""

from PyQt4.QtCore import QObject, SIGNAL, QString
from PyQt4.QtGui import QMessageBox

from PyQt4 import QtCore, QtGui

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from CC3DPythonHelper.Configuration  import Configuration

import os.path
import string
import re

# Start-Of-Header
name = "CC3D Python Helper Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = True
version = "0.9.0"
className = "CC3DPythonHelper"
packageName = "__core__"
shortDescription = "Plugin which assists with CC3D Python scripting"
longDescription = """This plugin provides provides users with CC3D Python code snippets - making Python scripting in CC3D more convenient."""
# End-Of-Header

error = QString("")

        
    

          

from PluginUtils.SnippetMenuParser import SnippetMenuParser          

class CC3DPythonHelper(QObject):
    """
    Class implementing the About plugin.
    """
    def __init__(self, ui):
        """
        Constructor
        
        @param ui reference to the user interface object (UI.UserInterface)
        """
        QObject.__init__(self, ui)
        self.__ui = ui
        
        self.__ui = ui        
        self.configuration=Configuration(self.__ui.configuration.settings)
        
        self.actions={}                
        self.initialize()   
        
        # useful regular expressions
        self.nonwhitespaceRegex=re.compile('^[\s]*[\S]+')
        self.commentRegex=re.compile('^[\s]*#')
        self.defFunRegex=re.compile('^[\s]*def')
        self.blockStatementRegex=re.compile(':[\s]*$') # block statement - : followed by whitespaces at the end of the line
        self.blockStatementWithCommentRegex=re.compile(':[\s]*[#]+[\s\S]*$') # block statement - : followed by whitespaces at the end of the line
        
#         self.lineWithCommentAtTheEndRegex=re.compile('([\s\S]*)([[#]+[\s\S]*]*)$') # line with comment at the end
        self.lineWithCommentAtTheEndRegex=re.compile('([^#]*)([\s\S]*)') # line with comment at the end first group matches anythin except '#' the remainig group catches the rest of the line
        
        self.skipCommentsFlag = False
        
    def initialize(self):
        '''  
            initializes containers used in the plugin
            '''
        self.actions={}
        self.actionGroupDict={}
        self.actionGroupMenuDict={}
        self.cppMenuAction=None
        
    def addSnippetDictionaryEntry(self,_snippetName,_snippetProperties):
        
        self.snippetDictionary[_snippetName]=_snippetProperties
        
    def getUI(self):
        return self.__ui
        
    def activate(self):
        """
        Public method to activate this plugin.
        
        @return tuple of None and activation status (boolean)
        """
        self.snippetMapper = QSignalMapper(self.__ui)
        self.__ui.connect(self.snippetMapper,SIGNAL("mapped(const QString&)"),  self.__insertSnippet)
                
        self.__initMenus()  
        self.__initActions()        

        
        return None, True
        
    def deactivate(self):
        """
        Public method to deactivate this plugin.
        """
        self.__ui.disconnect(self.snippetMapper,SIGNAL("mapped(const QString&)"),  self.__insertSnippet)
        
        for actionName, action in self.actions.iteritems():
            
            self.__ui.disconnect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))

        self.cc3dPythonMenu.clear()
        
        # self.cppMenuAction = self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(),self.cc3dcppMenu)
                
        skipCommentsInPythonSnippets=self.configuration.setSetting("SkipCommentsInPythonSnippets",self.actions["Skip Comments In Python Snippets"].isChecked())        
        self.__ui.menuBar().removeAction(self.cc3dPythonMenuAction)
        
        self.initialize()
        
            
    def __initMenus(self):
              
        self.cc3dPythonMenu=QMenu("CC3D P&ython",self.__ui.menuBar())
        #inserting CC3D Project Menu as first item of the menu bar of twedit++
        self.cc3dPythonMenuAction=self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(),self.cc3dPythonMenu)
        
        
        
        
    def __initActions(self):
        """
        Private method to initialize the actions.
        """
        # lists begining of action names which will be grouped 
        
        
        
        self.snippetDictionary={}
        psmp=SnippetMenuParser()
        snippetFilePath=os.path.abspath(os.path.join(os.path.dirname(__file__),'CC3DPythonHelper/Snippets.py.template'))
        
        psmp.readSnippetMenu(snippetFilePath)

        snippetMenuDict=psmp.getSnippetMenuDict()
        # print 'snippet menu dict = ',snippetMenuDict
        
        

        for menuName, submenuDict in iter(sorted(snippetMenuDict.iteritems())):
            print 'menuName=',menuName
            
            groupMenu=self.cc3dPythonMenu.addMenu(menuName)
            
            for subMenuName, snippetText in  iter(sorted(submenuDict.iteritems())):
                action=groupMenu.addAction(subMenuName)
                
                actionKey=menuName.strip()+' '+subMenuName.strip() # for lookup int he self.snippetDictionary 
                self.snippetDictionary[actionKey]=snippetText
                
                self.actions[actionKey]=action
                self.__ui.connect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))
                self.snippetMapper.setMapping(action, actionKey)
                

        self.actions["Skip Comments In Python Snippets"]=QtGui.QAction("Skip Comments In Python Snippets", self, shortcut="", statusTip="Skip Comments In Python Snippets")                
        self.actions["Skip Comments In Python Snippets"].setCheckable(True)              
        flag=self.configuration.setting("SkipCommentsInPythonSnippets")
        self.skipCommentsInPythonSnippets(flag)
        self.actions["Skip Comments In Python Snippets"].setChecked(flag)        
        self.connect(self.actions["Skip Comments In Python Snippets"],    SIGNAL('triggered(bool)'),  self.skipCommentsInPythonSnippets)  
        
        
        self.cc3dPythonMenu.addSeparator()
        #---------------------------------------
        self.cc3dPythonMenu.addAction(self.actions["Skip Comments In Python Snippets"])
        
    def skipCommentsInPythonSnippets(self,_flag):
        self.skipCommentsFlag=_flag
        
    def  __insertSnippet(self,_snippetName):        
        # print "GOT REQUEST FOR SNIPPET ",_snippetName
        snippetNameStr=str(_snippetName)
        
        text=self.snippetDictionary[str(_snippetName)]   
        
        editor=self.__ui.getCurrentEditor()
        curFileName=str(self.__ui.getCurrentDocumentName())
        
        basename,ext=os.path.splitext(curFileName)
        if ext!=".py" and ext!=".pyw":
            QMessageBox.warning(self.__ui,"Python files only","Python code snippets work only for Python files")
            return
        
        curLine=0
        curCol=0        
        if snippetNameStr=="Cell Attributes Add Dictionary To Cells" or snippetNameStr=="Cell Attributes Add List To Cells":
            curLine,curCol=self.findEntryLineForCellAttributes(editor)
            if curLine==-1:
                QMessageBox.warning(self.__ui,"Could not find insert point","Could not find insert point for code cell attribute code. Please make sure you are editing CC3D Main Python script")
                return
        elif snippetNameStr.startswith("Bionet Solver 3. Load SBML Model"):                
            print 'LOADING MODEL'
            from CC3DPythonHelper.sbmlloaddlg import SBMLLoadDlg
            currentPath=os.path.abspath(os.path.dirname(curFileName))                        
            print 'currentPath=',currentPath
            dlg=SBMLLoadDlg(self)
            dlg.setCurrentPath(currentPath)
            
            modelName='MODEL_NAME'
            modelNickname='MODEL_NICKNAME'            
            modelPath='PATH_TO_SBML_FILE'   
            ret=dlg.exec_()
            if ret:
                modelName=str(dlg.modelNameLE.text())
                modelNickname=str(dlg.modelNicknameLE.text())
                modelFileName=os.path.abspath(str(dlg.fileNameLE.text()))
                modelDir=os.path.abspath(os.path.dirname(modelFileName))
                
                modelPath='Simulation/'+os.path.basename(modelFileName)
                
                if modelDir != currentPath: # copy sbml file into simulation directory
                    import shutil
                    shutil.copy(modelFileName,currentPath)    

            text="""
modelName = "%s"
modelNickname  = "%s" # this is usually shorter version version of model name
modelPath="%s"
integrationStep = 0.2
bionetAPI.loadSBMLModel(modelName, modelPath,modelNickname,  integrationStep)
""" %(modelName,modelNickname,modelPath)    
                
            
            
        elif snippetNameStr.startswith("Extra Fields"):
            self.includeExtraFieldsImports(editor) # this function potentially inserts new text - will have to get new cursor position after that
            curLine,curCol=editor.getCursorPosition()
            
        else:    
            curLine,curCol=editor.getCursorPosition()
        
            
        indentationLevels , indentConsistency=self.findIndentationForSnippet(editor,curLine)
        print "indentationLevels=",indentationLevels," consistency=",indentConsistency
        
        
        
        textLines=text.splitlines(True)
        for i in range(len(textLines)):
            textLines[i]=' '*editor.indentationWidth()*indentationLevels+textLines[i]
            try: # since we dont want twedit to crash when removing coments the code catches all exceptions
                if self.skipCommentsFlag:
                    commentFound=re.match(self.commentRegex,textLines[i])
                    if commentFound: #if it is 'regular' line we check if this line is begining of a block statement
                        textLines[i]=''
                    else:
                        # print 'could not find whole line comment will search eol comment'
                        match=re.match(self.lineWithCommentAtTheEndRegex,textLines[i])
                        if match:
                            matchGroups=match.groups()
                            if matchGroups[1]!='':
#                                 print 'before textLines[i]=',textLines[i]
                                textLines[i] = self.lineWithCommentAtTheEndRegex .sub (r"\1\n",textLines[i])
#                                 print 'after textLines[i]=',textLines[i]                        
            
            except:
                print 'ERROR WHEN REMOVING COMMENTS IN ', textLines[i]
                
        
        indentedText=''.join(textLines)
        currentLineText=str(editor.text(curLine))
        nonwhitespaceFound =re.match(self.nonwhitespaceRegex, currentLineText)
        print "currentLineText=",currentLineText," nonwhitespaceFound=",nonwhitespaceFound

        
        
        editor.beginUndoAction() # begining of action sequence
        
        if nonwhitespaceFound: # we only add new line if the current line has someting in it other than whitespaces
            editor.insertAt("\n",curLine,editor.lineLength(curLine))
            curLine+=1                    
            
        editor.insertAt(indentedText,curLine,0)
        # editor.insertAt(text,curLine,0)
        
        editor.endUndoAction() # end of action sequence        
        
        #highlighting inserted text
        editor.findFirst(indentedText,False,False,False,True,curLine)
        lineFrom,colFrom,lineTo,colTo=editor.getSelection()
    

    def includeExtraFieldsImports(self,_editor):
        playerFromImportRegex=re.compile('^[\s]*from[\s]*PlayerPython[\s]*import[\s]*\*')
        compuCellSetupImportRegex=re.compile('^[\s]*import[\s]*CompuCellSetup')
        curLine,curCol=_editor.getCursorPosition()
        foundPlayerImports=None
        foundCompuCellSetupImport=None
        for line in range(curLine,-1,-1):
            text=str(_editor.text(line))
            foundPlayerImports=re.match(playerFromImportRegex, text)
            if foundPlayerImports:
                break
            
        for line in range(curLine,-1,-1):
            text=str(_editor.text(line))
            foundCompuCellSetupImport=re.match(compuCellSetupImportRegex, text)
            if foundCompuCellSetupImport:
                break
                
        if not foundCompuCellSetupImport:
            _editor.insertAt("import CompuCellSetup\n",0,0)
            
        if not foundPlayerImports:
            _editor.insertAt("from PlayerPython import * \n",0,0)
        
    def findEntryLineForCellAttributes(self,_editor):
        getCoreSimulationObjectsRegex=re.compile('^[\s]*sim.*CompuCellSetup\.getCoreSimulationObjects')
        text=''
        foundLine=-1
        for line in range(_editor.lines()):
            text=str(_editor.text(line))
            
            getCoreSimulationObjectsRegexFound =re.match(getCoreSimulationObjectsRegex, text)  # \S - non -white space \swhitespace
            
            if getCoreSimulationObjectsRegexFound: #  line with getCoreSimulationObjectsRegex
                foundLine=line
                break
                
        if foundLine>=0:
            # check for comment code  - #add extra attributes here
            attribCommentRegex=re.compile('^[\s]*#[\s]*add[\s]*extra[\s]*attrib')
            for line in range(foundLine,_editor.lines()):
                text=str(_editor.text(line))
                attribCommentFound=re.match(attribCommentRegex,text)
                if attribCommentFound:

                    foundLine=line
                    return foundLine,0
                    
            return foundLine,0
        
        return -1,-1       
        
        
    def findIndentationForSnippet(self,_editor,_line):
        # nonwhitespaceRegex=re.compile('^[\s]*[\S]+')
        # commentRegex=re.compile('^[\s]*#')
        # defFunRegex=re.compile('^[\s]*def')
        # blockStatementRegex=re.compile(':[\s]*$') # block statement - : followed by whitespaces at the end of the line
        # blockStatementWithCommentRegex=re.compile(':[\s]*[#]+[\s\S]*$') # block statement - : followed by whitespaces at the end of the line
        
        # ':[\s]*$|:[\s]*[#]+[\s\S*]$'
        
        # ':[\s]*[\#+[\s\S*]$'
        
        
        # ':[\s]*[#]+[\s\S]*' -  works
        
        text=''
        for line in range(_line,-1,-1):
            text=str(_editor.text(line))
            
            nonwhitespaceFound =re.match(self.nonwhitespaceRegex, text)  # \S - non -white space \swhitespace
            
            if nonwhitespaceFound: # once we have line with non-white spaces we check if this is non comment line
                commentFound=re.match(self.commentRegex,text)
                if not commentFound: #if it is 'regular' line we check if this line is begining of a block statement
                
                    blockStatementFound=re.search(self.blockStatementRegex,text)
                    blockStatementWithCommentFound=re.search(self.blockStatementWithCommentRegex,text)
                    # print "blockStatementFound=",blockStatementFound," blockStatementWithCommentFound=",blockStatementWithCommentFound
                    
                    
                    if blockStatementFound or blockStatementWithCommentFound: # we insert code snippet increasing indentation after begining of block statement
                        # print "_editor.indentationWidth=",_editor.indentationWidth
                        
                        indentationLevels=(_editor.indentation(line)+_editor.indentationWidth())/_editor.indentationWidth()
                        indentationLevelConsistency=not (_editor.indentation(line)+_editor.indentationWidth())%_editor.indentationWidth() # if this is non-zero indentations in the code are inconsistent 
                        if not indentationLevelConsistency:
                            QMessageBox.warning(self.__ui,"Possible indentation problems","Please position code snippet manually using TAB (indent) Shift+Tab (Unindent)")
                            return 0,indentationLevelConsistency
                            
                        return indentationLevels, indentationLevelConsistency
                        
                    else:# we use indentation of the previous line
                        indentationLevels=(_editor.indentation(line))/_editor.indentationWidth()
                        indentationLevelConsistency=not (_editor.indentation(line))%_editor.indentationWidth() # if this is non-zero indentations in the code are inconsistent 
                        if not indentationLevelConsistency:
                            QMessageBox.warning(self.__ui,"Possible indentation problems","Please position code snippet manually using TAB (indent) Shift+Tab (Unindent)")
                            return 0,indentationLevelConsistency
                            
                        return indentationLevels, indentationLevelConsistency
                                            
        
        return 0,0    
            
