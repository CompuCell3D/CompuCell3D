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
import os.path
import string
import re

# Start-Of-Header
name = "CC3DML  Helper Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = True
version = "0.9.0"
className = "CC3DMLHelper"
packageName = "__core__"
shortDescription = "Plugin which assists with CC3D Python scripting"
longDescription = """This plugin provides provides users with CC3D Python code snippets - making Python scripting in CC3D more convenient."""
# End-Of-Header

error = QString("")

        
    
from CC3DMLHelper import SnippetUtils

          

class CC3DMLHelper(QObject):
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
        self.snippetUtils=SnippetUtils.SnippetUtils(self)
        
        
        # self.configuration=Configuration(self.__ui.configuration.settings)
        self.initialize()   
        
        
        # those were moved to initialize fcn        
        # self.actions={}
        # self.actionGroupDict={}
        # self.actionGroupMenuDict={}
        
        # self.snippetMapper = QSignalMapper(self.__ui)
        # self.__ui.connect(self.snippetMapper,SIGNAL("mapped(const QString&)"),  self.__insertSnippet)
        
        # self.snippetDictionary=self.snippetUtils.getCodeSnippetsDict()
        # self.__initMenus()  
        # self.__initActions()        
        
        # useful regular expressions
        self.nonwhitespaceRegex=re.compile('^[\s]*[\S]+')
        self.commentRegex=re.compile('^[\s]*#')
        self.defFunRegex=re.compile('^[\s]*def')
        self.blockStatementRegex=re.compile(':[\s]*$') # block statement - : followed by whitespaces at the end of the line
        self.blockStatementWithCommentRegex=re.compile(':[\s]*[#]+[\s\S]*$') # block statement - : followed by whitespaces at the end of the line
        
        # self.__initUI()
        # self.__initToolbar()
        
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
        
        # self.snippetDictionary=self.snippetUtils.getCodeSnippetsDict()
        self.snippetDictionary=self.snippetUtils.getHandlersDict()
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

        self.cc3dmlMenu.clear()
        
        # self.cppMenuAction = self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(),self.cc3dcppMenu)
        
        self.__ui.menuBar().removeAction(self.cc3dmlMenuAction)
        
        self.initialize()
        
        return

        
  
            
    def __initMenus(self):
        self.cc3dmlMenu=QMenu("CC3DML",self.__ui.menuBar())
        #inserting CC3D Project Menu as first item of the menu bar of twedit++
        self.cc3dmlMenuAction=self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(),self.cc3dmlMenu)
        
    def getActionGroupAssignment(self, _actionName):
        groupActionListNames=["Plugins","Steppables","Potts","Metadata"]
        actionName=str(_actionName)
        for name in groupActionListNames:
            if actionName.startswith(name):
                return name
        return ""
        
    def __initActions(self):
        """
        Private method to initialize the actions.
        """
        # lists begining of action names which will be grouped 
        
        keys=self.snippetDictionary.keys()
        keys.sort()
        

        
        for key in keys:
            actionGroupName=self.getActionGroupAssignment(key)

            if actionGroupName!="":
                try:
                    actionGroupMenu=self.actionGroupMenuDict[actionGroupName] 
                    
                    actionName=key
                    
                    # removing action group anme from the name of te action
                    actionGroupNameStr=str(actionGroupName)
                    actionName =re.sub(actionGroupNameStr,'',actionName)
                    actionName.strip() #trimming leading and trailing spaces

                    
                    
                    action=actionGroupMenu.addAction(actionName)
                    self.actions[key]=action
                    self.__ui.connect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))
                    self.snippetMapper.setMapping(action, key)
                    
                except KeyError,e:


                    actionName=key
                    # removing action group anme from the name of te action
                    actionGroupNameStr=str(actionGroupName)
                    actionName =re.sub(actionGroupNameStr,'',actionName)
                    actionName.strip() #trimming leading and trailing spaces

                
                    # print 'actionGroupNameStr=',actionGroupNameStr
                    # print 'actionName=',actionName
                
                    self.actionGroupMenuDict[actionGroupName]=self.cc3dmlMenu.addMenu(actionGroupName)
                    action=self.actionGroupMenuDict[actionGroupName].addAction(actionName)
                    self.actions[key]=action
                    # action.setCheckable(True)
                    self.__ui.connect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))
                    self.snippetMapper.setMapping(action, key)
                    
                
                    # actionGroup=self.cc3dmlMenu.addAction(key)
                    # self.actionGroupDict[actionGroupName]=actionGroup
                    # action=actionGroup.addAction(key)
                    # self.actions[key]=action
                    # self.__ui.connect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))
                    # self.snippetMapper.setMapping(action, key)                    
            else:            
                
                action=self.cc3dmlMenu.addAction(key)
                self.actions[key]=action
                # action.setCheckable(True)
                self.__ui.connect(action,SIGNAL("triggered()"),self.snippetMapper,SLOT("map()"))
                self.snippetMapper.setMapping(action, key)
        
    def removeLines(self,_editor,_beginLine,_endLine):
        
        _editor.beginUndoAction() # begining of action sequence         
        _editor.setSelection(_beginLine,0,_endLine+1,0)
        _editor.removeSelectedText()
        _editor.endUndoAction() # end of action sequence    


    def getPottsData(self):
        import XMLUtils
        import xml
        from XMLUtils import dictionaryToMapStrStr as d2mss
        editor=self.__ui.getCurrentEditor()

        gpd={}
        # default values
        gpd['Dim']=[0,0,0]
        gpd['Temperature']=0
        gpd['NeighborOrder']=1
        gpd['MCS']=10000
        gpd['SimulationName']='PLEASE_PUT_SIMULATION_FILE_NAME_HERE'
        gpd['LatticeType']='Square'

        
        cc3dXML2ObjConverter = XMLUtils.Xml2Obj()       
        try:
            root_element=cc3dXML2ObjConverter.ParseString(str(editor.text()))
        except xml.parsers.expat.ExpatError,e:
            QMessageBox.critical(self.__ui,"Error Parsing CC3DML file",e.__str__())
            print 'GOT PARSING ERROR:', e
            return gpd
            
        root_element.getFirstElement('Plugin')
        print 'root_element=',root_element
        
        pottsElement=root_element.getFirstElement('Potts')
        
        if not pottsElement or pottsElement is None:            
            return gpd
        
        from XMLUtils import CC3DXMLListPy 

        #extract dimension
        dimElement=pottsElement.getFirstElement('Dimensions')
        
        if dimElement:
            
            if dimElement.findAttribute('x'):
                gpd['Dim'][0]=dimElement.getAttributeAsUInt('x')

            if dimElement.findAttribute('y'):
                gpd['Dim'][1]=dimElement.getAttributeAsUInt('y')
                
            if dimElement.findAttribute('z'):
                gpd['Dim'][2]=dimElement.getAttributeAsUInt('z')
                
        tempElement=pottsElement.getFirstElement('Temperature')                
        if tempElement:
            gpd['MembraneFluctuations']=float(tempElement.getText())
            
        nElement=pottsElement.getFirstElement('NeighborOrder')                
        if nElement:
            gpd['NeighborOrder']=float(nElement.getText())

        sElement=pottsElement.getFirstElement('Steps')                
        if sElement:
            gpd['MCS']=float(sElement.getText())
            
        return gpd    
        
    
    def getCellTypeData(self):
        import XMLUtils
        import xml
        from XMLUtils import dictionaryToMapStrStr as d2mss
        editor=self.__ui.getCurrentEditor()
        
        cc3dXML2ObjConverter = XMLUtils.Xml2Obj()       
        try:
            root_element=cc3dXML2ObjConverter.ParseString(str(editor.text()))
        except xml.parsers.expat.ExpatError,e:
            QMessageBox.critical(self.__ui,"Error Parsing CC3DML file",e.__str__())
            print 'GOT PARSING ERROR:', e
            return 
            
        root_element.getFirstElement('Plugin')
        print 'root_element=',root_element
        
        cellTypeElement=root_element.getFirstElement("Plugin",d2mss({"Name":"CellType"}))
        
        cellTypeDict={}
        if not cellTypeElement or cellTypeElement is None:
            
            return cellTypeDict
        
        from XMLUtils import CC3DXMLListPy 

        
        cellTypeElementVec=cellTypeElement.getElements("CellType")
        cellTypeElementVec=CC3DXMLListPy(cellTypeElement.getElements("CellType"))
        
        
        
        for element in cellTypeElementVec:
            typeName=''
            typeId=-1
            typeFreeze=False
            if element.findAttribute('TypeName'):
                typeName=element.getAttribute('TypeName')
            else:
                continue
                
            if element.findAttribute('TypeId'):
                typeId=element.getAttributeAsInt('TypeId')
            else:
                continue
                
            if element.findAttribute('Freeze'):
                typeFreeze=True
                
            cellTypeDict[typeId]=[typeName,typeFreeze]    
            
        return cellTypeDict    
        print 'cellTypeElementVec=',cellTypeElementVec
        
        print 'cellTypeElement=',dir(cellTypeElement)
        
        # print 'cellTypeElement=',cellTypeElement
    
        
        
    def  __insertSnippet(self,_snippetName):        
        # print "GOT REQUEST FOR SNIPPET ",_snippetName
        snippetNameStr=str(_snippetName)
        
        self.handlerDict=self.snippetUtils.getHandlersDict()
        
        text=self.snippetDictionary[str(_snippetName)]   
        
        editor=self.__ui.getCurrentEditor()
        curFileName=str(self.__ui.getCurrentDocumentName())
        
        basename,ext=os.path.splitext(curFileName)
        if ext!=".xml" and ext!=".cc3dml":
            QMessageBox.warning(self.__ui,"CC3DML files only","CC3DML code snippets work only for xml/cc3dml files")
            return
            
        # here we parse cell type plugin if found 
        import XMLUtils
        cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
        
        # root_element=cc3dXML2ObjConverter.ParseString(str(editor.text()))
        # print 'root_element=',root_element
        cellTypeData=self.getCellTypeData()
        gpd=self.getPottsData()
        
        print 'cellTypeData=',cellTypeData
        
        # self.findModuleLine(editor,_moduleType='Plugin',_moduleName=['Name','CellType'])
        # self.findModuleLine(editor,_moduleType='Steppable',_moduleName=['Type','PIFInitializer'])
        # self.findModuleLine(editor,_moduleType='Potts',_moduleName=[])
        # self.findModuleLine(editor,_moduleType='Plugin',_moduleName=['Name','CenterOfMass'])
        
        if cellTypeData is None: # could not obtain data by parsing xml file - most likely due to parsing error 
            return
            
        if not  len(cellTypeData):
            print 'self.handlerDict=',self.handlerDict
            print 'self.handlerDict[str(_snippetName)]=',self.handlerDict["Plugins CellType"]
            
            pottsBegin, pottsEnd=self.findModuleLine(editor,_moduleType='Potts',_moduleName=[])            
            
            if pottsEnd<0:
                editor.setCursorPosition(0,0)    
            else:    
                editor.setCursorPosition(pottsEnd+1,0)    
            
            self.handlerDict["Plugins CellType"](data=cellTypeData,editor=editor,generalPropertiesData=gpd)
            QMessageBox.warning(self.__ui,"Fresh Cell Type Plugin","Please check newly inserted code and call %s again"%snippetNameStr)           
            return
            #read freshly inseerted cell type plugin
        else:
            self.handlerDict[snippetNameStr](data=cellTypeData,editor=editor,generalPropertiesData=gpd)
        return

        
        # # # curLine=0
        # # # curCol=0        
        # # # if snippetNameStr=="Cell Attributes Add Dictionary To Cells" or snippetNameStr=="Cell Attributes Add List To Cells":
            # # # curLine,curCol=self.findEntryLineForCellAttributes(editor)
            # # # if curLine==-1:
                # # # QMessageBox.warning(self.__ui,"Could not find insert point","Could not find insert point for code cell attribute code. Please make sure you are editing CC3D Main Python script")
                # # # return
        # # # elif snippetNameStr.startswith("Extra Fields"):
            # # # self.includeExtraFieldsImports(editor) # this function potentially inserts new text - will have to get new cursor position after that
            # # # curLine,curCol=editor.getCursorPosition()
            
        # # # else:    
            # # # curLine,curCol=editor.getCursorPosition()
        
            
        # # # indentationLevels , indentConsistency=self.findIndentationForSnippet(editor,curLine)
        # # # print "indentationLevels=",indentationLevels," consistency=",indentConsistency
        
        # # # textLines=text.splitlines(True)
        # # # for i in range(len(textLines)):
            # # # textLines[i]=' '*editor.indentationWidth()*indentationLevels+textLines[i]
        
        # # # indentedText=''.join(textLines)
        
        # # # currentLineText=str(editor.text(curLine))
        # # # nonwhitespaceFound =re.match(self.nonwhitespaceRegex, currentLineText)
        # # # print "currentLineText=",currentLineText," nonwhitespaceFound=",nonwhitespaceFound

        
        
        # # # editor.beginUndoAction() # begining of action sequence
        
        # # # if nonwhitespaceFound: # we only add new line if the current line has someting in it other than whitespaces
            # # # editor.insertAt("\n",curLine,editor.lineLength(curLine))
            # # # curLine+=1                    
            
        # # # editor.insertAt(indentedText,curLine,0)
        # # # # editor.insertAt(text,curLine,0)
        
        # # # editor.endUndoAction() # end of action sequence        
        
        # # # #highlighting inserted text
        # # # editor.findFirst(indentedText,False,False,False,True,curLine)
        # # # lineFrom,colFrom,lineTo,colTo=editor.getSelection()
    
    # def  __visitAllCells(self):
        # nonwhitespaceRegex=re.compile('^[\s]*[\S]+')
        
        # print "__visitAllCells"
        # editor=self.__ui.getCurrentEditor()
        # curFileName=str(self.__ui.getCurrentDocumentName())
        
        # basename,ext=os.path.splitext(curFileName)
        # if ext!=".py" and ext!=".pyw":
            # QMessageBox.warning(self.__ui,"Python files only","Python code snippets work only for Python files")
            # return
        # text="""
# for cell in self.cellList:
    # print "cell.id=",cell.id
# """        
        
        # curLine,curCol=editor.getCursorPosition()
        # # print "editor.lines()=",editor.lines()," curLine=",curLine
        # # if editor.lines()==curLine+1:
            # # editor.insertAt("\n",curLine,editor.lineLength(curLine))
            # # curLine+=1
            
        # indentationLevels , indentConsistency=self.findIndentationForSnippet(editor,curLine)
        # print "indentationLevels=",indentationLevels," consistency=",indentConsistency
        
        # textLines=text.splitlines(True)
        # for i in range(len(textLines)):
            # textLines[i]=' '*editor.indentationWidth()*indentationLevels+textLines[i]
        
        # indentedText=''.join(textLines)
        
        # currentLineText=str(editor.text(curLine))
        # nonwhitespaceFound =re.match(nonwhitespaceRegex, currentLineText)
        # print "currentLineText=",currentLineText," nonwhitespaceFound=",nonwhitespaceFound
        
        
        # editor.beginUndoAction() # begining of action sequence
        
        # if nonwhitespaceFound: # we only add new line if the current line has someting in it other than whitespaces
            # editor.insertAt("\n",curLine,editor.lineLength(curLine))
            # curLine+=1                    
            
        # editor.insertAt(indentedText,curLine,0)
        # # editor.insertAt(text,curLine,0)
        
        # editor.endUndoAction() # end of action sequence        
        
        # # #highlighting inserted text
        # # editor.findFirst(text,False,False,False,True,curLine)
        # # lineFrom,colFrom,lineTo,colTo=editor.getSelection()
        # # for line in range(lineFrom,lineTo+1): 
            # # for i in range ()
                # # editor.indent(line)
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


    def findModuleLine(self,_editor,_moduleType='Plugin',_moduleName=['Name','CellType']):
        moduleLineLocatorRegex=None
        moduleClosingLineLocatorRegex=None
        moduleClosingLineLocatorSingleLineRegex=re.compile('^[\s\S]*<[\s]*'+_moduleType+'[\s\S]*/>')
        moduleClosingLineLocatorRegex=re.compile('^[\s\S]*</[\s]*'+_moduleType+'[\s]*>')
        # moduleClosingLineLocatorSingleLineRegex=re.compile('^[\s\S]*/>')
        if len(_moduleName):
            moduleLineLocatorRegex=re.compile('^[\s]*<[\s]*'+_moduleType+'[\s\S]*'+_moduleName[0]+'[\s]*=[\s]*\"'+_moduleName[1]+'\"')
            
        else:
            moduleLineLocatorRegex=re.compile('^[\s]*<[\s]*'+_moduleType+'[\s]*>')
        
        
        # getCoreSimulationObjectsRegex=re.compile('^[\s]*sim.*CompuCellSetup\.getCoreSimulationObjects')
        text=''
        beginLine=-1
        closingLine=-1
        for line in range(_editor.lines()):
            text=str(_editor.text(line))            
            moduleLineLocated =re.match(moduleLineLocatorRegex, text)  # \S - non -white space \s whitespace
            
            if moduleLineLocated: #  line with getCoreSimulationObjectsRegex
                beginLine=line
                print 'Module Line located: ',beginLine
                # return
                # break
                
        if beginLine>=0:
            # check for comment code  - #add extra attributes here
            # attribCommentRegex=re.compile('^[\s]*#[\s]*add[\s]*extra[\s]*attrib')
            for line in range(beginLine,_editor.lines()):
                text=str(_editor.text(line))
                # print 'text=',text
                
                moduleClosingLineLocated=re.match(moduleClosingLineLocatorSingleLineRegex, text)
                if moduleClosingLineLocated: #  line with getCoreSimulationObjectsRegex
                    closingLine=line
                    print 'Closing Module Line located: ',closingLine
                    print 'text=',text
                    break
                else:
                    moduleClosingLineLocated=re.match(moduleClosingLineLocatorRegex, text)
                    if moduleClosingLineLocated:
                        closingLine=line
                        print 'Closing Module Line located: ',closingLine
                        print 'text=',text
                        break
                        
                # attribCommentFound=re.match(attribCommentRegex,text)
                # if attribCommentFound:

                    # beginLine=line
                    # return beginLine,0
                    
            return beginLine,closingLine
            
        return beginLine,closingLine

            
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
            
