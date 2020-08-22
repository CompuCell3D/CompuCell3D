from cc3d.twedit5.twedit.utils.global_imports import *
import xml
from cc3d.core import XMLUtils
from cc3d.core.XMLUtils import CC3DXMLListPy
from .adhesionflexdlg import AdhesionFlexDlg
from .celltypedlg import CellTypeDlg
from .pottsdlg import PottsDlg
import functools
import re
from cc3d.twedit5.Plugins.CC3DMLGenerator.CC3DMLGeneratorBase import CC3DMLGeneratorBase

(MB_CANCEL, MB_OK, MB_REPLACE, MB_COMMENTOUTANDADD, MB_INSERTFRESH) = list(range(0, 5))

# this dictionary holds names of modules which are listed as arguents of decorators

global dictOfModules

dictOfModules = {}


class MultiDimDict(dict):

    def __init__(self, default=None):
        self.default = default

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.default()

        return dict.__getitem__(self, key)


class SnippetDecorator(object):

    def __init__(self, _moduleType, _moduleName, _menuName=None, _submenuName=None):

        global dictOfModules

        self.moduleType = _moduleType

        self.moduleName = _moduleName

        try:

            dictOfModules[_moduleName[1][0]] = _moduleType

        except IndexError as e:

            if _menuName and _submenuName:

                dictOfModules[_submenuName] = _moduleType

                return
            else:
                print('Index error in  SnippetDecorator: ', e.__str__())

    def __call__(self, _decoratedFn):

        @functools.wraps(_decoratedFn)
        def decorator(*args, **kwds):

            obj = args[0]

            cellTypeData = kwds['data']

            editor = kwds['editor']

            moduleAttributeLabel = self.moduleName[0]

            taskFlag, moduleAlias = obj.warnIfModuleExists(editor, self.moduleType, self.moduleName)

            if taskFlag == MB_CANCEL:
                return

            kwds['taskFlag'] = taskFlag

            try:

                # root_element,moduleBegin,moduleEnd= self.extractCurrentModule(editor,
                # cc3dXML2ObjConverter,_moduleType='Plugin',_moduleName=['Name','Contact'])

                # print 'moduleAttributeLabel,moduleAlias=',(moduleAttributeLabel,moduleAlias)

                if moduleAttributeLabel == '':  # modules with no attribute in the openeing element e.g. <Potts>

                    cc3dXML2ObjConverter, moduleBegin, moduleEnd = obj.extractCurrentModule(editor, self.moduleType, [])

                else:  # modules with attribute in the openeing element e.g. <Plugin Name='CellType'>

                    cc3dXML2ObjConverter, moduleBegin, moduleEnd = obj.extractCurrentModule(editor, self.moduleType,
                                                                                            [moduleAttributeLabel,
                                                                                             moduleAlias])





            except UserWarning as e:

                print('Could not extract Contact Plugin:', e.__str__())

                return

            kwds['cc3dXML2ObjConverter'] = cc3dXML2ObjConverter

            if cc3dXML2ObjConverter is not None:

                root_element = cc3dXML2ObjConverter.root

                kwds['root_element'] = root_element

            else:

                kwds['root_element'] = None

            newSnippet = _decoratedFn(*args, **kwds)

            # Add optional comment removal HERE
            if 'hiding_comments' in kwds.keys() and kwds['hiding_comments']:
                re_str = '<!--(.*)-->'
                newSnippet = ''.join([x for x in newSnippet.splitlines(keepends=True) if re.search(re_str, x) is None])

            obj.handleNewSnippet(editor, newSnippet, taskFlag, moduleBegin, moduleEnd)

        return decorator


class SnippetUtils(object):

    # def __init__(self):

    def __init__(self, _cc3dmlHelper=None):

        self.generator = CC3DMLGeneratorBase()

        self.cc3dmlHelper = _cc3dmlHelper

        self.__ui = self.cc3dmlHelper.getUI()

        self.snippetDict = {}

        self.handlerDict = {}

        self.initHandlers()

    def initHandlers(self):

        global dictOfModules

        # print 'dictOfModules=',dictOfModules

        for moduleName, moduleType in dictOfModules.items():

            # print  'moduleName=',moduleName,' moduleType=',moduleType

            if moduleType == 'Plugin':

                self.handlerDict['Plugins ' + moduleName] = getattr(self, 'handle' + moduleName)

            elif moduleType == 'Steppable':

                self.handlerDict['Steppables ' + moduleName] = getattr(self, 'handle' + moduleName)

            # elif moduleType=='Potts':

            # self.handlerDict['Potts '+'Simulation Configuration']=getattr(self,'handle'+'Potts')

            elif moduleType in ['Metadata', 'Potts']:

                self.handlerDict[moduleType + moduleName] = getattr(self,
                                                                    'handle' + moduleType + moduleName.replace(' ', ''))

    def getHandlersDict(self):

        return self.handlerDict

    def getCodeSnippetsDict(self):

        return self.snippetDict

    def extractSnippet(self, _editor, _beginLine, _endLine):

        snippet = ''

        for lineNumber in range(_beginLine,
                                _endLine + 1):  # +1 to include last line - range by default does not include last value from the interval

            snippet += _editor.text(lineNumber)

        print('snippet=', snippet)

        return str(snippet)

    def getMatrix(self):

        contactMatrix = MultiDimDict(dict)

        return contactMatrix

    def extractCurrentModule(self, _editor, _moduleType,
                             _moduleName):  # editor,_moduleType='Plugin',_moduleName=['Name','Contact']

        moduleName = []

        try:

            if _moduleName[0] == '' and _moduleName[1] == '':

                moduleName = []

            else:

                moduleName = _moduleName

        except:

            moduleName = _moduleName

        # print 'ecm _moduleType,moduleName=',(_moduleType,moduleName)

        moduleBegin, moduleEnd = self.cc3dmlHelper.findModuleLine(_editor=_editor, _moduleType=_moduleType,
                                                                  _moduleName=moduleName)

        # moduleBegin,moduleEnd=self.cc3dmlHelper.findModuleLine(_editor=_editor,_moduleType=_moduleType,_moduleName=_moduleName)

        # print 'ecm moduleBegin,moduleEnd=',(moduleBegin,moduleEnd)

        snippet = self.extractSnippet(_editor, moduleBegin, moduleEnd)

        # print 'EXTRACT moduleBegin,moduleEnd=',(moduleBegin,moduleEnd)

        if moduleBegin < 0 and moduleEnd < 0:
            return None, moduleBegin, moduleEnd

        cc3dXML2ObjConverter = XMLUtils.Xml2Obj()

        try:

            root_element = cc3dXML2ObjConverter.ParseString(snippet)

        except xml.parsers.expat.ExpatError as e:

            QMessageBox.critical(self.__ui, "Error Parsing CC3DML String ", e.__str__())

            raise UserWarning(e.__str__())

        # return root_element,moduleBegin,moduleEnd

        return cc3dXML2ObjConverter, moduleBegin, moduleEnd  # have to return cc3dXML2ObjConverter instead of root_element because if we return root_element then cc3dXML2ObjConverter

        # will get deleted and child elements in root_elements get lost

    def extractElementListProperties(self, _root_element, _elementFormat):

        ''' 

            _elementFormat=[ElementName,TypeOfElementValue,[AttribName,Type,Optional,MultiDictKey],[AttribName,Type,Optional,MultiDictKey],...]

            e.g. [VolumeEnergyParameters,None,[CellType,String,False,True],[LambdaVolume,Double,False,False],[TargetVolume,Double,False,False]]

            

            _returnObjectFormat=[MultiDimDict={(Key1,Key2):[ElementValue,Attrib1,Attrib2]}]    

            

        '''

        # parsing _elementFormat to determine type of the return object

        try:

            elementName = _elementFormat[0]

            elementValueFormat = _elementFormat[1]

            multiDictKeys = []

            attributes = []

            elementFormatLength = len(_elementFormat)

            attributeFormatList = _elementFormat[2:]

            for attributeFormat in attributeFormatList:

                if attributeFormat[3]:
                    multiDictKeys.append(attributeFormat[0])

        except IndexError as e:

            print('wrong description of element format:', _elementFormat)

            return None

        multiDictConstructionCode = ''

        closingPart = ''

        multiDictAssignmentCode = ''

        print('THIS IS multiDictConstructionCode=', multiDictConstructionCode)

        for number, key in enumerate(multiDictKeys):

            if multiDictConstructionCode == '':

                multiDictConstructionCode = 'MultiDimDict('

                closingPart = 'dict)'

                multiDictAssignmentCode = 'moduleDataDict[keyList[%s]]' % str(number)

            else:

                multiDictConstructionCode += 'lambda:MultiDimDict('

                closingPart += ')'

                multiDictAssignmentCode += '[keyList[%s]]' % str(number)

        if multiDictConstructionCode == '':

            return None

        else:

            multiDictConstructionCode += closingPart

            print('THIS IS multiDictConstructionCode=', multiDictConstructionCode)

            print('multiDictAssignmentCode=', multiDictAssignmentCode)

        # Constructing multiDict

        moduleDataDict = eval(multiDictConstructionCode)

        print('moduleDataDict=', moduleDataDict)

        # parsing XML element and extracting data to put them into object defined by multiDictConstructionCode

        print('elementName=', elementName)

        elementVec = CC3DXMLListPy(_root_element.getElements(elementName))

        print('elementVec.size()=', elementVec.getBaseClass().size())

        print('elementVec=', elementVec)

        print('attributeFormatList=', attributeFormatList)

        for element in elementVec:

            print('processing element')

            keyList = []

            valueList = []

            if elementValueFormat is not None:

                functionName = 'get'

                if elementValueFormat == '':

                    functionName += 'Text'

                else:

                    functionName += elementValueFormat

                elementValueFetcherFunction = getattr(element, functionName)

                elementValue = elementValueFetcherFunction()

                valueList.append(elementValue)

            for attributeFormat in attributeFormatList:

                attributeName = attributeFormat[0]

                attributeType = attributeFormat[1]

                attributeOptionalFlag = attributeFormat[2]

                attributeKeyFlag = attributeFormat[3]

                attributeValue = None

                print('attributeFormat=', attributeFormat)

                if element.findAttribute(attributeName):

                    attributeFetcherFunction = getattr(element, 'getAttribute' + attributeType)

                    attributeValue = attributeFetcherFunction(attributeName)

                    # adding key or value

                    if attributeKeyFlag:

                        keyList.append(attributeValue)

                    else:

                        valueList.append(attributeValue)



                else:

                    if not attributeOptionalFlag:
                        keyList = []

                        valueList = []

                        break

            print('keyList=', keyList)

            print('valueList=', valueList)

            if len(keyList):
                exec(multiDictAssignmentCode + '=valueList')

        print('moduleDataDict=', moduleDataDict)

        return moduleDataDict

        # def findModule(self,editor,aliases,_moduleType='Plugin',_moduleName=['Name',''])

    def warnIfModuleExists(self, _editor, _moduleType,
                           _moduleName):  # _moduleType='Plugin',_moduleName=['Name',['Volume','VolumeLocalFlex']]

        # added for clarity purposes

        editor = _editor

        moduleType = _moduleType

        moduleName = _moduleName[0]

        aliases = _moduleName[1]

        currentAlias = ''

        moduleBegin = -1

        moduleEnd = -1

        # print 'aliases=',aliases

        # print 'self.cc3dmlHelper=',self.cc3dmlHelper

        for alias in aliases:

            moduleBegin, moduleEnd = self.cc3dmlHelper.findModuleLine(editor, _moduleType=moduleType,
                                                                      _moduleName=[moduleName, alias])

            # print  'moduleBegin,moduleEnd=',(moduleBegin,moduleEnd)

            if moduleBegin >= 0 and moduleEnd >= 0:
                currentAlias = alias

                break

        if not len(aliases):  # this is for modules like <Potts> which have no attributes in the opening element

            moduleBegin, moduleEnd = self.cc3dmlHelper.findModuleLine(editor, _moduleType=moduleType, _moduleName=[])

            # print 'moduleBegin,moduleEnd=',(moduleBegin,moduleEnd)

            if moduleBegin >= 0 and moduleEnd >= 0:
                currentAlias = moduleType

        if currentAlias != '':

            editor.ensureLineVisible(moduleBegin)

            editor.setCursorPosition(moduleBegin, 0)

            # manually constructing MessageBox

            mb = QMessageBox(self.__ui)

            mb.setWindowModality(Qt.WindowModal)

            mb.setIcon(QMessageBox.Question)

            mb.setWindowTitle("%s module already defined " % currentAlias)

            mb.setText(
                "%s module is already defined. Would you like to replace %s using cell-type-based definition? Or would you like to comment out the existing code and add new code?" % (
                    currentAlias, currentAlias))

            replaceButton = mb.addButton('&Replace', QMessageBox.AcceptRole)

            commentOutAndAddButton = mb.addButton('&Comment Out And Add', QMessageBox.AcceptRole)

            mb.addButton(QMessageBox.Cancel)

            mb.setDefaultButton(replaceButton)

            mb.exec_()

            if mb.clickedButton() == mb.button(QMessageBox.Cancel):
                return MB_CANCEL, currentAlias

            if mb.clickedButton() == replaceButton:
                return MB_REPLACE, currentAlias

            if mb.clickedButton() == commentOutAndAddButton:
                return MB_COMMENTOUTANDADD, currentAlias

        return MB_INSERTFRESH, ''

    def commentOutExistingCode(self, _editor, _moduleBegin, _moduleEnd):

        _editor.setSelection(_moduleBegin, 0, _moduleEnd, 1)

        self.__ui.block_comment()

        curLine, curPos = _editor.getCursorPosition()

        _editor.setCursorPosition(curLine + 1, 0)

    def remmoveExistingCode(self, _editor, _moduleBegin, _moduleEnd):

        if _moduleBegin >= 0 and _moduleEnd >= 0 and _moduleBegin <= _moduleEnd:
            self.cc3dmlHelper.removeLines(_editor, _moduleBegin, _moduleEnd)

    def handleNewSnippet(self, _editor, _newSnippet, _taskFlag, _moduleBegin, _moduleEnd):

        # print 'handleNewSnippet _moduleBegin,_moduleEnd = ',(_moduleBegin,_moduleEnd)

        if _taskFlag == MB_CANCEL:
            return

        if _taskFlag == MB_REPLACE:

            self.remmoveExistingCode(_editor, _moduleBegin, _moduleEnd)

        elif _taskFlag == MB_COMMENTOUTANDADD:

            self.commentOutExistingCode(_editor, _moduleBegin, _moduleEnd)

        elif _taskFlag == MB_INSERTFRESH:

            pass  # we will insert code at the current position of the cursor

        _editor.beginUndoAction()

        _editor.insert(_newSnippet)

        _editor.endUndoAction()

    @SnippetDecorator('Plugin', ['Name', ['Connectivity']])
    def handleConnectivity(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newXMLElement = self.generator.generateConnectivityPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ConnectivityGlobal']])
    def handleConnectivityGlobal(self, *args, **kwds):

        newXMLElement = self.generator.generateConnectivityGlobalPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ImplicitMotility', 'ImplicitMotilityLocal']])
    def handleImplicitMotility(self, *args, **kwds):

        root_element = kwds['root_element']

        constraintDataDict = {}

        keyString = 'Motility'

        # parse existing plugin (VolumeLocalFlex)       

        if root_element is not None:
            constraintDataDict = self.extractElementListProperties(root_element, [keyString + 'EnergyParameters', None,
                                                                                  ['CellType', '', False, True],
                                                                                  ['Target' + keyString, 'AsDouble',
                                                                                   False, False],
                                                                                  ['Lambda' + keyString, 'AsDouble',
                                                                                   False, False]])

        kwds['constraintDataDict'] = constraintDataDict

        newXMLElement = self.generator.generateVolumeFlexPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Volume', 'VolumeFlex', 'VolumeLocalFlex']])
    def handleVolume(self, *args, **kwds):

        root_element = kwds['root_element']

        constraintDataDict = {}

        keyString = 'Volume'

        # parse existing plugin (VolumeLocalFlex)       

        if root_element is not None:
            constraintDataDict = self.extractElementListProperties(root_element, [keyString + 'EnergyParameters', None,
                                                                                  ['CellType', '', False, True],
                                                                                  ['Target' + keyString, 'AsDouble',
                                                                                   False, False],
                                                                                  ['Lambda' + keyString, 'AsDouble',
                                                                                   False, False]])

        kwds['constraintDataDict'] = constraintDataDict

        newXMLElement = self.generator.generateVolumeFlexPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Surface', 'SurfaceFlex', 'SurfaceLocalFlex']])
    def handleSurface(self, *args, **kwds):

        root_element = kwds['root_element']

        constraintDataDict = {}

        keyString = 'Surface'

        # parse existing plugin (VolumeLocalFlex)       

        if root_element is not None:
            constraintDataDict = self.extractElementListProperties(root_element, [keyString + 'EnergyParameters', None,
                                                                                  ['CellType', '', False, True],
                                                                                  ['Target' + keyString, 'AsDouble',
                                                                                   False, False],
                                                                                  ['Lambda' + keyString, 'AsDouble',
                                                                                   False, False]])

        kwds['constraintDataDict'] = constraintDataDict

        newXMLElement = self.generator.generateSurfaceFlexPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ExternalPotential']])
    def handleExternalPotential(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newXMLElement = self.generator.generateExternalPotentialPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['CenterOfMass']])
    def handleCenterOfMass(self, *args, **kwds):

        newXMLElement = self.generator.generateCenterOfMassPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['CellTypeMonitor']])
    def handleCellTypeMonitor(self, *args, **kwds):

        newXMLElement = self.generator.generateCellTypeMonitorPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['NeighborTracker']])
    def handleNeighborTracker(self, *args, **kwds):

        newXMLElement = self.generator.generateNeighborTrackerPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['PixelTracker']])
    def handlePixelTracker(self, *args, **kwds):

        newXMLElement = self.generator.generatePixelTrackerPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['BoundaryPixelTracker']])
    def handleBoundaryPixelTracker(self, *args, **kwds):

        newXMLElement = self.generator.generateBoundaryPixelTrackerPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['LengthConstraint', 'LengthConstraintLocalFlex']])
    def handleLengthConstraint(self, *args, **kwds):

        root_element = kwds['root_element']

        constraintDataDict = {}

        # parse existing plugin (VolumeLocalFlex)       

        if root_element is not None:
            constraintDataDict = self.extractElementListProperties(root_element, ['LengthEnergyParameters', None,
                                                                                  ['CellType', '', False, True], \
 \
                                                                                  ['TargetLength', 'AsDouble', False,
                                                                                   False],
                                                                                  ['MinorTargetLength', 'AsDouble',
                                                                                   False, False],
                                                                                  ['LambdaLength', 'AsDouble', False,
                                                                                   False]])

        kwds['constraintDataDict'] = constraintDataDict

        newXMLElement = self.generator.generateLengthConstraintPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Contact']])
    def handleContact(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        contactMatrix = {}

        if root_element:
            contactMatrix = self.extractElementListProperties(root_element,
                                                              ['Energy', 'Double', ['Type1', '', False, True],
                                                               ['Type2', '', False, True]])

            # contactElement=ElementCC3D("Plugin",{"Name":"Contact"})

        kwds['contactMatrix'] = contactMatrix

        kwds['NeighborOrder'] = self.getNeighborOrder(root_element)

        # kwds['insert_root_element']=contactElement

        # we use existing entries to rewrite contact matrix

        newXMLElement = self.generator.generateContactPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ContactInternal']])
    def handleContactInternal(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        contactMatrix = {}

        if root_element:
            contactMatrix = self.extractElementListProperties(root_element,
                                                              ['Energy', 'Double', ['Type1', '', False, True],
                                                               ['Type2', '', False, True]])

        # contactElement=ElementCC3D("Plugin",{"Name":"ContactInternal"})

        kwds['contactMatrix'] = contactMatrix

        kwds['NeighborOrder'] = self.getNeighborOrder(root_element)

        # kwds['insert_root_element']=contactElement

        newXMLElement = self.generator.generateContactInternalPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ContactCompartment']])
    def handleContactCompartment(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        contactMatrix = {}

        internalContactMatrix = {}

        if root_element:
            contactMatrix = self.extractElementListProperties(root_element,
                                                              ['Energy', 'Double', ['Type1', '', False, True],
                                                               ['Type2', '', False, True]])

            internalContactMatrix = self.extractElementListProperties(root_element, ['InternalEnergy', 'Double',
                                                                                     ['Type1', '', False, True],
                                                                                     ['Type2', '', False, True]])

        kwds['contactMatrix'] = contactMatrix

        kwds['internalContactMatrix'] = internalContactMatrix

        kwds['NeighborOrder'] = self.getNeighborOrder(root_element)

        newXMLElement = self.generator.generateCompartmentPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ContactLocalProduct']])
    def handleContactLocalProduct(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        specificityMatrix = {}

        if root_element:
            specificityMatrix = self.extractElementListProperties(root_element, ['ContactSpecificity', 'Double',
                                                                                 ['Type1', '', False, True],
                                                                                 ['Type2', '', False, True]])

        kwds['specificityMatrix'] = specificityMatrix

        kwds['NeighborOrder'] = self.getNeighborOrder(root_element)

        newXMLElement = self.generator.generateContactLocalProductPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['FocalPointPlasticity']])
    def handleFocalPointPlasticity(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        kwds['NeighborOrder'] = self.getNeighborOrder(root_element)

        newXMLElement = self.generator.generateFocalPointPlasticityPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['ElasticityTracker']])
    def handleElasticityTracker(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newXMLElement = self.generator.generateElasticityTrackerPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Elasticity']])
    def handleElasticity(self, *args, **kwds):

        root_element = kwds['root_element']

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newXMLElement = self.generator.generateElasticityPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['AdhesionFlex']])
    def handleAdhesionFlex(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        dlg = AdhesionFlexDlg(editor, self.__ui)

        ret = dlg.exec_()

        newSnippet = ''

        if ret:
            afData, formula = dlg.extractInformation()

            kwds['afData'] = afData

            kwds['formula'] = formula

            newXMLElement = self.generator.generateAdhesionFlexPlugin(*args, **kwds)

            newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Chemotaxis']])
    def handleChemotaxis(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        chemotaxisData = {}

        chemDict1 = {}

        chemDict1["CellType"] = 'CHEMOTAXING_TYPE'

        chemDict1["Lambda"] = 1.0

        chemDict1["ChemotaxTowards"] = 'CELL_TYPES'

        chemDict1["SatCoef"] = 0.0

        chemDict1["ChemotaxisType"] = 'regular'

        try:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'].append(chemDict1)

        except LookupError:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'] = [chemDict1]

        chemDict2 = {}

        chemDict2["CellType"] = 'CHEMOTAXING_TYPE'

        chemDict2["Lambda"] = 1.0

        chemDict2["ChemotaxTowards"] = 'CELL_TYPES'

        chemDict2["SatCoef"] = 100.0

        chemDict2["ChemotaxisType"] = 'saturation'

        try:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'].append(chemDict2)

        except LookupError:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'] = [chemDict2]

        chemDict3 = {}

        chemDict3["CellType"] = 'CHEMOTAXING_TYPE'

        chemDict3["Lambda"] = 1.0

        chemDict3["ChemotaxTowards"] = 'CELL_TYPES'

        chemDict3["SatCoef"] = 10.1

        chemDict3["ChemotaxisType"] = 'saturation linear'

        try:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'].append(chemDict3)

        except LookupError:

            chemotaxisData['FIELD_FROM_PDE_SOLVER'] = [chemDict3]

        kwds['chemotaxisData'] = chemotaxisData

        # kwds['pdeFieldData']=pdeFieldData

        newXMLElement = self.generator.generateChemotaxisPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['Secretion']])
    def handleSecretion(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        secretionData = {}

        secrDict1 = {}

        secrDict1["CellType"] = 'CELL_TYPE_NAME'

        secrDict1["Rate"] = 0.1

        secrDict1["OnContactWith"] = ''

        secrDict1["SecretionType"] = 'uniform'

        try:

            secretionData['FIELD_FROM_PDE_SOLVER'].append(secrDict1)

        except LookupError:

            secretionData['FIELD_FROM_PDE_SOLVER'] = [secrDict1]

        secrDict2 = {}

        secrDict2["CellType"] = 'CELL_TYPE_NAME'

        secrDict2["Rate"] = 1.1

        secrDict2["OnContactWith"] = 'COMMA_SEPARATED_TYPE_NAMES'

        secrDict2["SecretionType"] = 'on contact'

        try:

            secretionData['FIELD_FROM_PDE_SOLVER'].append(secrDict2)

        except LookupError:

            secretionData['FIELD_FROM_PDE_SOLVER'] = [secrDict2]

        secrDict3 = {}

        secrDict3["CellType"] = 'CELL_TYPE_NAME'

        secrDict3["Rate"] = 0.5

        secrDict3["OnContactWith"] = ''

        secrDict3["SecretionType"] = 'constant concentration'

        try:

            secretionData['FIELD_FROM_PDE_SOLVER'].append(secrDict3)

        except LookupError:

            secretionData['FIELD_FROM_PDE_SOLVER'] = [secrDict3]

        kwds['secretionData'] = secretionData

        # kwds['pdeFieldData']=pdeFieldData

        newXMLElement = self.generator.generateSecretionPlugin(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['UniformInitializer']])
    def handleUniformInitializer(self, *args, **kwds):

        newXMLElement = self.generator.generateUniformInitializerSteppable(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['BlobInitializer']])
    def handleBlobInitializer(self, *args, **kwds):

        newXMLElement = self.generator.generateBlobInitializerSteppable(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['PIFInitializer']])
    def handlePIFInitializer(self, *args, **kwds):

        newXMLElement = self.generator.generatePIFInitializerSteppable(*args, **kwds)

        return newXMLElement.getCC3DXMLElementString()

    @SnippetDecorator('Steppable', ['Type', ['PIFDumper']])
    def handlePIFDumper(self, *args, **kwds):

        newXMLElement = self.generator.generatePIFDumperSteppable(*args, **kwds)

        return newXMLElement.getCC3DXMLElementString()

    @SnippetDecorator('Steppable', ['Type', ['BoxWatcher']])
    def handleBoxWatcher(self, *args, **kwds):

        newXMLElement = self.generator.generateBoxWatcherSteppable(*args, **kwds)

        return newXMLElement.getCC3DXMLElementString()

    @SnippetDecorator('Steppable', ['Type', ['DiffusionSolverFE']])
    def handleDiffusionSolverFE(self, *args, **kwds):

        pdeFieldData = {}

        pdeFieldData['FIELD_NAME_1'] = 'DiffusionSolverFE'

        pdeFieldData['FIELD_NAME_2'] = 'DiffusionSolverFE'

        kwds['pdeFieldData'] = pdeFieldData

        newXMLElement = self.generator.generateDiffusionSolverFE(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['FlexibleDiffusionSolverFE']])
    def handleFlexibleDiffusionSolverFE(self, *args, **kwds):

        pdeFieldData = {}

        pdeFieldData['FIELD_NAME_1'] = 'FlexibleDiffusionSolverFE'

        pdeFieldData['FIELD_NAME_2'] = 'FlexibleDiffusionSolverFE'

        kwds['pdeFieldData'] = pdeFieldData

        newXMLElement = self.generator.generateFlexibleDiffusionSolverFE(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['FastDiffusionSolver2DFE']])
    def handleFastDiffusionSolver2DFE(self, *args, **kwds):

        pdeFieldData = {}

        pdeFieldData['FIELD_NAME_1'] = 'FastDiffusionSolver2DFE'

        pdeFieldData['FIELD_NAME_2'] = 'FastDiffusionSolver2DFE'

        kwds['pdeFieldData'] = pdeFieldData

        newXMLElement = self.generator.generateFastDiffusionSolver2DFE(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['KernelDiffusionSolver']])
    def handleKernelDiffusionSolver(self, *args, **kwds):

        pdeFieldData = {}

        pdeFieldData['FIELD_NAME_1'] = 'KernelDiffusionSolver'

        pdeFieldData['FIELD_NAME_2'] = 'KernelDiffusionSolver'

        kwds['pdeFieldData'] = pdeFieldData

        newXMLElement = self.generator.generateKernelDiffusionSolver(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Steppable', ['Type', ['SteadyStateDiffusionSolver', 'SteadyStateDiffusionSolver2D']])
    def handleSteadyStateDiffusionSolver(self, *args, **kwds):

        pdeFieldData = {}

        pdeFieldData['FIELD_NAME_1'] = 'SteadyStateDiffusionSolver'

        pdeFieldData['FIELD_NAME_2'] = 'SteadyStateDiffusionSolver'

        kwds['pdeFieldData'] = pdeFieldData

        newXMLElement = self.generator.generateSteadyStateDiffusionSolver(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Plugin', ['Name', ['CellType']])
    def handleCellType(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        dlg = CellTypeDlg(editor, self.__ui)

        ret = dlg.exec_()

        newSnippet = ''

        if ret:
            cellTypeData = dlg.extractInformation()

            kwds['data'] = cellTypeData

            newXMLElement = self.generator.generateCellTypePlugin(*args, **kwds)

            newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Potts', ['', []], 'Potts', 'CPM Configuration')
    # def handlePotts(self,*args,**kwds):

    def handlePottsCPMConfiguration(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        gpd = self.generator.getCurrentPottsSection(*args, **kwds)

        dlg = PottsDlg(editor, self.__ui)

        dlg.initialize(gpd=gpd)

        ret = dlg.exec_()

        newSnippet = ''

        if not ret:
            return

        newXMLElement = dlg.generateXML()

        # newXMLElement = self.generator.generatePottsSection(*args,**kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Metadata', ['', []], 'Metadata', 'Simulation Properties')
    def handleMetadataSimulationProperties(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        newXMLElement = self.generator.generateMetadataSimulationProperties(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Metadata', ['', []], 'Metadata', 'Debug Output Frequency')
    def handleMetadataDebugOutputFrequency(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        newXMLElement = self.generator.generateMetadataDebugOutputFrequency(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Metadata', ['', []], 'Metadata', 'Parallel Execution')
    def handleMetadataParallelExecution(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        newXMLElement = self.generator.generateMetadataParallelExecution(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    @SnippetDecorator('Metadata', ['', []], 'Metadata', 'Parallel Execution Single CPU Potts')
    def handleMetadataParallelExecutionSingleCPUPotts(self, *args, **kwds):

        cellTypeData = kwds['data']

        editor = kwds['editor']

        newSnippet = ''

        newXMLElement = self.generator.generateMetadataParallelExecutionSingleCPUPotts(*args, **kwds)

        newSnippet = newXMLElement.getCC3DXMLElementString()

        return newSnippet

    def getNeighborOrder(self, _root_element):

        if not _root_element:
            return 1

        neighborOrder = 1

        neighborOrderElement = _root_element.getFirstElement('NeighborOrder')

        # print 'neighborOrderElement=',neighborOrderElement

        if neighborOrderElement:
            neighborOrder = neighborOrderElement.getUInt()

        if neighborOrder <= 0:
            neighborOrder = 1

        return neighborOrder
