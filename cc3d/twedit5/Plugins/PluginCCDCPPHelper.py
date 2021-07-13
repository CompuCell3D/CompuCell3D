# Start-Of-Header

name = "CC3D C++ Helper Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = True
version = "0.9.0"
className = "CC3DCPPHelper"
packageName = "__core__"
shortDescription = "Plugin assists with CC3D C++ module development scripting"
longDescription = """This plugin provides provides users with CC3D C++ code generator and code snippets -
 making CC3D C++ plugin and steppable development  more convenient."""

# End-Of-Header

from cc3d.twedit5.Plugins.TweditPluginBase import TweditPluginBase
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.Plugins.CC3DCPPHelper.Configuration import Configuration
from cc3d.twedit5.Plugins.PluginUtils.SnippetMenuParser import SnippetMenuParser
from cc3d.twedit5.Plugins.CC3DCPPHelper.CPPModuleGeneratorDialog import CPPModuleGeneratorDialog
from cc3d.twedit5.Plugins.CC3DCPPHelper.CppTemplates import CppTemplates
from distutils.dir_util import mkpath
import os.path
import re

error = ''


class CC3DCPPHelper(QObject, TweditPluginBase):
    """

    Class implementing the About plugin.

    """

    def __init__(self, ui):

        """

        Constructor

        

        @param ui reference to the user interface object (UI.UserInterface)

        """

        QObject.__init__(self, ui)
        TweditPluginBase.__init__(self)

        self.__ui = ui

        self.configuration = Configuration(self.__ui.configuration.settings)

        self.snippetDictionary = {}

        self.initialize()

        self.cppTemplates = None

        # useful regular expressions

        self.nonwhitespaceRegex = re.compile('^[\s]*[\S]+')

        self.commentRegex = re.compile('^[\s]*#')

        self.defFunRegex = re.compile('^[\s]*def')

        self.blockStatementRegex = re.compile(

            ':[\s]*$')  # block statement - : followed by whitespaces at the end of the line

        self.blockStatementWithCommentRegex = re.compile(

            ':[\s]*[#]+[\s\S]*$')  # block statement - : followed by whitespaces at the end of the line

    def initialize(self):

        '''  

            initializes containers used in the plugin

            '''

        self.actions = {}

        self.actionGroupDict = {}

        self.actionGroupMenuDict = {}

        self.cppMenuAction = None

    def addSnippetDictionaryEntry(self, _snippetName, _snippetProperties):

        self.snippetDictionary[_snippetName] = _snippetProperties

    def getUI(self):

        return self.__ui

    def activate(self):

        """

        Public method to activate this plugin.

        

        @return tuple of None and activation status (boolean)

        """

        self.snippetMapper = QSignalMapper(self.__ui)

        # self.__ui.connect(self.snippetMapper, SIGNAL("mapped(const QString&)"), self.__insertSnippet)

        self.snippetMapper.mapped[str].connect(self.__insertSnippet)

        # # # self.snippetDictionary=self.snippetUtils.getCodeSnippetsDict()

        self.__initMenus()

        self.__initActions()

        return None, True

    def deactivate(self):

        """

        Public method to deactivate this plugin.

        """

        # self.__ui.disconnect(self.snippetMapper, SIGNAL("mapped(const QString&)"), self.__insertSnippet)

        self.snippetMapper.mapped[str].disconnect(self.__insertSnippet)

        try:

            # self.__ui.disconnect(self.actions['Generate New Module...'], SIGNAL("triggered()"), self.generateNewModule)

            self.actions['Generate New Module...'].triggered.disconnect(self.generateNewModule)

        except LookupError as e:

            pass

        for actionName, action in self.actions.items():

            # self.__ui.disconnect(action, SIGNAL("triggered()"), self.snippetMapper, SLOT("map()"))

            try:

                action.triggered.disconnect(self.snippetMapper.map)

            except TypeError:

                print('Skipping disconnecting from map of action {}'.format(actionName))

                pass

        self.cc3dcppMenu.clear()

        self.__ui.menuBar().removeAction(self.cc3dcppMenuAction)

        self.initialize()

        return

    def __initMenus(self):

        self.cc3dcppMenu = QMenu("CC3&D C++", self.__ui.menuBar())

        # inserting CC3D Project Menu as first item of the menu bar of twedit++

        self.cc3dcppMenuAction = self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(), self.cc3dcppMenu)

    def __initActions(self):

        """

        Private method to initialize the actions.        

        """

        self.actions['Generate New Module...'] = QAction(QtGui.QIcon(':/icons/document-new.png'),

                                                         "&Generate New Module...",

                                                         self,

                                                         statusTip="Create new C++ CC3D module")

        # self.__ui.connect(self.actions['Generate New Module...'], SIGNAL("triggered()"), self.generateNewModule)

        self.actions['Generate New Module...'].triggered.connect(self.generateNewModule)

        self.actions['Deactivate'] = QAction("Deactivate",

                                             self,

                                             statusTip="Deactivate C++ CC3D plugin", triggered=self.deactivate)

        self.cc3dcppMenu.addAction(self.actions['Generate New Module...'])

        self.cc3dcppMenu.addAction(self.actions['Deactivate'])

        # ----------------------------

        self.cc3dcppMenu.addSeparator()

        self.snippetDictionary = {}

        psmp = SnippetMenuParser()

        snippetFilePath = os.path.abspath(

            os.path.join(os.path.dirname(__file__), 'CC3DCPPHelper/Snippets.cpp.template'))

        psmp.readSnippetMenu(snippetFilePath)

        snippetMenuDict = psmp.getSnippetMenuDict()

        for menuName, submenuDict in iter(sorted(snippetMenuDict.items())):

            print('menuName=', menuName)

            groupMenu = self.cc3dcppMenu.addMenu(menuName)

            for subMenuName, snippetText in iter(sorted(submenuDict.items())):
                action = groupMenu.addAction(subMenuName)

                actionKey = menuName.strip() + ' ' + subMenuName.strip()  # for lookup int he self.snippetDictionary

                self.snippetDictionary[actionKey] = snippetText

                self.actions[actionKey] = action

                # self.__ui.connect(action, SIGNAL("triggered()"), self.snippetMapper, SLOT("map()"))

                action.triggered.connect(self.snippetMapper.map)

                self.snippetMapper.setMapping(action, actionKey)

    def generateNewModule(self):

        cmgd = CPPModuleGeneratorDialog(self.__ui)

        # set recent  directory name

        cmgd.moduleDirLE.setText(self.configuration.setting('RecentModuleDirectory'))

        features = {}

        if not cmgd.exec_():
            return

        capitalFirst = lambda s: s[:1].upper() + s[1:] if s else ''

        # first check if directory exists and if it is writeable. If the module directory  already exists ask if user wants to overwrite files there with new, generated ones    

        dirName = str(cmgd.moduleDirLE.text()).rstrip()

        codeLayout = 'maincode'

        if cmgd.developerZoneLayoutRB.isChecked():
            codeLayout = 'developerzone'

        self.configuration.setSetting("RecentModuleDirectory", dirName)

        moduleCoreName = str(cmgd.moduleCoreNameLE.text()).rstrip()

        moduleCoreName = capitalFirst(moduleCoreName)

        fullModuleDir = os.path.join(dirName, moduleCoreName)

        fullModuleDir = os.path.abspath(fullModuleDir)

        # SWIG files

        coreDir = os.path.join(dirName, '../../')

        coreDir = os.path.abspath(coreDir)

        swigFileDir = os.path.join(coreDir, 'pyinterface/CompuCellPython')

        swigFileDir = os.path.abspath(swigFileDir)

        try:

            mainSwigFile = os.path.join(swigFileDir, 'CompuCell.i')

            mainSwigFile = os.path.abspath(mainSwigFile)

            declarationsSwigFile = os.path.join(swigFileDir, 'CompuCellExtraDeclarations.i')

            declarationsSwigFile = os.path.abspath(declarationsSwigFile)

            includesSwigFile = os.path.join(swigFileDir, 'CompuCellExtraIncludes.i')

            includesSwigFile = os.path.abspath(includesSwigFile)

            cmakeSwigFile = os.path.join(swigFileDir, 'CMakeLists.txt')

            cmakeSwigFile = os.path.abspath(cmakeSwigFile)

            f = open(mainSwigFile, 'r+')

            f.close()

            f = open(declarationsSwigFile, 'r+')

            f.close()

            f = open(includesSwigFile, 'r+')

            f.close()

            f = open(cmakeSwigFile, 'r+')

            f.close()



        except IOError as e:

            mainSwigFile = ''

            declarationsSwigFile = ''

            includesSwigFile = ''

            cmakeSwigFile = ''

        if codeLayout == 'developerzone':

            try:

                coreDir = os.path.abspath(dirName)

                swigFileDir = os.path.abspath(os.path.join(coreDir, 'pyinterface/CompuCellExtraModules'))

                mainSwigFile = os.path.join(swigFileDir, 'CompuCellExtraModules.i')

                cmakeSwigFile = os.path.join(swigFileDir, 'CMakeLists.txt')

            except IOError as e:

                mainSwigFile = ''

                cmakeSwigFile = ''

        print(swigFileDir)

        print(mainSwigFile)

        features['codeLayout'] = codeLayout

        features['mainSwigFile'] = mainSwigFile

        features['declarationsSwigFile'] = declarationsSwigFile

        features['includesSwigFile'] = includesSwigFile

        features['cmakeSwigFile'] = cmakeSwigFile

        if cmgd.extraAttribCB.isChecked():
            features['ExtraAttribute'] = True

        if os.path.exists(fullModuleDir):

            message = "Directory %s already exists. <br>Is it OK to overwrite content in this directory with generated files?" % fullModuleDir

            ret = QtWidgets.QMessageBox.warning(self.__ui, " Module directory already exists",

                                                message,

                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

            if ret == QtWidgets.QMessageBox.No:
                return

        else:

            try:

                self.makeDirectory(fullModuleDir)

            except IOError as e:

                message = "Could not create directory %s. <br> Please make sure you have necessary permissions" % fullModuleDir

                QtGui.QMessageBox.information(self.__ui, "Could not create directory", message,
                                              QtWidgets.QMessageBox.Ok)

                return

        if not self.cppTemplates:
            self.cppTemplates = CppTemplates()

        if cmgd.pluginRB.isChecked():

            generatedFileList = []

            features['Plugin'] = moduleCoreName

            features['Module'] = moduleCoreName

            if cmgd.energyFcnCB.isChecked():
                features['EnergyFunction'] = True

            if cmgd.latticeMonitorCB.isChecked():
                features['LatticeMonitor'] = True

            if cmgd.stepperCB.isChecked():
                features['Stepper'] = True

            # write CMake file

            cmakeText = self.cppTemplates.generateCMakeFile(features)

            cmakeFileName = os.path.join(fullModuleDir, "CMakeLists.txt")

            cmakeFileName = os.path.abspath(cmakeFileName)

            self.writeTextToFile(cmakeFileName, cmakeText)

            generatedFileList.append(cmakeFileName)

            # write plugin header file

            pluginHeaderName = moduleCoreName + 'Plugin.h'

            pluginHeaderText = self.cppTemplates.generatePluginHeaderFile(features)

            pluginHeaderName = os.path.join(fullModuleDir, pluginHeaderName)

            pluginHeaderName = os.path.abspath(pluginHeaderName)

            self.writeTextToFile(pluginHeaderName, pluginHeaderText)

            generatedFileList.append(pluginHeaderName)

            # write proxy initializer

            pluginProxyName = moduleCoreName + 'PluginProxy.cpp'

            pluginProxyText = self.cppTemplates.generatePluginProxyFile(features)

            pluginProxyName = os.path.join(fullModuleDir, pluginProxyName)

            pluginProxyName = os.path.abspath(pluginProxyName)

            self.writeTextToFile(pluginProxyName, pluginProxyText)

            generatedFileList.append(pluginProxyName)

            # write DLL specifier

            pluginDLLSpecifierName = moduleCoreName + 'DLLSpecifier.h'

            pluginDLLSpecifierText = self.cppTemplates.generatePluginDLLSpecifier(features)

            pluginDLLSpecifierName = os.path.join(fullModuleDir, pluginDLLSpecifierName)

            pluginDLLSpecifierName = os.path.abspath(pluginDLLSpecifierName)

            self.writeTextToFile(pluginDLLSpecifierName, pluginDLLSpecifierText)

            generatedFileList.append(pluginDLLSpecifierName)

            # write plugin implementation

            pluginImplementationName = moduleCoreName + 'Plugin.cpp'

            pluginImplementationText = self.cppTemplates.generatePluginImplementationFile(features)

            pluginImplementationName = os.path.join(fullModuleDir, pluginImplementationName)

            pluginImplementationName = os.path.abspath(pluginImplementationName)

            self.writeTextToFile(pluginImplementationName, pluginImplementationText)

            generatedFileList.append(pluginImplementationName)

            try:

                features['ExtraAttribute']

                # write extra attribute data header

                pluginExtraAttributeFileName = moduleCoreName + 'Data.h'

                pluginExtraAttributeText = self.cppTemplates.generatePluginExtraAttributeFile(features)

                pluginExtraAttributeFileName = os.path.join(fullModuleDir, pluginExtraAttributeFileName)

                pluginExtraAttributeFileName = os.path.abspath(pluginExtraAttributeFileName)

                self.writeTextToFile(pluginExtraAttributeFileName, pluginExtraAttributeText)

                generatedFileList.append(pluginExtraAttributeFileName)



            except LookupError as e:

                pass

            # adding entry in the    plugins/CMakeLists.txt

            self.addModuleToModuleDirCMakeFile(moduleCoreName, dirName)

            # moduleMainCMakeFile=os.path.join(dirName,'CMakeLists.txt')

            # generatedFileList.append(moduleMainCMakeFile)

            if cmgd.pythonWrapCB.isChecked():

                if features['codeLayout'] == 'maincode':

                    self.modifySwigFiles(features)

                elif features['codeLayout'] == 'developerzone':

                    self.modifySwigFilesDeveloperZone(features)

            # open all generated files in Twedit

            self.__ui.loadFiles(generatedFileList)



        elif cmgd.steppableRB.isChecked():

            features['Steppable'] = moduleCoreName

            features['Module'] = moduleCoreName

            generatedFileList = []

            # write CMake file

            cmakeText = self.cppTemplates.generateCMakeFileSteppable(features)

            cmakeFileName = os.path.join(fullModuleDir, "CMakeLists.txt")

            cmakeFileName = os.path.abspath(cmakeFileName)

            self.writeTextToFile(cmakeFileName, cmakeText)

            generatedFileList.append(cmakeFileName)

            # write proxy initializer

            steppableProxyName = moduleCoreName + 'Proxy.cpp'

            steppableProxyText = self.cppTemplates.generateSteppableProxyFile(features)

            steppableProxyName = os.path.join(fullModuleDir, steppableProxyName)

            steppableProxyName = os.path.abspath(steppableProxyName)

            self.writeTextToFile(steppableProxyName, steppableProxyText)

            generatedFileList.append(steppableProxyName)

            # write DLL specifier

            steppableDLLSpecifierName = moduleCoreName + 'DLLSpecifier.h'

            steppableDLLSpecifierText = self.cppTemplates.generateSteppableDLLSpecifier(features)

            steppableDLLSpecifierName = os.path.join(fullModuleDir, steppableDLLSpecifierName)

            steppableDLLSpecifierName = os.path.abspath(steppableDLLSpecifierName)

            self.writeTextToFile(steppableDLLSpecifierName, steppableDLLSpecifierText)

            generatedFileList.append(steppableDLLSpecifierName)

            # write steppable header file

            steppableHeaderName = moduleCoreName + '.h'

            steppableHeaderText = self.cppTemplates.generateSteppableHeaderFile(features)

            steppableHeaderName = os.path.join(fullModuleDir, steppableHeaderName)

            steppableHeaderName = os.path.abspath(steppableHeaderName)

            self.writeTextToFile(steppableHeaderName, steppableHeaderText)

            generatedFileList.append(steppableHeaderName)

            # write steppable implementation

            steppableImplementationName = moduleCoreName + '.cpp'

            steppableImplementationText = self.cppTemplates.generateSteppableImplementationFile(features)

            steppableImplementationName = os.path.join(fullModuleDir, steppableImplementationName)

            steppableImplementationName = os.path.abspath(steppableImplementationName)

            self.writeTextToFile(steppableImplementationName, steppableImplementationText)

            generatedFileList.append(steppableImplementationName)

            try:

                features['ExtraAttribute']

                # write extra attribute data header

                steppableExtraAttributeFileName = moduleCoreName + 'Data.h'

                steppableExtraAttributeText = self.cppTemplates.generateSteppableExtraAttributeFile(features)

                steppableExtraAttributeFileName = os.path.join(fullModuleDir, steppableExtraAttributeFileName)

                steppableExtraAttributeFileName = os.path.abspath(steppableExtraAttributeFileName)

                self.writeTextToFile(steppableExtraAttributeFileName, steppableExtraAttributeText)

                generatedFileList.append(steppableExtraAttributeFileName)



            except LookupError as e:

                pass

            # adding entry in the  steppables/CMakeLists.txt

            self.addModuleToModuleDirCMakeFile(moduleCoreName, dirName)

            print('\n\n\n\n\n\n\n\n CODELAYOUT=', features['codeLayout'])

            if cmgd.pythonWrapCB.isChecked():

                if features['codeLayout'] == 'maincode':

                    self.modifySwigFiles(features)

                elif features['codeLayout'] == 'developerzone':

                    self.modifySwigFilesDeveloperZone(features)

            # open all generated files in Twedit

            self.__ui.loadFiles(generatedFileList)





        else:

            return

    def addModuleToModuleDirCMakeFile(self, _moduleName, _moduleDir):

        cmakePath = os.path.join(_moduleDir, 'CMakeLists.txt')

        cmakePath = os.path.abspath(cmakePath)

        try:

            f = open(cmakePath, 'r+')

            f.close()

        except IOError as e:

            message = "Could not find file %s. <br> Please make sure you have necessary permissions and that file exists. You will need to add newly generated module to CMakeFile.txt manually" % cmakePath

            QtGui.QMessageBox.information(self.__ui, "Could not write to file", message, QtGui.QMessageBox.Ok)

            return

        fileList = []

        fileList.append(cmakePath)

        self.__ui.loadFiles(fileList)

        editor = self.__ui.getActiveEditor()

        cmakeLine = 'ADD_SUBDIRECTORY(' + _moduleName + ')\n'

        # cmakeLineRegex='^[\s]*ADD_SUBDIRECTORY[\s]*\([\s]*'+_moduleName+'[\s]*\)'

        cmakeLineRegex = '^[\s]*[A|a][D|d][D|d]_[S|s][U|u][B|b][D|d][I|i][R|r][E|e][C|c][T|t][O|o][R|r][Y|y][\s]*([\s]*' + _moduleName + '[\s]*)'

        # first check if the line is already there

        foundFlag = editor.findFirst(cmakeLineRegex, True, True, False, False, True, 0, 0, False)

        print('FOUND REGEX ', cmakeLineRegex, ' =', foundFlag)

        if foundFlag:

            print("FOUND ALREADY THE LINE")

            return

        else:

            foundFlag = editor.findFirst('#AutogeneratedModules', False, True, False, False, True, 0, 0, False)

            if foundFlag:

                currentLine, currentIndex = editor.getCursorPosition()  # record current cursor position

                editor.insertAt(cmakeLine, currentLine + 1, 0)

                editor.ensureLineVisible(currentLine + 10)

                self.__ui.save()  # saves active editor

            else:

                return

    def modifySwigFilesDeveloperZone(self, _features):

        fileLocalizationError = False

        if _features['codeLayout'] == 'developerzone':

            if _features['mainSwigFile'] == '' or _features['cmakeSwigFile'] == '':
                fileLocalizationError = True

        if fileLocalizationError:
            message = "Could not succesfully locate SWIG files. <br> Please make sure you have necessary permissions and that file exist. You will need to add newly generated module to SWIG filesmanually"

            QtGui.QMessageBox.information(self.__ui, "Problem with SWIG files", message, QtGui.QMessageBox.Ok)

            return

        # this is the label which helps identify autogenerated code snippet.

        insertedCodeHeader = '//' + _features['Module'] + '_autogenerated'

        insertedCodeRegex1 = '^[\s]*//[\s]*' + _features['Module'] + '_autogenerated'

        insertionLocationLabel1 = '//AutogeneratedModules1'

        # inserting module header files definitione into .i file

        pluginIncludeCode = '''EXTRA_ATTRIB_INCLUDE

#include <PLUGIN_NAME_CORE/PLUGIN_NAME_COREPlugin.h>

'''

        steppableIncludeCode = '''EXTRA_ATTRIB_INCLUDE

#include <STEPPABLE_NAME_CORE/STEPPABLE_NAME_CORE.h>

'''

        insertedCode = '' + insertedCodeHeader + '\n'

        if 'Plugin' in list(_features.keys()):

            PLUGIN_NAME_CORE = _features['Plugin']

            pluginIncludeCode = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, pluginIncludeCode)

            print('_features=', _features)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_INCLUDE = '#include <' + PLUGIN_NAME_CORE + '/' + PLUGIN_NAME_CORE + 'Data.h>'

                pluginIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', EXTRA_ATTRIB_INCLUDE, pluginIncludeCode)

            else:

                pluginIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', '', pluginIncludeCode)

            insertedCode += pluginIncludeCode



        elif 'Steppable' in list(_features.keys()):

            STEPPABLE_NAME_CORE = _features['Steppable']

            steppableIncludeCode = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, steppableIncludeCode)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_INCLUDE = '#include <' + STEPPABLE_NAME_CORE + '/' + STEPPABLE_NAME_CORE + 'Data.h>'

                steppableIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', EXTRA_ATTRIB_INCLUDE, steppableIncludeCode)

            else:

                steppableIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', '', steppableIncludeCode)

            insertedCode += steppableIncludeCode

        self.addCodeToSwigFile(_features['mainSwigFile'], insertedCode, insertedCodeRegex1, insertionLocationLabel1)

        # inserting windows dll export labels

        insertedCode = insertedCodeHeader + '\n'

        insertedCode += '#define ' + _features['Module'].upper() + '_EXPORT\n'

        insertionLocationLabel2 = '//AutogeneratedModules2'

        insertedCodeRegex2 = '^[\s]*//[\s]*' + _features['Module'] + '_autogenerated'

        row, col = self.addCodeToSwigFile(_features['mainSwigFile'], insertedCode, insertedCodeRegex2,

                                          insertionLocationLabel2)

        insertionLocationLabel3 = '//AutogeneratedModules3'

        insertedCodeRegex3 = '^[\s]*//[\s]*' + _features['Module'] + '_autogenerated'

        self.addCodeToSwigFile(_features['mainSwigFile'], insertedCode, insertedCodeRegex3, insertionLocationLabel3)

        # inserting declaration part of the header file - this is the code prepended by %{

        insertedCode = insertedCodeHeader + '\n'

        insertionLocationLabel4 = '//AutogeneratedModules4'

        insertedCodeRegex4 = '^[\s]*//[\s]*' + _features['Module'] + '_autogenerated'

        pluginDeclarationCode = '''EXTRA_ATTRIB_DECLARE

%include <PLUGIN_NAME_CORE/PLUGIN_NAME_COREPlugin.h>



%inline %{

 PLUGIN_NAME_COREPlugin * getPLUGIN_NAME_COREPlugin(){

      return (PLUGIN_NAME_COREPlugin *)Simulator::pluginManager.get("PLUGIN_NAME_CORE");

   }



%}

'''

        steppableDeclarationCode = '''EXTRA_ATTRIB_DECLARE

%include <STEPPABLE_NAME_CORE/STEPPABLE_NAME_CORE.h>



%inline %{

 STEPPABLE_NAME_CORE * getSTEPPABLE_NAME_CORE(){

      return (STEPPABLE_NAME_CORE *)Simulator::steppableManager.get("STEPPABLE_NAME_CORE");

   }



%}

'''

        #  Module declaration code 

        if 'Plugin' in list(_features.keys()):

            PLUGIN_NAME_CORE = _features['Plugin']

            pluginDeclarationCode = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, pluginDeclarationCode)

            EXTRA_ATTRIB_DECLARE = ''

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_DECLARE = '''%include <PLUGIN_NAME_CORE/PLUGIN_NAME_COREData.h>

%template (PLUGIN_NAME_COREDataAccessorTemplate) BasicClassAccessor<PLUGIN_NAME_COREData>; //necessary to get PLUGIN_NAME_COREData accessor working in Python

'''

                EXTRA_ATTRIB_DECLARE = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, EXTRA_ATTRIB_DECLARE)

                pluginDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', EXTRA_ATTRIB_DECLARE, pluginDeclarationCode)





            else:

                pluginDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', '', pluginDeclarationCode)

            insertedCode += pluginDeclarationCode



        elif 'Steppable' in list(_features.keys()):

            STEPPABLE_NAME_CORE = _features['Steppable']

            steppableDeclarationCode = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, steppableDeclarationCode)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_DECLARE = '''%include <STEPPABLE_NAME_CORE/STEPPABLE_NAME_COREData.h>

%template (STEPPABLE_NAME_COREDataAccessorTemplate) BasicClassAccessor<STEPPABLE_NAME_COREData>; //necessary to get STEPPABLE_NAME_COREData accessor working in Python

'''

                EXTRA_ATTRIB_DECLARE = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, EXTRA_ATTRIB_DECLARE)

                steppableDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', EXTRA_ATTRIB_DECLARE,

                                                  steppableDeclarationCode)



            else:

                steppableDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', '', steppableDeclarationCode)

            insertedCode += steppableDeclarationCode

        self.addCodeToSwigFile(_features['mainSwigFile'], insertedCode, insertedCodeRegex4, insertionLocationLabel4)

        # CMakeListe file in the CompuCellPython directory

        insertedCodeHeader5 = '#' + _features['Module'] + '_autogenerated\n'

        insertedCodeRegex5 = '^[\s]*#[\s]*' + _features['Module'] + '_autogenerated'

        insertionLocationLabel5 = '#AutogeneratedModules'

        insertedCode = insertedCodeHeader5

        insertedCode += _features['Module'] + 'Shared\n'

        self.addCodeToSwigFile(_features['cmakeSwigFile'], insertedCode, insertedCodeRegex5, insertionLocationLabel5)

    def modifySwigFiles(self, _features):

        if _features['mainSwigFile'] == '' or _features['declarationsSwigFile'] == '' or _features[

            'declarationsSwigFile'] == '' or _features['cmakeSwigFile'] == '':
            message = "Could not succesfully locate SWIG files. <br> Please make sure you have necessary permissions and that file exist. You will need to add newly generated module to SWIG filesmanually"

            QtGui.QMessageBox.information(self.__ui, "Problem with SWIG files", message, QtGui.QMessageBox.Ok)

            return

        insertedCodeRegex = '^[\s]*//[\s]*' + _features['Module'] + '_autogenerated'

        insertedCodeHeader = '//' + _features['Module'] + '_autogenerated'

        # mainSwigFile ##############################################

        insertedCode = insertedCodeHeader + '\n'

        insertedCode += '#define ' + _features['Module'].upper() + '_EXPORT\n'

        insertionLocationLabel = '//AutogeneratedModules'

        self.addCodeToSwigFile(_features['mainSwigFile'], insertedCode, insertedCodeRegex, insertionLocationLabel)

        # declarationsSwigFile ##############################################

        insertionLocationLabel = '//AutogeneratedModules1'

        insertedCodeRegex1 = insertedCodeRegex + '1'

        insertedCode = insertedCodeHeader + '1\n'

        insertedCode += '#define ' + _features['Module'].upper() + '_EXPORT\n'

        self.addCodeToSwigFile(_features['declarationsSwigFile'], insertedCode, insertedCodeRegex1,

                               insertionLocationLabel)

        insertionLocationLabel = '//AutogeneratedModules2'

        insertedCode = insertedCodeHeader + '2\n'

        insertedCodeRegex2 = insertedCodeRegex + '2'

        pluginDeclarationCode = '''EXTRA_ATTRIB_DECLARE

%include <CompuCell3D/plugins/PLUGIN_NAME_CORE/PLUGIN_NAME_COREPlugin.h>



%inline %{

 PLUGIN_NAME_COREPlugin * getPLUGIN_NAME_COREPlugin(){

      return (PLUGIN_NAME_COREPlugin *)Simulator::pluginManager.get("PLUGIN_NAME_CORE");

   }



%}

'''

        steppableDeclarationCode = '''EXTRA_ATTRIB_DECLARE

%include <CompuCell3D/steppables/STEPPABLE_NAME_CORE/STEPPABLE_NAME_CORE.h>



%inline %{

 STEPPABLE_NAME_CORE * getSTEPPABLE_NAME_CORE(){

      return (STEPPABLE_NAME_CORE *)Simulator::steppableManager.get("STEPPABLE_NAME_CORE");

   }



%}

'''

        if 'Plugin' in list(_features.keys()):

            PLUGIN_NAME_CORE = _features['Plugin']

            pluginDeclarationCode = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, pluginDeclarationCode)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_DECLARE = '''%include <CompuCell3D/plugins/PLUGIN_NAME_CORE/PLUGIN_NAME_COREData.h>

%template (PLUGIN_NAME_COREDataAccessorTemplate) BasicClassAccessor<PLUGIN_NAME_COREData>; //necessary to get PLUGIN_NAME_COREData accessor working in Python

'''

                EXTRA_ATTRIB_DECLARE = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, EXTRA_ATTRIB_DECLARE)

                pluginDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', EXTRA_ATTRIB_DECLARE, pluginDeclarationCode)

                print('replacing EXTRA_ATTRIB_DECLARE with', EXTRA_ATTRIB_DECLARE)



            else:

                pluginDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', '', pluginDeclarationCode)

            insertedCode += pluginDeclarationCode



        elif 'Steppable' in list(_features.keys()):

            STEPPABLE_NAME_CORE = _features['Steppable']

            steppableDeclarationCode = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, steppableDeclarationCode)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_DECLARE = '''%include <CompuCell3D/steppables/STEPPABLE_NAME_CORE/STEPPABLE_NAME_COREData.h>

%template (STEPPABLE_NAME_COREDataAccessorTemplate) BasicClassAccessor<STEPPABLE_NAME_COREData>; //necessary to get STEPPABLE_NAME_COREData accessor working in Python

'''

                EXTRA_ATTRIB_DECLARE = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, EXTRA_ATTRIB_DECLARE)

                steppableDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', EXTRA_ATTRIB_DECLARE,

                                                  steppableDeclarationCode)



            else:

                steppableDeclarationCode = re.sub('EXTRA_ATTRIB_DECLARE', '', steppableDeclarationCode)

            insertedCode += steppableDeclarationCode

        self.addCodeToSwigFile(_features['declarationsSwigFile'], insertedCode, insertedCodeRegex2,

                               insertionLocationLabel)

        # includesSwigFile ##############################################

        insertionLocationLabel = '//AutogeneratedModules'

        insertedCode = insertedCodeHeader + '\n'

        pluginIncludeCode = '''EXTRA_ATTRIB_INCLUDE

#include <CompuCell3D/plugins/PLUGIN_NAME_CORE/PLUGIN_NAME_COREPlugin.h>

'''

        steppableIncludeCode = '''EXTRA_ATTRIB_INCLUDE

#include <CompuCell3D/steppables/STEPPABLE_NAME_CORE/STEPPABLE_NAME_CORE.h>

'''

        if 'Plugin' in list(_features.keys()):

            PLUGIN_NAME_CORE = _features['Plugin']

            pluginIncludeCode = re.sub('PLUGIN_NAME_CORE', PLUGIN_NAME_CORE, pluginIncludeCode)

            print('_features=', _features)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_INCLUDE = '#include <CompuCell3D/plugins/' + PLUGIN_NAME_CORE + '/' + PLUGIN_NAME_CORE + 'Data.h>'

                pluginIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', EXTRA_ATTRIB_INCLUDE, pluginIncludeCode)

            else:

                pluginIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', '', pluginIncludeCode)

            insertedCode += pluginIncludeCode

        elif 'Steppable' in list(_features.keys()):

            STEPPABLE_NAME_CORE = _features['Steppable']

            steppableIncludeCode = re.sub('STEPPABLE_NAME_CORE', STEPPABLE_NAME_CORE, steppableIncludeCode)

            if 'ExtraAttribute' in list(_features.keys()):

                EXTRA_ATTRIB_INCLUDE = '#include <CompuCell3D/steppables/' + STEPPABLE_NAME_CORE + '/' + STEPPABLE_NAME_CORE + 'Data.h>'

                steppableIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', EXTRA_ATTRIB_INCLUDE, steppableIncludeCode)

            else:

                steppableIncludeCode = re.sub('EXTRA_ATTRIB_INCLUDE', '', steppableIncludeCode)

            insertedCode += steppableIncludeCode

        self.addCodeToSwigFile(_features['includesSwigFile'], insertedCode, insertedCodeRegex, insertionLocationLabel)

        # CMakeListe file in the CompuCellPython directory

        insertedCodeHeader = '#' + _features['Module'] + '_autogenerated\n'

        insertedCodeRegex = '^[\s]*#[\s]*' + _features['Module'] + '_autogenerated'

        insertionLocationLabel = '#AutogeneratedModules'

        insertedCode = insertedCodeHeader

        insertedCode += _features['Module'] + 'Shared\n'

        self.addCodeToSwigFile(_features['cmakeSwigFile'], insertedCode, insertedCodeRegex, insertionLocationLabel)

    def addCodeToSwigFile(self, _swigFileName, _insertedCode, _insertedCodeRegex, _insertionLocationLabel, startRow=0,

                          startColumn=0):

        fileList = []

        if str(_swigFileName) == '':

            return

        else:

            fileList.append(_swigFileName)

        self.__ui.loadFiles(fileList)

        editor = self.__ui.getActiveEditor()

        # insertedCode='//'+_features['Module']+'_autogenerated\n'

        # insertedCode+='#define '+_features['Module'].upper()+'_EXPORT'

        # insertedCodeRegex='^[\s]*//[\s]*'+_features['Module']+'_autogenerated'

        # localize insertion point first

        foundFlag = editor.findFirst(_insertionLocationLabel, False, True, False, False, True, startRow, startColumn,

                                     False)

        if foundFlag:

            currentLine, currentIndex = editor.getCursorPosition()  # record current cursor position

            foundRegexFlag = editor.findFirst(_insertedCodeRegex, True, True, False, False, True, currentLine,

                                              currentIndex, False)

            if foundRegexFlag:
                print("Generated is already in the file")

                return editor.getCursorPosition()

            editor.insertAt(_insertedCode, currentLine + 1, 0)

            editor.ensureLineVisible(currentLine + 10)

            self.__ui.save()  # saves active editor

            return currentLine, currentIndex

        else:

            print('Could not locate insertion point')

            return editor.getCursorPosition()

    def writeTextToFile(self, _fileName, _text=''):

        try:

            file = open(_fileName, "w")

            file.write("%s" % _text)

            file.close()

        except IOError as e:

            message = "Could not write to file %s. <br> Please make sure you have necessary permissions" % _fileName

            QtGui.QMessageBox.information(self.__ui, "Could not write to file", message, QtGui.QMessageBox.Ok)

    def makeDirectory(self, fullDirPath):

        """

            This fcn attmpts to make directory or if directory exists it will do nothing

        """

        # dirName=os.path.dirname(fullDirPath)

        try:

            mkpath(fullDirPath)

        except:

            raise IOError

        return

    def __insertSnippet(self, _snippetName):

        # print "GOT REQUEST FOR SNIPPET ",_snippetName

        snippetNameStr = str(_snippetName)

        text = self.snippetDictionary[str(_snippetName)].snippet_text

        editor = self.__ui.getCurrentEditor()

        curFileName = str(self.__ui.getCurrentDocumentName())

        basename, ext = os.path.splitext(curFileName)

        if ext != ".cpp" and ext != ".h" and ext != ".cxx" and ext != ".hpp":
            QMessageBox.warning(self.__ui, "C++/C++header files only", "C++ code snippets work only for C++ files")

            return

        curLine = 0

        curCol = 0

        if snippetNameStr == "Cell Attributes Add Dictionary To Cells" or snippetNameStr == "Cell Attributes Add List To Cells":

            curLine, curCol = self.findEntryLineForCellAttributes(editor)

            if curLine == -1:
                QMessageBox.warning(self.__ui, "Could not find insert point",

                                    "Could not find insert point for code cell attribute code. Please make sure you are editing CC3D Main Python script")

                return

        elif snippetNameStr.startswith("Extra Fields"):

            # this function potentially inserts new text - will have to get new cursor position after that
            self.includeExtraFieldsImports(editor)

            curLine, curCol = editor.getCursorPosition()



        else:

            curLine, curCol = editor.getCursorPosition()

        indentationLevels, indentConsistency = self.findIndentationForSnippet(editor, curLine)

        print("indentationLevels=", indentationLevels, " consistency=", indentConsistency)

        textLines = text.splitlines(True)

        for i in range(len(textLines)):
            textLines[i] = ' ' * editor.indentationWidth() * indentationLevels + textLines[i]

        indentedText = ''.join(textLines)

        currentLineText = str(editor.text(curLine))

        nonwhitespaceFound = re.match(self.nonwhitespaceRegex, currentLineText)

        print("currentLineText=", currentLineText, " nonwhitespaceFound=", nonwhitespaceFound)

        editor.beginUndoAction()  # begining of action sequence

        if nonwhitespaceFound:  # we only add new line if the current line has someting in it other than whitespaces

            editor.insertAt("\n", curLine, editor.lineLength(curLine))

            curLine += 1

        editor.insertAt(indentedText, curLine, 0)

        # editor.insertAt(text,curLine,0)

        editor.endUndoAction()  # end of action sequence

        # highlighting inserted text

        editor.findFirst(indentedText, False, False, False, True, curLine)

        lineFrom, colFrom, lineTo, colTo = editor.getSelection()

    def includeExtraFieldsImports(self, _editor):

        playerFromImportRegex = re.compile('^[\s]*from[\s]*PlayerPython[\s]*import[\s]*\*')

        compuCellSetupImportRegex = re.compile('^[\s]*import[\s]*CompuCellSetup')

        curLine, curCol = _editor.getCursorPosition()

        foundPlayerImports = None

        foundCompuCellSetupImport = None

        for line in range(curLine, -1, -1):

            text = str(_editor.text(line))

            foundPlayerImports = re.match(playerFromImportRegex, text)

            if foundPlayerImports:
                break

        for line in range(curLine, -1, -1):

            text = str(_editor.text(line))

            foundCompuCellSetupImport = re.match(compuCellSetupImportRegex, text)

            if foundCompuCellSetupImport:
                break

        if not foundCompuCellSetupImport:
            _editor.insertAt("import CompuCellSetup\n", 0, 0)

        if not foundPlayerImports:
            _editor.insertAt("from PlayerPython import * \n", 0, 0)

    def findEntryLineForCellAttributes(self, _editor):

        getCoreSimulationObjectsRegex = re.compile('^[\s]*sim.*CompuCellSetup\.getCoreSimulationObjects')

        text = ''

        foundLine = -1

        for line in range(_editor.lines()):

            text = str(_editor.text(line))

            getCoreSimulationObjectsRegexFound = re.match(getCoreSimulationObjectsRegex,

                                                          text)  # \S - non -white space \swhitespace

            if getCoreSimulationObjectsRegexFound:  # line with getCoreSimulationObjectsRegex

                foundLine = line

                break

        if foundLine >= 0:

            # check for comment code  - #add extra attributes here

            attribCommentRegex = re.compile('^[\s]*#[\s]*add[\s]*extra[\s]*attrib')

            for line in range(foundLine, _editor.lines()):

                text = str(_editor.text(line))

                attribCommentFound = re.match(attribCommentRegex, text)

                if attribCommentFound:
                    foundLine = line

                    return foundLine, 0

            return foundLine, 0

        return -1, -1

    def findIndentationForSnippet(self, _editor, _line):

        # nonwhitespaceRegex=re.compile('^[\s]*[\S]+')

        # commentRegex=re.compile('^[\s]*#')

        # defFunRegex=re.compile('^[\s]*def')

        # blockStatementRegex=re.compile(':[\s]*$') # block statement - : followed by whitespaces at the end of the line

        # blockStatementWithCommentRegex=re.compile(':[\s]*[#]+[\s\S]*$') # block statement - : followed by whitespaces at the end of the line

        # ':[\s]*$|:[\s]*[#]+[\s\S*]$'

        # ':[\s]*[\#+[\s\S*]$'

        # ':[\s]*[#]+[\s\S]*' -  works

        text = ''

        for line in range(_line, -1, -1):

            text = str(_editor.text(line))

            nonwhitespaceFound = re.match(self.nonwhitespaceRegex, text)  # \S - non -white space \swhitespace

            if nonwhitespaceFound:  # once we have line with non-white spaces we check if this is non comment line

                commentFound = re.match(self.commentRegex, text)

                if not commentFound:  # if it is 'regular' line we check if this line is begining of a block statement

                    blockStatementFound = re.search(self.blockStatementRegex, text)

                    blockStatementWithCommentFound = re.search(self.blockStatementWithCommentRegex, text)

                    # print "blockStatementFound=",blockStatementFound," blockStatementWithCommentFound=",blockStatementWithCommentFound

                    if blockStatementFound or blockStatementWithCommentFound:  # we insert code snippet increasing indentation after begining of block statement

                        # print "_editor.indentationWidth=",_editor.indentationWidth

                        indentationLevels = int((_editor.indentation(

                            line) + _editor.indentationWidth()) / _editor.indentationWidth())

                        indentationLevelConsistency = not (_editor.indentation(

                            line) + _editor.indentationWidth()) % _editor.indentationWidth()  # if this is non-zero indentations in the code are inconsistent

                        if not indentationLevelConsistency:
                            QMessageBox.warning(self.__ui, "Possible indentation problems",

                                                "Please position code snippet manually using TAB (indent) Shift+Tab (Unindent)")

                            return 0, indentationLevelConsistency

                        return indentationLevels, indentationLevelConsistency



                    else:  # we use indentation of the previous line

                        indentationLevels = int((_editor.indentation(line)) / _editor.indentationWidth())

                        indentationLevelConsistency = not (_editor.indentation(

                            line)) % _editor.indentationWidth()  # if this is non-zero indentations in the code are inconsistent

                        if not indentationLevelConsistency:
                            QMessageBox.warning(self.__ui, "Possible indentation problems",

                                                "Please position code snippet manually using TAB (indent) Shift+Tab (Unindent)")

                            return 0, indentationLevelConsistency

                        return indentationLevels, indentationLevelConsistency

        return 0, 0
