from cc3d.twedit5.twedit.utils.global_imports import *
import ui_configurationdlg
import sys

MAC = "qt_mac_set_native_menubar" in dir()


class ConfigurationDlg(QDialog, ui_configurationdlg.Ui_ConfigurationDlg):

    # signals

    # gotolineSignal = QtCore.pyqtSignal( ('int',))

    def __init__(self, _currentEditor=None, parent=None):

        super(ConfigurationDlg, self).__init__(parent)

        self.editorWindow = parent

        self.editorWindowConfiguration = self.editorWindow.configuration

        pluginAutoloadData = self.editorWindowConfiguration.pluginAutoloadData()

        print(' __init__ pluginAutoloadData=', pluginAutoloadData)

        self.currentEditor = _currentEditor

        # there are issues with Drawer dialog not getting focus when being displayed on linux

        # they are also not positioned properly so, we use "regular" windows 

        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        # self.gotolineSignal.connect(self.editorWindow.goToLine)

        self.setupUi(self)

        # self.connect(self.pluginsLW,SIGNAL('currentItemChanged(QListWidgetItem *,QListWidgetItem *)'),self.updatePluginInfoOptions)

        self.pluginsLW.currentItemChanged.connect(self.updatePluginInfoOptions)

        self.loadOnStartupCHB.clicked.connect(self.processLoadOnStartupChange)

        # self.connect(self.loadOnStartupCHB,SIGNAL('clicked(bool)'),self.processLoadOnStartupChange)

        self.populatePluginsLW()

        self.populateThemeCB()

        if not MAC:
            self.cancelButton.setFocusPolicy(Qt.NoFocus)

        self.updateUi()

        self.themeCB.currentIndexChanged.connect(self.changeEditorWindowTheme)

    def changeEditorWindowTheme(self):

        theme = self.themeCB.currentText()

        self.editorWindow.applyTheme(theme)

    def populateThemeCB(self):

        themeNameList = self.editorWindow.themeManager.getThemeNames()

        for themeName in themeNameList:
            self.themeCB.addItem(themeName)

    def updatePluginInfoOptions(self, _currentItem, _previousItem):

        print('updatePluginInfoOptions')

        pm = self.editorWindow.pm

        moduleName = str(_currentItem.text())

        moduleName.rstrip()

        bpd = pm.get_basic_plugin_data(moduleName)

        isActive = pm.is_plugin_active(moduleName)

        if bpd:
            self.pluginsTE.setText(bpd.longDescription)

        # setting load/unload buttons

        self.loadPB.setEnabled(not isActive)

        self.unloadPB.setEnabled(bpd.deactivateable and isActive)

        loadOnStartup = bpd.autoactivate

        # autoactivate option from plugin can be overridden by user setting

        pluginAutoloadData = self.editorWindowConfiguration.pluginAutoloadData()  # dictionary in the Configuration.py storing data abuot which plugin whould be loaded

        print('\n\n\n\n\n pluginAutoloadData=', pluginAutoloadData)

        try:

            loadOnStartup = pluginAutoloadData[moduleName]

        except LookupError as e:

            print('COULD NOT FIND moduleName=', moduleName, ' in ', pluginAutoloadData)

        self.loadOnStartupCHB.setChecked(loadOnStartup)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_loadPB_clicked(self):

        pm = self.editorWindow.pm

        currentItem = self.pluginsLW.currentItem()

        moduleName = str(currentItem.text())

        moduleName.rstrip()

        bpd = pm.get_basic_plugin_data(moduleName)

        # plugins listed in the list widget are in fact modules which appear to be instantiable plugins 

        # pm.activatePlugin(moduleName)        

        pm.load_plugin(moduleName, True)

        if pm.is_plugin_active(moduleName):
            self.loadPB.setEnabled(False)

            self.unloadPB.setEnabled(bpd.deactivateable)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_unloadPB_clicked(self):

        pm = self.editorWindow.pm

        currentItem = self.pluginsLW.currentItem()

        moduleName = str(currentItem.text())

        moduleName.rstrip()

        bpd = pm.get_basic_plugin_data(moduleName)

        # plugins listed in the list widget are in fact modules which appear to be instantiable plugins 

        pm.unload_plugin(moduleName)

        if not pm.is_plugin_active(moduleName):
            self.loadPB.setEnabled(True)

            self.unloadPB.setEnabled(False)

    def processLoadOnStartupChange(self, _flag):

        print('_flag=', _flag)

        currentItem = self.pluginsLW.currentItem()

        pluginName = str(currentItem.text())

        pluginName.rstrip()

        self.editorWindowConfiguration.setPluginAutoloadData(pluginName, _flag)

    def populatePluginsLW(self):

        # TODO enable it

        return

        pm = self.editorWindow.pm

        moduleList = pm.get_available_modules()  # this is alist of python modules (not python objects!) whci are in the PLugins directory and appear to be vaild plugins

        self.pluginsLW.addItems(moduleList)

        if self.pluginsLW.count():
            self.pluginsLW.setCurrentRow(0)

    @pyqtSlot()  # signature of the signal emited by the button
    def on_okButton_clicked(self):

        self.findChangedConfigs()

        self.close()

    def findChangedConfigs(self):

        configuration = self.editorWindow.configuration

        configuration.updatedConfigs = {}

        if configuration.setting("UseTabSpaces") != self.tabSpacesCheckBox.isChecked():
            configuration.updatedConfigs["UseTabSpaces"] = self.tabSpacesCheckBox.isChecked()

        if configuration.setting("TabSpaces") != self.spacesSpinBox.value():
            configuration.updatedConfigs["TabSpaces"] = self.spacesSpinBox.value()

        if configuration.setting("DisplayLineNumbers") != self.lineNumberCheckBox.isChecked():
            configuration.updatedConfigs["DisplayLineNumbers"] = self.lineNumberCheckBox.isChecked()

        if configuration.setting("FoldText") != self.foldTextCheckBox.isChecked():
            configuration.updatedConfigs["FoldText"] = self.foldTextCheckBox.isChecked()

        if configuration.setting("TabGuidelines") != self.tabGuidelinesCheckBox.isChecked():
            configuration.updatedConfigs["TabGuidelines"] = self.tabGuidelinesCheckBox.isChecked()

        if configuration.setting("DisplayWhitespace") != self.whiteSpaceCheckBox.isChecked():
            configuration.updatedConfigs["DisplayWhitespace"] = self.whiteSpaceCheckBox.isChecked()

        if configuration.setting("DisplayEOL") != self.eolCheckBox.isChecked():
            configuration.updatedConfigs["DisplayEOL"] = self.eolCheckBox.isChecked()

        if configuration.setting("WrapLines") != self.wrapLinesCheckBox.isChecked():
            configuration.updatedConfigs["WrapLines"] = self.wrapLinesCheckBox.isChecked()

        if configuration.setting("ShowWrapSymbol") != self.showWrapSymbolCheckBox.isChecked():
            configuration.updatedConfigs["ShowWrapSymbol"] = self.showWrapSymbolCheckBox.isChecked()

        if configuration.setting("RestoreTabsOnStartup") != self.restoreTabsCheckBox.isChecked():
            configuration.updatedConfigs["RestoreTabsOnStartup"] = self.restoreTabsCheckBox.isChecked()

        if configuration.setting("EnableAutocompletion") != self.autocompletionCheckBox.isChecked():
            configuration.updatedConfigs["EnableAutocompletion"] = self.autocompletionCheckBox.isChecked()

        if configuration.setting("EnableQuickTextDecoding") != self.quickTextDecodingCB.isChecked():
            configuration.updatedConfigs["EnableQuickTextDecoding"] = self.quickTextDecodingCB.isChecked()

        if configuration.setting("AutocompletionThreshold") != self.autocompletionSpinBox.value():
            configuration.updatedConfigs["AutocompletionThreshold"] = self.autocompletionSpinBox.value()

        if configuration.setting("BaseFontName") != self.fontComboBox.currentText():
            configuration.updatedConfigs["BaseFontName"] = self.fontComboBox.currentText()

        if configuration.setting("BaseFontSize") != self.fontSizeComboBox.currentText():
            configuration.updatedConfigs["BaseFontSize"] = self.fontSizeComboBox.currentText()

        if configuration.setting("Theme") != self.themeCB.currentText():
            configuration.updatedConfigs["Theme"] = self.themeCB.currentText()

        # store changed values in settings

        for key in list(configuration.updatedConfigs.keys()):
            configuration.setSetting(key, configuration.updatedConfigs[key])

    def updateUi(self):

        configuration = self.editorWindow.configuration

        self.tabSpacesCheckBox.setChecked(configuration.setting("UseTabSpaces"))

        self.spacesSpinBox.setValue(configuration.setting("TabSpaces"))

        self.lineNumberCheckBox.setChecked(configuration.setting("DisplayLineNumbers"))

        self.foldTextCheckBox.setChecked(configuration.setting("FoldText"))

        self.tabGuidelinesCheckBox.setChecked(configuration.setting("TabGuidelines"))

        self.whiteSpaceCheckBox.setChecked(configuration.setting("DisplayWhitespace"))

        self.eolCheckBox.setChecked(configuration.setting("DisplayEOL"))

        self.wrapLinesCheckBox.setChecked(configuration.setting("WrapLines"))

        if not self.wrapLinesCheckBox.isChecked():
            self.showWrapSymbolCheckBox.setEnabled(False)

        self.showWrapSymbolCheckBox.setChecked(configuration.setting("ShowWrapSymbol"))

        self.restoreTabsCheckBox.setChecked(configuration.setting("RestoreTabsOnStartup"))

        self.autocompletionCheckBox.setChecked(configuration.setting("EnableAutocompletion"))

        self.quickTextDecodingCB.setChecked(configuration.setting("EnableQuickTextDecoding"))

        self.autocompletionSpinBox.setValue(configuration.setting("AutocompletionThreshold"))

        # not ideal solution but should work

        baseFontName = configuration.setting("BaseFontName")

        for idx in range(self.fontComboBox.count()):

            if baseFontName == self.fontComboBox.itemText(idx):
                self.fontComboBox.setCurrentIndex(idx)

                break

        baseFontSize = configuration.setting("BaseFontSize")

        for idx in range(self.fontSizeComboBox.count()):

            if baseFontSize == self.fontSizeComboBox.itemText(idx):
                self.fontSizeComboBox.setCurrentIndex(idx)

                break

        # not ideal solution but should work

        themeName = configuration.setting("Theme")

        for idx in range(self.themeCB.count()):

            if themeName == self.themeCB.itemText(idx):
                self.themeCB.setCurrentIndex(idx)

                break
