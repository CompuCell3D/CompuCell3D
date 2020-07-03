"""

Module implementing the Plugin Manager.

"""
import imp
import traceback
import cc3d.twedit5.twedit as twedit
from cc3d.twedit5.PluginManager.PluginExceptions import *


# this class will store basic plugin data
class BasicPluginData(object):

    def __init__(self, _name='', _fileName=''):

        self.name = _name

        self.fileName = _fileName

        self.author = ''

        self.autoactivate = False

        self.deactivateable = False

        self.version = ''

        self.className = ''

        self.packageName = ''

        self.shortDescription = ''

        self.longDescription = ''

    def __str__(self):

        attributeNames = ['name', 'fileName', 'author', 'autoactivate', 'deactivateable', 'version', 'className',
                          'packageName', 'shortDescription', 'longDescription']

        retStr = ''

        for attributeName in attributeNames:

            try:

                retStr += attributeName + '=' + str(getattr(self, attributeName)) + '\n'

            except TypeError as e:

                retStr += attributeName + '=' + getattr(self, attributeName) + '\n'

        return retStr


class PluginManager(QObject):
    """

    Class implementing the Plugin Manager.

    



    """

    def __init__(self, parent=None, doLoadPlugins=True, develPlugin=None):

        """

        Constructor

        

        The Plugin Manager deals with three different plugin directories.

        The first is the one, that is part of eric4 (eric4/Plugins). The

        second one is the global plugin directory called 'eric4plugins', 

        which is located inside the site-packages directory. The last one

        is the user plugin directory located inside the .eric4 directory

        of the users home directory.

        

        @param parent reference to the parent object (QObject)

        @keyparam doLoadPlugins flag indicating, that plugins should 

            be loaded (boolean)

        @keyparam develPlugin filename of a plugin to be loaded for 

            development (string)

        """

        QObject.__init__(self, parent)

        self.__ui = parent

        self.__pluginQueries = {}  #

        # self.__inactiveModules={}

        self.__activePlugins = {}

        self.__availableModules = {}

        # self.__inactivePlugins = {}

        # self.__activeModules = {}

        self.__failedModules = {}

        # self.__onDemandInactiveModules={}

        self.__modulesCount = 0

        osp_dir = os.path.dirname
        self.tweditRootPath = osp_dir(osp_dir(twedit.__file__))
        # apisPath = os.path.join(self.tweditRootPath, "APIs")

        # self.tweditRootPath = os.path.dirname(Configuration.__file__)

        self.pluginPath = os.path.join(self.tweditRootPath, "Plugins")

        # check if it exists

        if not os.path.exists(self.pluginPath):
            # when packaging on Windows with pyinstaller the path to executable is accesible via sys.executable as Python is bundled with the distribution

            # os.path.dirname(Configuration.__file__) returned by pyInstaller will not work without some modifications so it is best tu use os.path.dirname(sys.executable) approach

            self.tweditRootPath = os.path.dirname(sys.executable)

            self.pluginPath = os.path.join(self.tweditRootPath, "Plugin")

        # self.__checkPluginsDownloadDirectory()

        pluginModules = self.getPluginModules(self.pluginPath)

        self.__insertPluginsPaths()

        print('pluginModules=', pluginModules)

        for pluginName in pluginModules:
            self.queryPlugin(pluginName, self.pluginPath)

            # checking out which plugins were succesfully querried

        # discovered plugin names are storred in pluginModules list

        defined_plugin_module_name_set = set(pluginModules)

        # print 'defined_plugin_module_name_set=',defined_plugin_module_name_set

        # all plugins from discovered list are querried to check if they load correctly. If the do not

        # self.__pluginQueries will not contain those plugins for which query failed

        queried_plugin_module_name_set = set(self.__pluginQueries.keys())

        # print 'queried_plugin_module_name_set=',queried_plugin_module_name_set

        problematic_module_set = defined_plugin_module_name_set - queried_plugin_module_name_set

        # print 'problematic_module_set=',problematic_module_set

        warning_string = 'The following plugins have errors and will not be loaded: '

        for problematic_plugin_name in problematic_module_set:
            warning_string += problematic_plugin_name + ", "

        # removing trailing ", "

        warning_string = warning_string[:-2]

        if len(problematic_module_set):
            self.__ui.display_popup_message(message_title='Module Error', message_text=warning_string)

            # print 'WARNING  '+warning_string

        # print '\n\n\n\n self.__pluginQueries=',self.__pluginQueries

        # load and activate plugins

        # for pluginName in pluginModules:        

        for pluginName, bpd in list(self.__pluginQueries.items()):
            # bpd=self.__pluginQueries[pluginName]

            print('************PLUGIN NAME=', pluginName)

            print(bpd)

            self.loadPlugin(pluginName)

    def runForAllPlugins(self, _functionName, _argumentDict):

        '''attempts to run function (_functionName) with arguments given by _argumentDict for all plugins

        '''

        for pluginName, plugin in self.__activePlugins.items():

            try:

                function = getattr(plugin, _functionName)

                function(_argumentDict)

            except:

                print('PLUGIN: ', pluginName, ' COULD NOT APPLY ', _functionName, ' with arguments=', _argumentDict)

    def getActivePlugin(self, _name):

        try:

            return self.__activePlugins[_name]

        except LookupError as e:

            return None

    def getPluginModules(self, pluginPath):

        """

        Public method to get a list of plugin modules.

        

        @param pluginPath name of the path to search (string)

        @return list of plugin module names (list of string)

        """

        pluginFiles = [f[:-3] for f in os.listdir(pluginPath) \
 \
                       if self.isValidPluginName(f)]

        return pluginFiles[:]

    def isValidPluginName(self, pluginName):

        """

        Public methode to check, if a file name is a valid plugin name.

        

        Plugin modules must start with "Plugin" and have the extension ".py".

        

        @param pluginName name of the file to be checked (string)

        @return flag indicating a valid plugin name (boolean)

        """

        return pluginName.startswith("Plugin") and pluginName.endswith(".py")

    def __insertPluginsPaths(self):

        """

        Private method to insert the valid plugin paths intos the search path.

        """

        sys.path.insert(2, self.pluginPath)

    def queryPlugin(self, name, directory, reload_=False):

        """

        Public method to preload a plugin module and run few queries like plugin name, author, descriptiopn etc 

                

        @param name name of the module to be loaded (string)

        @param directory name of the plugin directory (string)

        @param reload_ flag indicating to reload the module (boolean)

        """

        attributeNames = ['author', 'autoactivate', 'deactivateable', 'version', 'className', 'packageName',
                          'shortDescription', 'longDescription']

        try:

            print('plugin name=', name)

            fname = "%s.py" % os.path.join(directory, name)

            print("\n\n\n ##################loading source for ", fname)

            module = imp.load_source(name, fname)

            bpd = BasicPluginData(name, fname)

            for attributeName in attributeNames:

                if hasattr(module, "autoactivate"):
                    setattr(bpd, attributeName, getattr(module, attributeName))

            self.__pluginQueries[name] = bpd



        except Exception as err:

            module = imp.new_module(name)

            module.error = "\n\n\n\n\n Module failed to load. Error: %s" % str(err)

            print("Error loading plugin module:", name)

            print(str(err))

            traceback.print_exc(file=sys.stdout)

            # sys.exit()

    def loadPlugin(self, name, forceActivate=False):

        """

        Public method to load aand activate plugin module.

        

        @param name name of the p[lugin to load  - used as a key to self.__pluginQueries

        """

        try:

            bpd = self.__pluginQueries[name]

        except LookupError as e:

            return

        try:

            name = bpd.name

            fname = bpd.fileName

            print('LOADING ', name, 'from ', fname)

            # fname = "%s.py" % os.path.join(directory, name)

            print("loading source for ", fname)

            module = imp.load_source(name, fname)

            module.pluginModuleName = name

            module.pluginModuleFilename = fname

            self.__modulesCount += 1

            # self.__inactiveModules[name] = module

            self.__availableModules[name] = module

            # checking whether user has specific load on startup setting for this plugin

            configuration = self.__ui.configuration

            pluginAutoloadData = configuration.pluginAutoloadData()

            autoloadFlag = bpd.autoactivate

            try:

                autoloadFlag = pluginAutoloadData[name]

            except LookupError as e:

                pass

            if autoloadFlag:

                self.activatePlugin(name)  # instantiates plugin object based on code stored in module

            elif forceActivate:

                self.activatePlugin(name)  # instantiates plugin object based on code stored in module

            return module



        except Exception as err:

            module = imp.new_module(name)

            module.error = "Module failed to load. Error: %s" % str(err)

            self.__failedModules[name] = module

            print("Error loading plugin module:", name)

            print("\n\n\n", str(err), '\n\n\n ')

            traceback.print_exc(file=sys.stdout)

            # sys.exit()

    def isPluginActive(self, name):

        return name in list(self.__activePlugins.keys())

    def getBasicPluginData(self, name):

        try:

            return self.__pluginQueries[name]

        except LookupError as e:

            return None

    def getAvailableModules(self):

        return list(self.__availableModules.keys())

    def unloadPlugin(self, name):

        """

        Public method to unload and deactivate plugin module.

        

        @param name name of the module to be unloaded (string)

        @param directory name of the plugin directory (string)

        @return flag indicating success (boolean)

        """

        if name in self.__activePlugins:
            # self.__activeModules[name].eric4PluginModuleFilename == fname:

            self.deactivatePlugin(name)

            self.__availableModules[name] = self.__pluginQueries[
                name]  # replacing module (potentially arge object with simple bdp plugin descriptor)

        self.__modulesCount -= 1

        return True

    def unloadPlugins(self):

        unloadedPluginNames = []

        for pluginName in list(self.__activePlugins.keys()):
            self.__activePlugins[pluginName].deactivate()

            unloadedPluginNames.append(pluginName)

        for pluginName in unloadedPluginNames:
            del self.__activePlugins[pluginName]

    def removePluginFromSysModules(self, pluginName, package, internalPackages):

        """

        Public method to remove a plugin and all related modules from sys.modules.

        

        @param pluginName name of the plugin module (string)

        @param package name of the plugin package (string)

        @param internalPackages list of intenal packages (list of string)

        @return flag indicating the plugin module was found in sys.modules (boolean)

        """

        packages = [package] + internalPackages

        found = False

        if not package:
            package = "__None__"

        for moduleName in list(sys.modules.keys())[:]:

            if moduleName == pluginName or moduleName.split(".")[0] in packages:
                found = True

                del sys.modules[moduleName]

        return found

    def initOnDemandPlugins(self):

        """

        Public method to create plugin objects for all on demand plugins.

        

        Note: The plugins are not activated.

        """

        names = sorted(self.__onDemandInactiveModules.keys())

        for name in names:
            self.initOnDemandPlugin(name)

    def initOnDemandPlugin(self, name):

        """

        Public method to create a plugin object for the named on demand plugin.

        

        Note: The plugin is not activated.

        """

        try:

            try:

                module = self.__onDemandInactiveModules[name]

            except KeyError:

                return None

            if not self.__canActivatePlugin(module):
                raise PluginActivationError(module.eric4PluginModuleName)

            version = getattr(module, "version")

            className = getattr(module, "className")

            pluginClass = getattr(module, className)

            pluginObject = None

            if name not in self.__onDemandInactivePlugins:
                pluginObject = pluginClass(self.__ui)

                pluginObject.eric4PluginModule = module

                pluginObject.eric4PluginName = className

                pluginObject.eric4PluginVersion = version

                self.__onDemandInactivePlugins[name] = pluginObject

        except PluginActivationError:

            return None

    def activatePlugins(self):

        """

        Public method to activate all plugins having the "autoactivate" attribute

        set to True.

        """

        ial = Preferences.Prefs.settings.value(self.__inactivePluginsKey)

        if ial.isValid():

            savedInactiveList = ial.toStringList()

        else:

            savedInactiveList = None

        if self.__develPluginName is not None and savedInactiveList is not None:
            savedInactiveList.removeAll(self.__develPluginName)

        names = list(self.__inactiveModules.keys())

        names.sort()

        for name in names:

            if savedInactiveList is None or name not in savedInactiveList:
                self.activatePlugin(name)

        self.emit(SIGNAL("allPlugginsActivated()"))

    def activatePlugin(self, name):

        """

        Public method to activate a plugin.

        

        @param name name of the module to be activated

        @return reference to the initialized plugin object

        """

        try:

            try:

                module = self.__availableModules[name]

            except KeyError:

                return None

            if not self.__canActivatePlugin(module):
                raise PluginActivationError(module.pluginModuleName)

            version = getattr(module, "version")

            className = getattr(module, "className")

            pluginClass = getattr(module, className)

            print("version=", version, " className=", className, " pluginClass=", pluginClass)

            pluginObject = None

            pluginObject = pluginClass(self.__ui)

            try:

                print("WILL TRY TO ACTIVATE ", pluginObject)

                obj, ok = pluginObject.activate()

                print("ACTIVATED")

            except TypeError:

                module.error = "Incompatible plugin activation method."

                obj = None

                ok = True

            except Exception as err:

                module.error = str(err)

                traceback.print_exc(file=sys.stdout)

                obj = None

                ok = False

            if not ok:
                return None

            pluginObject.pluginModule = module

            pluginObject.pluginName = className

            pluginObject.pluginVersion = version

            self.__activePlugins[name] = pluginObject

            return obj

        except PluginActivationError:

            return None

    def __canActivatePlugin(self, module):

        """

        Private method to check, if a plugin can be activated.

        

        @param module reference to the module to be activated

        @return flag indicating, if the module satisfies all requirements

            for being activated (boolean)

        """

        try:

            if not hasattr(module, "version"):
                raise PluginModuleFormatError(module.pluginModuleName, "version")

            if not hasattr(module, "className"):
                raise PluginModuleFormatError(module.pluginModuleName, "className")

            className = getattr(module, "className")

            if not hasattr(module, className):
                raise PluginModuleFormatError(module.pluginModuleName, className)

            pluginClass = getattr(module, className)

            if not hasattr(pluginClass, "__init__"):
                raise PluginClassFormatError(module.pluginModuleName,

                                             className, "__init__")

            if not hasattr(pluginClass, "activate"):
                raise PluginClassFormatError(module.pluginModuleName,

                                             className, "activate")

            if not hasattr(pluginClass, "deactivate"):
                raise PluginClassFormatError(module.pluginModuleName,

                                             className, "deactivate")

            return True

        except PluginModuleFormatError as e:

            print(repr(e))

            return False

        except PluginClassFormatError as e:

            print(repr(e))

            return False

    def deactivatePlugin(self, name, onDemand=False):

        """

        Public method to deactivate a plugin.

        

        @param name name of the module to be deactivated

        @keyparam onDemand flag indicating deactivation of an 

            on demand plugin (boolean)

        """

        try:

            module = self.__availableModules[name]

        except KeyError:

            return

        if self.__canDeactivatePlugin(module):

            pluginObject = None

            if name in self.__activePlugins:
                pluginObject = self.__activePlugins[name]

            if pluginObject:

                self.emit(SIGNAL("pluginAboutToBeDeactivated"), name, pluginObject)

                pluginObject.deactivate()

                self.emit(SIGNAL("pluginDeactivated"), name, pluginObject)

                try:

                    self.__activePlugins.pop(name)

                except KeyError:

                    pass

    def __canDeactivatePlugin(self, module):

        """

        Private method to check, if a plugin can be deactivated.

        

        @param module reference to the module to be deactivated

        @return flag indicating, if the module satisfies all requirements

            for being deactivated (boolean)

        """

        return getattr(module, "deactivateable", True)

    def isPluginLoaded(self, pluginName):

        """

        Public method to check, if a certain plugin is loaded.

        

        @param pluginName name of the plugin to check for (string or QString)

        @return flag indicating, if the plugin is loaded (boolean)

        """

        return str(pluginName) in list(self.__activePlugins.keys())
