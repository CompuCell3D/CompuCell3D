"""

Module implementing the Plugin Manager.

"""
import traceback
import cc3d.twedit5.twedit as twedit
from cc3d.twedit5.PluginManager.PluginExceptions import *


# this class will store basic plugin data
class BasicPluginData(object):

    def __init__(self, name='', file_name=''):

        self.name = name
        self.fileName = file_name
        self.author = ''
        self.autoactivate = False
        self.deactivateable = False
        self.version = ''
        self.className = ''
        self.packageName = ''
        self.shortDescription = ''
        self.longDescription = ''

    def __str__(self):

        attribute_names = ['name', 'fileName', 'author', 'autoactivate', 'deactivateable', 'version', 'className',
                           'packageName', 'shortDescription', 'longDescription']

        ret_str = ''

        for attributeName in attribute_names:

            try:
                ret_str += attributeName + '=' + str(getattr(self, attributeName)) + '\n'
            except TypeError:
                ret_str += attributeName + '=' + getattr(self, attributeName) + '\n'

        return ret_str


class PluginManager(QObject):
    """
    Class implementing the Plugin Manager.
    """

    def __init__(self, parent=None, do_load_plugins=True, devel_plugin=None):

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
            # when packaging on Windows with pyinstaller the path to executable is
            # accesible via sys.executable as Python is bundled with the distribution

            # os.path.dirname(Configuration.__file__) returned by pyInstaller will not work without
            # some modifications so it is best tu use os.path.dirname(sys.executable) approach

            self.tweditRootPath = os.path.dirname(sys.executable)

            self.pluginPath = os.path.join(self.tweditRootPath, "Plugin")

        # self.__checkPluginsDownloadDirectory()

        plugin_modules = self.get_plugin_modules(self.pluginPath)

        self.__insert_plugins_paths()

        print('pluginModules=', plugin_modules)

        for pluginName in plugin_modules:
            self.query_plugin(pluginName, self.pluginPath)

            # checking out which plugins were succesfully querried

        # discovered plugin names are storred in pluginModules list

        defined_plugin_module_name_set = set(plugin_modules)

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

        # load and activate plugins

        for pluginName, bpd in list(self.__pluginQueries.items()):
            print('************PLUGIN NAME=', pluginName)
            print(bpd)
            self.load_plugin(pluginName)

        # postactivate initialization -  extra initialization steps needed after plugin is ready
        self.run_for_all_plugins(function_name='post_activate', argument_dict={})

    def run_for_all_plugins(self, function_name, argument_dict):
        """
        attempts to run function (function_name) with arguments given by argument_dict for all plugins
        :param function_name:
        :param argument_dict:
        :return:
        """

        for pluginName, plugin in self.__activePlugins.items():
            try:
                function = getattr(plugin, function_name)
                function(**argument_dict)
            except (TypeError, AttributeError) as e:
                print('PLUGIN: ', pluginName, ' COULD NOT APPLY ', function_name, ' with arguments=', argument_dict)
                print(e)

    def get_active_plugin(self, name):

        try:

            return self.__activePlugins[name]

        except LookupError:

            return None

    def get_plugin_modules(self, plugin_path):

        """
        Public method to get a list of plugin modules.

        @param plugin_path name of the path to search (string)

        @return list of plugin module names (list of string)

        """

        plugin_files = [f[:-3] for f in os.listdir(plugin_path) if self.is_valid_plugin_name(f)]

        return plugin_files[:]

    @staticmethod
    def is_valid_plugin_name(plugin_name):

        """

        Public methode to check, if a file name is a valid plugin name.

        Plugin modules must start with "Plugin" and have the extension ".py".

        @param plugin_name name of the file to be checked (string)

        @return flag indicating a valid plugin name (boolean)

        """

        return plugin_name.startswith("Plugin") and plugin_name.endswith(".py")

    def __insert_plugins_paths(self):

        """
        Private method to insert the valid plugin paths into the search path.

        """

        sys.path.insert(2, self.pluginPath)

    @staticmethod
    def load_plugin_source(plugin_name, plugin_fname):
        import importlib.machinery
        print("\n\n\n ##################loading source for ", plugin_fname)
        loader = importlib.machinery.SourceFileLoader(plugin_name, plugin_fname)
        module = loader.load_module(plugin_name)
        return module

    def query_plugin(self, name, directory, reload_=False):

        """

        Public method to preload a plugin module and run few queries like plugin name, author, descriptiopn etc

        @param name name of the module to be loaded (string)

        @param directory name of the plugin directory (string)

        @param reload_ flag indicating to reload the module (boolean)

        """

        attribute_names = ['author', 'autoactivate', 'deactivateable', 'version', 'className', 'packageName',
                           'shortDescription', 'longDescription']

        try:

            print('plugin name=', name)

            fname = f"{os.path.join(directory, name)}.py"

            module = self.load_plugin_source(plugin_name=name, plugin_fname=fname)

            bpd = BasicPluginData(name, fname)

            for attributeName in attribute_names:

                if hasattr(module, "autoactivate"):
                    setattr(bpd, attributeName, getattr(module, attributeName))

            self.__pluginQueries[name] = bpd

        except Exception as err:

            print("Error loading plugin module:", name)

            print(str(err))

            traceback.print_exc(file=sys.stdout)

    def load_plugin(self, name, force_activate=False):

        """
        Public method to load aand activate plugin module.

        @param name name of the p[lugin to load  - used as a key to self.__pluginQueries
        @param force_activate
        """

        try:
            bpd = self.__pluginQueries[name]
        except LookupError:

            return

        try:
            name = bpd.name
            fname = bpd.fileName
            print('LOADING ', name, 'from ', fname)
            print("loading source for ", fname)

            module = self.load_plugin_source(plugin_name=name, plugin_fname=fname)
            module.pluginModuleName = name

            module.pluginModuleFilename = fname

            self.__modulesCount += 1
            # self.__inactiveModules[name] = module
            self.__availableModules[name] = module

            # checking whether user has specific load on startup setting for this plugin
            configuration = self.__ui.configuration

            plugin_autoload_data = configuration.pluginAutoloadData()

            autoload_flag = bpd.autoactivate

            try:
                autoload_flag = plugin_autoload_data[name]
            except LookupError:
                pass

            if autoload_flag:
                # instantiates plugin object based on code stored in module
                self.activate_plugin(name)

            elif force_activate:
                # instantiates plugin object based on code stored in module
                self.activate_plugin(name)

            return module

        except Exception as err:
            self.__failedModules[name] = "Module failed to load. Error: %s" % str(err)
            print("Error loading plugin module:", name)
            traceback.print_exc(file=sys.stdout)

    def is_plugin_active(self, name):

        return name in list(self.__activePlugins.keys())

    def get_basic_plugin_data(self, name):

        try:

            return self.__pluginQueries[name]

        except LookupError:

            return None

    def get_available_modules(self):

        return list(self.__availableModules.keys())

    def unload_plugin(self, name):

        """
        Public method to unload and deactivate plugin module.
        @param name name of the module to be unloaded (string)

        @return flag indicating success (boolean)

        """

        if name in self.__activePlugins:
            self.deactivate_plugin(name)
            # replacing module (potentially arge object with simple bdp plugin descriptor)
            self.__availableModules[name] = self.__pluginQueries[name]

        self.__modulesCount -= 1

        return True

    def unload_plugins(self):

        unloaded_plugin_names = []

        for pluginName in list(self.__activePlugins.keys()):
            self.__activePlugins[pluginName].deactivate()

            unloaded_plugin_names.append(pluginName)

        for pluginName in unloaded_plugin_names:
            del self.__activePlugins[pluginName]

    @staticmethod
    def remove_plugin_from_sys_modules(plugin_name, package, internal_packages):

        """
        Public method to remove a plugin and all related modules from sys.modules.

        @param plugin_name name of the plugin module (string)

        @param package name of the plugin package (string)

        @param internal_packages list of intenal packages (list of string)

        @return flag indicating the plugin module was found in sys.modules (boolean)

        """

        packages = [package] + internal_packages

        found = False

        for moduleName in list(sys.modules.keys())[:]:

            if moduleName == plugin_name or moduleName.split(".")[0] in packages:
                found = True

                del sys.modules[moduleName]

        return found

    def init_on_demand_plugins(self):
        """
        Public method to create plugin objects for all on demand plugins.

        Note: The plugins are not activated.

        """

        names = sorted(self.__onDemandInactiveModules.keys())

        for name in names:
            self.init_on_demand_plugin(name)

    def init_on_demand_plugin(self, name):

        """
        Public method to create a plugin object for the named on demand plugin.

        Note: The plugin is not activated.

        """

        try:
            try:
                module = self.__onDemandInactiveModules[name]
            except KeyError:
                return None

            if not self.__can_activate_plugin(module):
                raise PluginActivationError(module.eric4PluginModuleName)

            version = getattr(module, "version")

            class_name = getattr(module, "className")

            plugin_class = getattr(module, class_name)

            if name not in self.__onDemandInactivePlugins:
                plugin_object = plugin_class(self.__ui)

                plugin_object.eric4PluginModule = module

                plugin_object.eric4PluginName = class_name

                plugin_object.eric4PluginVersion = version

                self.__onDemandInactivePlugins[name] = plugin_object

        except PluginActivationError:
            return None

    def activate_plugin(self, name):
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

            if not self.__can_activate_plugin(module):
                raise PluginActivationError(module.pluginModuleName)

            version = getattr(module, "version")
            class_name = getattr(module, "className")
            plugin_class = getattr(module, class_name)

            print("version=", version, " className=", class_name, " pluginClass=", plugin_class)

            plugin_object = plugin_class(self.__ui)
            try:
                print("WILL TRY TO ACTIVATE ", plugin_object)
                obj, ok = plugin_object.activate()
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

            plugin_object.pluginModule = module
            plugin_object.pluginName = class_name
            plugin_object.pluginVersion = version
            self.__activePlugins[name] = plugin_object

            return obj

        except PluginActivationError:

            return None

    @staticmethod
    def __can_activate_plugin(module):
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

            class_name = getattr(module, "className")

            if not hasattr(module, class_name):
                raise PluginModuleFormatError(module.pluginModuleName, class_name)

            plugin_class = getattr(module, class_name)

            if not hasattr(plugin_class, "__init__"):
                raise PluginClassFormatError(module.pluginModuleName, class_name, "__init__")

            if not hasattr(plugin_class, "activate"):
                raise PluginClassFormatError(module.pluginModuleName, class_name, "activate")

            if not hasattr(plugin_class, "deactivate"):
                raise PluginClassFormatError(module.pluginModuleName, class_name, "deactivate")

            return True

        except PluginModuleFormatError as e:

            print(repr(e))

            return False

        except PluginClassFormatError as e:

            print(repr(e))

            return False

    def deactivate_plugin(self, name):

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

        if self.__can_deactivate_plugin(module):

            plugin_object = None

            if name in self.__activePlugins:
                plugin_object = self.__activePlugins[name]

            if plugin_object:

                self.emit(SIGNAL("pluginAboutToBeDeactivated"), name, plugin_object)

                plugin_object.deactivate()

                self.emit(SIGNAL("pluginDeactivated"), name, plugin_object)

                try:

                    self.__activePlugins.pop(name)

                except KeyError:

                    pass

    @staticmethod
    def __can_deactivate_plugin(module):
        """
        Private method to check, if a plugin can be deactivated.
        @param module reference to the module to be deactivated
        @return flag indicating, if the module satisfies all requirements
            for being deactivated (boolean)
        """

        return getattr(module, "deactivateable", True)
