"""

Module implementing the exceptions raised by the plugin system.

"""

from cc3d.twedit5.twedit.utils.global_imports import *


class PluginError(Exception):
    """

    Class defining a special error for the plugin classes.

    """

    def __init__(self):
        """

        Constructor

        """

        self._errorMessage = QApplication.translate("PluginError", "Unspecific plugin error.")

    def __repr__(self):
        """

        Private method returning a representation of the exception.

        

        @return string representing the error message

        """

        return str(self._errorMessage)

    def __str__(self):
        """

        Private method returning a string representation of the exception.

        

        @return string representing the error message

        """

        return str(self._errorMessage)


class PluginPathError(PluginError):
    """

    Class defining an error raised, when the plugin paths were not found and

    could not be created.

    """

    def __init__(self, msg=None):

        """

        Constructor

        

        @param msg message to be used by the exception (string or QString)

        """

        if msg:

            self._errorMessage = msg

        else:

            self._errorMessage = QApplication.translate("PluginError", "Plugin paths not found or not creatable.")


class PluginModulesError(PluginError):
    """

    Class defining an error raised, when no plugin modules were found.

    """

    def __init__(self):
        """

        Constructor

        """

        self._errorMessage = QApplication.translate("PluginError", "No plugin modules found.")


class PluginLoadError(PluginError):
    """

    Class defining an error raised, when there was an error during plugin loading.

    """

    def __init__(self, name):
        """

        Constructor

        

        @param name name of the plugin module (string)

        """

        self._errorMessage = QApplication.translate("PluginError", "Error loading plugin module: %1").arg(name)


class PluginActivationError(PluginError):
    """

    Class defining an error raised, when there was an error during plugin activation.

    """

    def __init__(self, name):
        """

        Constructor

        

        @param name name of the plugin module (string)

        """

        self._errorMessage = QApplication.translate("PluginError", "Error activating plugin module: %1").arg(name)


class PluginModuleFormatError(PluginError):
    """

    Class defining an error raised, when the plugin module is invalid.

    """

    def __init__(self, name, missing):
        """

        Constructor

        

        @param name name of the plugin module (string)

        @param missing description of the missing element (string)

        """

        self._errorMessage = QApplication.translate("PluginError", "The plugin module %1 is missing %2.").arg(name).arg(
            missing)


class PluginClassFormatError(PluginError):
    """

    Class defining an error raised, when the plugin module's class is invalid.

    """

    def __init__(self, name, class_, missing):
        """

        Constructor

        

        @param name name of the plugin module (string)

        @param class_ name of the class not satisfying the requirements (string)

        @param missing description of the missing element (string)

        """

        self._errorMessage = QApplication.translate("PluginError",
                                                    "The plugin class %1 of module %2 is missing %3.").arg(class_).arg(
            name).arg(missing)
