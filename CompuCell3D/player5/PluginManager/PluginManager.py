# Plugin Manager is NOT used anywhere so far

class PluginManager(QObject):
    def __init__(self, parent = None, doLoadPlugins = True, develPlugin = None):
        QObject.__init__(self, parent)
        self.__onDemandInactiveModules = {}
        self.__inactiveModules = {}
        
        
    def getPluginObject(self, type_, typename):
        """
        Public method to activate an ondemand plugin given by type and typename.
        
        @param type_ type of the plugin to be activated (string)
        @param typename name of the plugin within the type category (string)
        @return reference to the initialized plugin object
        """
        for name, module in self.__onDemandInactiveModules.items():
            if getattr(module, "pluginType") == type_ and \
               getattr(module, "pluginTypename") == typename:
                return self.activatePlugin(name)
        
        return None

    def activatePlugin(self, name):
        """
        Public method to activate a plugin.
        
        @param name name of the module to be activated
        @keyparam onDemand flag indicating activation of an 
            on demand plugin (boolean). onDemand = True
        @return reference to the initialized plugin object
        """
        
        try:
            try:
               module = self.__onDemandInactiveModules[name]
            except KeyError:
                return None
            
            version     = getattr(module, "version")
            className   = getattr(module, "className")
            pluginClass = getattr(module, className)

            pluginObject = self.__onDemandInactivePlugins[name]

            try:
                obj, ok = pluginObject.activate()
            except TypeError:
                module.error = "Incompatible plugin activation method."
                obj = None
                ok = True
            except StandardError, err:
                module.error = QString(unicode(err))
                obj = None
                ok = False
            if not ok:
                return None
            
            return obj
        except PluginActivationError:
            return None
