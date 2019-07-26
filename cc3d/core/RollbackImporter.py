# Full description of the RollbackImporter can be found on
# http://pyunit.sourceforge.net/notes/reloading.html
# Taking a step back, the situation is as follows:

# 1. A set of modules are loaded
# 2. The modules are used
# 3. The code is changed
# 4. The modules must be used again, but be freshly imported

# The solution is to draw a line in the 'module sand' before loading and using the modules,
# then roll back to that point before re-running the code. This is accomplished, in PyUnit, by the following class:

# RollbackImporter instances install themselves as a proxy for the built-in __
# import__ function that lies behind the 'import' statement. Once installed, they note all imported modules,
# and when uninstalled, they delete those modules from the system module list; this ensures that the modules
# will be freshly loaded from their source code when next imported.

# The rollback importer is used as follows in the Player see functions __ runSim and __stepSim

# def runClicked(self):
# if self.rollbackImporter:
# self.rollbackImporter.uninstall()

# Credits: Steve Purcell

import sys


class RollbackImporter:
    def __init__(self):
        """
        Creates an instance and installs as the global importer
        """
        self.previousModules = sys.modules.copy()
        # print("__builtins__",__builtins__)
        self.realImport = __builtins__['__import__']
        __builtins__['__import__'] = self._import
        self.newModules = {}

    def _import(self, name, globals=None, locals=None, fromlist=[], level=0):
        """
        import override. Modules/packages that have word "steppable" in their name
        are imported as level 0. This means that PYTHONPATH will be used to
        search for them and even some of them are  relative imports
        e.g.  from .bacterium_macrophage_steppables import MySteppable

        this import will be resolved as level 0 import. If we did not intercept level variable
        the above relative import would be given level=1 and this would result in error

        This special handling of level import variable allows 3rd party packages to function properly and at the same
        time have relative imports for steppable modules

        :param name:
        :param globals:
        :param locals:
        :param fromlist:
        :param level:
        :return:
        """

        level_import = level
        if name.find('steppable') > 0 or name.find('Steppable') > 0:
            level_import = 0
        else:
            if fromlist:
                for mod_name in fromlist:
                    if mod_name.find('steppable') > 0 or mod_name.find('Steppable') > 0:
                        level_import = 0
                        break

        result = self.realImport(*(name, globals, locals, fromlist, level_import))

        self.newModules[name] = 1
        return result

    def uninstall(self):
        for modname in list(self.newModules.keys()):
            if modname not in self.previousModules:
                # Force reload when modname next imported
                if modname in sys.modules:
                    del sys.modules[modname]
        __builtins__['__import__'] = self.realImport
