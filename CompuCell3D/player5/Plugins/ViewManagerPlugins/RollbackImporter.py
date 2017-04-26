# Full description of the RollbackImporter can be found on
# http://pyunit.sourceforge.net/notes/reloading.html
# Taking a step back, the situation is as follows:

   # 1. A set of modules are loaded
   # 2. The modules are used
   # 3. The code is changed
   # 4. The modules must be used again, but be freshly imported 

# The solution is to draw a line in the 'module sand' before loading and using the modules, then roll back to that point before re-running the code. This is accomplished, in PyUnit, by the following class: 

# RollbackImporter instances install themselves as a proxy for the built-in __import__ function that lies behind the 'import' statement. Once installed, they note all imported modules, and when uninstalled, they delete those modules from the system module list; this ensures that the modules will be freshly loaded from their source code when next imported.

# The rollback importer is used as follows in the Player see functions __ runSim and __stepSim

        # def runClicked(self):
            # if self.rollbackImporter:
                # self.rollbackImporter.uninstall()
            
# Credits: Steve Purcell

import sys
class RollbackImporter:
    def __init__(self):
        "Creates an instance and installs as the global importer"
        self.previousModules = sys.modules.copy()
        print "__builtins__",__builtins__
        self.realImport = __builtins__['__import__']
        __builtins__['__import__'] = self._import
        self.newModules = {}
        
    def _import(self, name, globals=None, locals=None, fromlist=[], level=0):
        result = apply(self.realImport, (name, globals, locals, fromlist))
        self.newModules[name] = 1
        return result
        
    def uninstall(self):
        for modname in self.newModules.keys():
            if not self.previousModules.has_key(modname):
                # Force reload when modname next imported
		if sys.modules.has_key(modname):
		    del sys.modules[modname]
        __builtins__['__import__'] = self.realImport