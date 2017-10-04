import os
import subprocess
import sys

class CompuCell3DCLI:

    def __init__(self):
        '''
         Set the CompuCell3D installation directory path. Structure-
         <CompuCell3D_Installation>/CLI/<this_Script.py>
        '''
        scriptPath = os.path.realpath(__file__)
        CLIDirectory = os.path.dirname(scriptPath)
        compuCell3DInstallationPath = os.path.dirname(CLIDirectory)
        self.__compuCell3DPath = compuCell3DInstallationPath

        CC3DLauncherFileName = "CC3DLauncher.py"
        self.__CC3DLauncherFilePath = os.path.join(CLIDirectory, CC3DLauncherFileName)

        self.__LIBRARY_PATH_LABEL = ""
        if sys.platform.startswith('linux'):
            self.__LIBRARY_PATH_LABEL = "LD_LIBRARY_PATH"
        elif sys.platform.startswith('darwin'):
            self.__LIBRARY_PATH_LABEL = "DYLD_LIBRARY_PATH"

    def setupEnvironment(self):
        """

        :return:
        """
        os.environ['COMPUCELL3D_MAJOR_VERSION'] = '3'
        os.environ['COMPUCELL3D_MINOR_VERSION'] = '7'
        os.environ['COMPUCELL3D_BUILD_VERSION'] = '6'

        os.environ['PREFIX_CC3D'] = self.__compuCell3DPath
        os.environ['PYTHON_EXEC'] = self.__compuCell3DPath + "/Python27/bin/python2.7"

        os.environ['PYTHON_MODULE_PATH'] = self.__compuCell3DPath + "/pythonSetupScripts"

        os.environ['COMPUCELL3D_PLUGIN_PATH'] = self.__compuCell3DPath + "/lib/CompuCell3DPlugins"
        os.environ['COMPUCELL3D_STEPPABLE_PATH'] = self.__compuCell3DPath + "/lib/CompuCell3DSteppables"

        os.environ['SWIG_LIB_INSTALL_DIR']= self.__compuCell3DPath + "/lib/python"
        os.environ['SOSLIB_PATH'] = self.__compuCell3DPath

        os.environ['PYTHONPATH'] = self.__compuCell3DPath + "/pythonSetupScripts"
        os.environ['PYTHONPATH'] += os.pathsep + self.__compuCell3DPath + "/lib/python"
        os.environ['PYTHONPATH'] += os.pathsep + self.__compuCell3DPath + "/vtk/lib/python2.7/site-packages/"

        os.environ[self.__LIBRARY_PATH_LABEL] = self.__compuCell3DPath + "/lib"
        os.environ[self.__LIBRARY_PATH_LABEL] += os.pathsep + self.__compuCell3DPath + "/lib/python"
        os.environ[self.__LIBRARY_PATH_LABEL] += os.pathsep + self.__compuCell3DPath + "/vtk/lib"
        os.environ[self.__LIBRARY_PATH_LABEL] += os.pathsep + self.__compuCell3DPath + "/player5/Utilities"
        os.environ[self.__LIBRARY_PATH_LABEL] += os.pathsep + os.environ['COMPUCELL3D_PLUGIN_PATH']
        os.environ[self.__LIBRARY_PATH_LABEL] += os.pathsep + os.environ['COMPUCELL3D_STEPPABLE_PATH']

    def invokeCC3DLauncher(self):
        currentArguments = sys.argv
        currentArguments.remove(currentArguments[0])
        newArguments = [os.environ['PYTHON_EXEC'], self.__CC3DLauncherFilePath]
        newArguments = newArguments + currentArguments

        subprocess.call(newArguments)

def main():
    compuCell3DCLI = CompuCell3DCLI()
    compuCell3DCLI.setupEnvironment()
    compuCell3DCLI.invokeCC3DLauncher()

if __name__ == '__main__':
    main()