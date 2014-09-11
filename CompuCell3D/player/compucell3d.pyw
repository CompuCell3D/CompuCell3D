#
# compucell3d.pyw - main Python script for the CompuCell3D interactive (GUI) application
#
import sys
import os




def setVTKPaths():
   import sys
   from os import environ
   import string
   import sys
   platform=sys.platform
   if platform=='win32':
      sys.path.insert(0,environ["PYTHON_DEPS_PATH"])
      sys.path.append(environ["VTKPATH"])
      # sys.path.append(environ["VTKPATH1"])
      
      # # sys.path.append(environ["PYQT_PATH"])
      # # sys.path.append(environ["SIP_PATH"])
      # # sys.path.append(environ["SIP_UTILS_PATH"])
      
      # sys.path.insert(0,environ["PYQT_PATH"])
      # sys.path.insert(0,environ["SIP_PATH"])
      # sys.path.insert(0,environ["SIP_UTILS_PATH"])
      
#   else:
#      swig_path_list=string.split(environ["VTKPATH"])
#      for swig_path in swig_path_list:
#         sys.path.append(swig_path)

# print "PATH=",sys.path
setVTKPaths()
# print "PATH=",sys.path  



#mainWindow = None

# setting api for QVariant is necessary to get player workign with MinGW-compiled PyQt4
import sip
sip.setapi('QVariant', 1)


from PyQt4.QtCore import *
from PyQt4.QtGui import *


def main(argv):
    argc=len(argv)
    from CQt.CQApplication import CQApplication    
    app = CQApplication(argv)
    
    pixmap = QPixmap("icons/splash_angio.png")
    splash = QSplashScreen(pixmap)
    

    splash.show()

    
    if sys.platform.startswith('darwin'):
        splash.raise_()    

    # RWH:  not sure why vtk was being imported here
    #splash.showMessage("Loading VTK modules...",Qt.AlignLeft,  Qt.white)
    # import vtk    

    sys.path.append(os.environ["PYTHON_MODULE_PATH"])
    sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

    
    versionStr='3.6.0'    
    revisionStr='0'
    try:
        import Version            
        versionStr=Version.getVersionAsString()    
        revisionStr=Version.getSVNRevisionAsString()    
    except ImportError,e:
        pass
    baseMessage="CompuCell3D Version: %s Revision: %s\n"%(versionStr,revisionStr)
    firstMessage=baseMessage+"Loading User Interface ..."
    
    splash.showMessage(firstMessage,Qt.AlignLeft,  Qt.white)

    # splash.showMessage("Loading User Interface ...",Qt.AlignLeft,  Qt.white)
    from UI.UserInterface import UserInterface    
    from CQt.CQApplication import CQApplication
    
    # sys.path.append(os.environ["PYTHON_MODULE_PATH"])
    # sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])
    secondMessage=baseMessage+"Loading CompuCell3D Python Modules..."
    
    splash.showMessage(secondMessage,Qt.AlignLeft,  Qt.white)    
    # splash.showMessage("Loading CompuCell3D Python Modules...",Qt.AlignLeft,  Qt.white)
    import CompuCellSetup
    
    CompuCellSetup.playerType="new" # the value of CompuCellSetup.playerType (can be "new" or "old") determines which PlayerPython module will be loaded. For the new player we want PlayerPythonNew
    import PlayerPython # from now on import PlayerPython will import PlayerPythonNew
    
    app.processEvents()

    print 'compucell3d.pyw:   type(argv)=',type(argv)
    print 'compucell3d.pyw:   argv=',argv
    
    import getopt
    opts=None
    args=None
    try:
#        opts, args = getopt.getopt(sys.argv[1:], "i:s:o:f:c:h", ["help","noOutput","exitWhenDone","currentDir=","outputFrequency=","port=","tweditPID=","prefs=" ])
        print '   argv[0:] =',argv[0:]
#        opts, args = getopt.getopt(argv[1:],  ["prefs="])
        opts, args = getopt.getopt(argv[0:],  "i:s:o:f:c:h:p:w:", ["help","noOutput","exitWhenDone","currentDir=","outputFrequency=","port=","tweditPID=","prefs=" ])
        print "type(opts), opts=",type(opts),opts
        print "type(args), args=",type(args),args
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        # self.usage()
        sys.exit(2)
    output = None
    verbose = False
    currentDir=""
    for o, a in opts:
        print "o=",o
        print "a=",a
        if o in ("--prefs"):
            print 'compucell3d.pyw:  prefsFile=',a
            
            import Configuration
            Configuration.setPrefsFile(a)
            # Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Biocomplexity", a)
            # Configuration.setSetting("PreferencesFile", a)            
    
    
    from UI.UserInterface import UserInterface    
    from CQt.CQApplication import CQApplication
    mainWindow = UserInterface()
    mainWindow.setArgv(argv) # passing command line to the code
    
    # process reminder of the command line options
    if argv != "":
        mainWindow.viewmanager.processCommandLineOptions(opts)
    
    mainWindow.show()
    splash.finish(mainWindow)

    # 2010: mainWindow.raise_() must be called after mainWindow.show() 
    #       otherwise the CC3D player GUI won't receive foreground focus. It's a
    #       workaround for a well-known bug caused by PyQt4/Qt on Mac OS X, as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    mainWindow.raise_()
    
    app.exec_()

if __name__ == '__main__':
    main(sys.argv[1:])
