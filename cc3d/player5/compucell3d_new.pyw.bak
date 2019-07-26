#
# compucell3d.pyw - main Python script for the CompuCell3D interactive (GUI) application
#
import sys
import os
import numpy
import vtk
import CC3DXML

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PyQt5

# TODO
# * restore xml widget prepareXMLTreeView in simpleTabView.py

# # setting api for QVariant is necessary to get player5 workign with MinGW-compiled PyQt4
# import sip
# sip.setapi('QVariant', 1)

#print sys.path

'''
QUESTION::
Why do we need it?
#instaling message handler to suppress spurious qt messages 
'''
if sys.platform=='darwin':
    import platform
    mac_ver = platform.mac_ver()
    mac_ver_float = float('.'.join(mac_ver[0].split('.')[:2]))
    if mac_ver_float == 10.11:

        def handler(msg_type, msg_log_context, msg_string=None):
            if msg_log_context.startswith('QCocoaView handleTabletEvent'):
                return
            print msg_log_context

        PyQt4.QtCore.qInstallMsgHandler(handler)

    elif mac_ver_float == 10.10:

        def handler(msg_type, msg_log_context, msg_string=None):
            # pass
            if msg_log_context.startswith('Qt: qfontForThemeFont:'):
                return
            print msg_log_context

        PyQt5.QtCore.qInstallMsgHandler(handler)


# Initializing debugger and setting output level
from Messaging import setDebugging
setDebugging(0)

def main(argv):
    '''
    Main method this  displays splash screen and opens the MainWindow of CompuCell3D.
    :param argv: Default parameter
    :return: none
    '''
    # argument count
    argc=len(argv)

    from CQt.CQApplication import CQApplication
    app = CQApplication(argv)

    # Display Splash Screen
    pixmap = QPixmap("icons/splash_angio.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    '''
       NOTE:: 2010- mainWindow.raise_() must be called after mainWindow.show()
              otherwise the CC3D player5 GUI won't receive foreground focus. It's a
              workaround for a well-known bug caused by PyQt4/Qt on Mac OS X, as shown here:
              http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    '''
    if sys.platform.startswith('darwin'):
        splash.raise_()


    # TODO Fix this - set paths and uncomment
    sys.path.append(os.environ["PYTHON_MODULE_PATH"])
    sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

    import XMLUtils
    # sys.exit()

    '''
    QUESTION::
    Why version what's the use of it?
    '''
    versionStr='3.6.0'
    revisionStr='0'
    try:
        import Version
        versionStr=Version.getVersionAsString()
        revisionStr=Version.getSVNRevisionAsString()
    except ImportError,e:
        pass

    # Show messages about the progress on splash screen - 1
    baseMessage="CompuCell3D Version: %s Revision: %s\n"%(versionStr,revisionStr)
    firstMessage=baseMessage+"Loading User Interface ..."
    splash.showMessage(firstMessage,Qt.AlignLeft,  Qt.white)

    # Show messages about the progress on splash screen - 1
    secondMessage=baseMessage+"Loading CompuCell3D Python Modules..."
    splash.showMessage(secondMessage,Qt.AlignLeft,  Qt.white)

    # Import CompuCellSetup for ---
    import CompuCellSetup
    CompuCellSetup.playerType="new"
    '''
    The value of CompuCellSetup.playerType (can be "new" or "old") determines which PlayerPython module will be loaded.
    For the new player5 we want PlayerPythonNew
    '''

    app.processEvents()

    # printing on console
    print 'compucell3d.pyw:   type(argv)=',type(argv)
    print 'compucell3d.pyw:   argv=',argv

    import getopt
    opts=None
    args=None
    try:
        print '   argv[0:] =',argv[0:]
        opts, args = getopt.getopt(argv[0:],  "i:s:o:f:c:h:p:w:", ["help","noOutput","exitWhenDone","guiScan","currentDir=","outputFrequency=","port=","tweditPID=","prefs=","maxNumberOfRuns=" ])
        print "type(opts), opts=",type(opts),opts
        print "type(args), args=",type(args),args
    except getopt.GetoptError, err:
        # Print error and exit:
        # will print something like "option -a not recognized"
        print str(err)
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

    # Import modules related UI developed in CC3D
    # UserInterface creates the MainWindow in CC3D
    from UI.UserInterface import UserInterface
    from CQt.CQApplication import CQApplication

    mainWindow = UserInterface()
    mainWindow.setArgv(argv)

    '''
    passing command line to the code
    process reminder of the command line options
    TODO::
    if argv != "":
        mainWindow.viewmanager.processCommandLineOptions(opts)
    '''

    mainWindow.show()
    splash.finish(mainWindow)

    '''
       NOTE:: 2010- mainWindow.raise_() must be called after mainWindow.show()
              otherwise the CC3D player5 GUI won't receive foreground focus. It's a
              workaround for a well-known bug caused by PyQt4/Qt on Mac OS X, as shown here:
              http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    '''
    mainWindow.raise_()

    app.exec_()

if __name__ == '__main__':
    main(sys.argv[1:])
