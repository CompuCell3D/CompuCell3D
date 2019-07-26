"""
CC3D main script that lunches CC#D in the GUI mode
When running automated testing f Demo suite use the following cml options:


 --exitWhenDone --testOutputDir=/Users/m/cc3d_tests --numSteps=100

 or for automatic starting of a particular simulation you use :
 --input=/home/m/376_dz/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d
"""
#
# compucell3d.pyw - main Python script for the CompuCell3D interactive (GUI) application
#
import sys
import os
import argparse
import vtk

# TODO
# * restore xml widget prepareXMLTreeView in simpleTabView.py

# # setting api for QVariant is necessary to get player5 workign with MinGW-compiled PyQt4
# import sip
# sip.setapi('QVariant', 1)

print sys.path
import numpy
import vtk

if sys.platform.lower().startswith('linux'):
    # On linux have to import rr early on to avoid
    # PyQt-related crash - appears to only affect VirtualBox Installs
    # of linux
    try:
        import roadrunner
    except ImportError:
        print 'Could not import roadrunner'
        pass

# import CC3DXML

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PyQt5

if sys.platform.startswith('win'):
    # this takes care of the need to distribute qwindows.dll with the qt5 application
    # it needs to be locarted in the directory <library_path>/platforms
    QCoreApplication.addLibraryPath("./bin/")

# instaling message handler to suppres spurious qt messages
if sys.platform == 'darwin':
    import platform

    mac_ver = platform.mac_ver()
    mac_ver_float = float('.'.join(mac_ver[0].split('.')[:2]))
    if mac_ver_float == 10.11:

        def handler(msg_type, msg_log_context, msg_string=None):
            if msg_log_context.startswith('QCocoaView handleTabletEvent'):
                return
            print msg_log_context

            # looks like we do not need those in PyQt5 version
            # PyQt5.QtCore.qInstallMsgHandler(handler)

    elif mac_ver_float == 10.10:

        def handler(msg_type, msg_log_context, msg_string=None):
            # pass
            if msg_log_context.startswith('Qt: qfontForThemeFont:'):
                return
            print msg_log_context

            # looks like we do not need those in PyQt5 version
            # PyQt5.QtCore.qInstallMsgHandler(handler)

# setting debug information output
from Messaging import setDebugging

setDebugging(0)


def main(argv):
    argc = len(argv)

    # if sys.platform.startswith('darwin'):
    #     PyQt5.QtCore.QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)

    from CQt.CQApplication import CQApplication
    app = CQApplication(argv)

    pixmap = QPixmap("icons/splash_angio.png")
    splash = QSplashScreen(pixmap)

    splash.show()

    if sys.platform.startswith('darwin'):
        splash.raise_()

    # RWH:  not sure why vtk was being imported here
    # splash.showMessage("Loading VTK modules...",Qt.AlignLeft,  Qt.white)
    # import vtk

    # TODO Fix this - set paths and uncomment
    sys.path.append(os.environ["PYTHON_MODULE_PATH"])
    sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

    import XMLUtils

    versionStr = '3.6.0'
    revisionStr = '0'
    try:
        import Version
        versionStr = Version.getVersionAsString()
        revisionStr = Version.getSVNRevisionAsString()
    except ImportError, e:
        pass
    baseMessage = "CompuCell3D Version: %s Revision: %s\n" % (versionStr, revisionStr)
    firstMessage = baseMessage + "Loading User Interface ..."

    splash.showMessage(firstMessage, Qt.AlignLeft, Qt.white)

    # splash.showMessage("Loading User Interface ...",Qt.AlignLeft,  Qt.white)
    from UI.UserInterface import UserInterface
    from CQt.CQApplication import CQApplication

    # sys.path.append(os.environ["PYTHON_MODULE_PATH"])
    # sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])
    secondMessage = baseMessage + "Loading CompuCell3D Python Modules..."

    splash.showMessage(secondMessage, Qt.AlignLeft, Qt.white)
    # splash.showMessage("Loading CompuCell3D Python Modules...",Qt.AlignLeft,  Qt.white)
    import CompuCellSetup

    CompuCellSetup.playerType = "new"  # the value of CompuCellSetup.playerType (can be "new" or "old") determines which PlayerPython module will be loaded. For the new player5 we want PlayerPythonNew
    import PlayerPython  # from now on import PlayerPython will import PlayerPythonNew

    app.processEvents()

    print 'compucell3d.pyw:   type(argv)=', type(argv)
    print 'compucell3d.pyw:   argv=', argv

    from CMLParser import CMLParser

    cml_parser = CMLParser()
    cml_parser.processCommandLineOptions()
    cml_args = cml_parser.cml_args

    from UI.UserInterface import UserInterface
    from CQt.CQApplication import CQApplication
    mainWindow = UserInterface()
    mainWindow.setArgv(argv)  # passing command line to the code

    # process reminder of the command line options
    # TODO
    if argv != "":
        # mainWindow.viewmanager.processCommandLineOptions(opts)
        mainWindow.viewmanager.processCommandLineOptions(cml_args)

    mainWindow.show()
    splash.finish(mainWindow)

    # 2010: mainWindow.raise_() must be called after mainWindow.show()
    #       otherwise the CC3D player5 GUI won't receive foreground focus. It's a
    #       workaround for a well-known bug caused by PyQt4/Qt on Mac OS X, as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    mainWindow.raise_()

    # mainWindow.showMinimized()
    # mainWindow.showNormal()

    error_code = app.exec_()
    return error_code


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    sys.exit(1)


if __name__ == '__main__':
    # enable it during debugging in pycharm
    sys.excepthook = except_hook

    error_code = main(sys.argv[1:])

    # if error_code !=0 :
    #     print traceback.print_tb()

    # sys.excepthook = except_hook
    print "GOT HERE"
    sys.exit(error_code)
