#!/usr/bin/env python


def setPaths():
    from os import environ
    import sys
    platform=sys.platform
    if platform.startswith('win'):
        try:
            sys.path.insert(0,environ["PYTHON_DEPS_PATH"])
            sys.path.append(environ["SWIG_LIB_INSTALL_DIR"])
            sys.path.append(environ["PYTHON_MODULE_PATH"])

        except:
            pass
    else:
        try:
            sys.path.append(environ["PYTHON_MODULE_PATH"])
            sys.path.append(environ["SWIG_LIB_INSTALL_DIR"])
        except:
            pass

setPaths()

import sys

import PyQt5
from utils.global_imports import *

if sys.platform.startswith('win'):
	# this takes care of the need to distribute qwindows.dll with the qt5 application
	# it needs to be locarted in the directory <library_path>/platforms
	QCoreApplication.addLibraryPath("./bin/")


#instaling message handler to suppres spurious qt messages
if sys.platform=='darwin':
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



from CQt.CQApplication import CQApplication
from EditorWindow import EditorWindow

from DataSocketCommunicators import FileNameSender

import sys
from windowsUtils import *

from Messaging import dbgMsg, setDebugging


# this globally enables/disables debug statements
setDebugging(0)



class Twedit(object):
    def __init__(self):
        self.fileList=[]
        return
        # import sys
        # self.lockfile = os.path.normpath(tempfile.gettempdir() + '/' + os.path.basename(__file__) + '.lock')
        # if sys.platform == 'win32':
            # try:
                # # file already exists, we try to remove (in case previous execution was interrupted)
                # if(os.path.exists(self.lockfile)):
                    # os.unlink(self.lockfile)
                # self.fd =  os.open(self.lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)
            # except OSError, e:
                # if e.errno == 13:
                    # dbgMsg("Another instance is already running, quitting.")
                    # raise

                # dbgMsg(e.errno)
                # raise
        # else: # non Windows
            # import fcntl, sys

            # try:
                # self.fp = open(self.lockfile, 'w')
                # fcntl.lockf(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # except IOError:
                # dbgMsg("NON-WINDOWS PLATFORM Another instance is already running, quitting.")
                # raise OSError

    # def __del__(self):

        # import sys
        # if sys.platform == 'win32':
            # if hasattr(self, 'fd'):
                # os.close(self.fd)
                # os.unlink(self.lockfile)

    def getFileList(self):
        return self.fileList

    def processCommandLineOptions(self):
        import getopt

        print "processCommandLineOptions\n\n\n\n"
        opts=None
        args=None
        try:
            opts, args = getopt.getopt(sys.argv[1:], "p", ["file=","port=","socket="])
            print "opts=",opts
            print "args=",args
        except getopt.GetoptError, err:
            # print help information and exit:
            print str(err) # will print something like "option -a not recognized"
            # self.usage()
            sys.exit(2)
        port=47406

        for o, a in opts:
            print "o=",o
            print "a=",a
            if o in ("--port"):
                port=a
                print "THIS IS PORT=",port
            if o in ("--file"):
                file=a
                print "THIS IS file=",file

        for  a in args:
            self.fileList.append(a)
            # elif o in ("--exitWhenDone"):
                # self.closePlayerAfterSimulationDone=True

            # else:
                # print "GOT ARGUMENT:", a
                # assert False, "unhandled option"
        print "FILE LIST=",self.fileList


    def main(self,argv):


        #global mainWindow
        app = CQApplication(argv)

        qtVersion=str(QT_VERSION_STR).split('.')
        import platform

        if platform.mac_ver()[0]!='' and qtVersion[1]>=2: # style sheets may not work properly for qt < 4.2
            app.setStyleSheet( "QDockWidget::close-button, QDockWidget::float-button { padding: 0px;icon-size: 24px;}")







        pixmap = QPixmap("icons/lizard-at-a-computer-small.png")
        print "pixmap=",pixmap
        splash = QSplashScreen(pixmap)
        splash.showMessage("Please wait.\nLoading Twedit++5 ...",Qt.AlignLeft,  Qt.black)
        splash.show()

        app.processEvents()
        #app.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))
        self.mainWindow = EditorWindow(False)
        self.mainWindow.setArgv(argv) # passing command line to the code

        self.mainWindow.show()
        splash.finish(self.mainWindow)

        # self.mainWindow.processCommandLine()
        self.mainWindow.openFileList(self.fileList)
        QApplication.setWindowIcon(QIcon("Twedit++5/icons/twedit-icon.png"))

        self.mainWindow.raise_() # to make sure on OSX window is in the foreground

        if sys.platform.startswith('win'):
            import win32process
            self.mainWindow.setProcessId(win32process.GetCurrentProcessId())
            # showTweditWindowInForeground()

        app.exec_()



if __name__ == '__main__':

    try:
        twedit=Twedit()
        twedit.processCommandLineOptions()
    except OSError,e:
        dbgMsg("GOT OS ERROR")

        # argvSendSocket=QUdpSocket()
        fileList=twedit.getFileList()
        print "\n\n\n\n FILE LIST=",fileList
        for fileName in fileList:
            datagram=fileName
            # argvSendSocket.writeDatagram(datagram,QHostAddress.LocalHost,47405)
            fileSender=FileNameSender(datagram)
            fileSender.send()

        if sys.platform == 'win32':
            showTweditWindowInForeground()
        else:
            # notice, on linux you may have to change "focus stealing prevention level" setting to None in window behavior settings , to enable bringing window to foreground
            dbgMsg("NON-WINDOWS PLATFORM - TRY TO ACTIVATE WINDOW")


        # sys.exit()


    twedit.main(sys.argv[1:])
