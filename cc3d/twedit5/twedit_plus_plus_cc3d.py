from os import environ
import platform
import getopt
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.twedit.CQt.CQApplication import CQApplication
from cc3d.twedit5.EditorWindow import EditorWindow
from cc3d.twedit5.DataSocketCommunicators import FileNameSender
import sys
from cc3d.twedit5.windowsUtils import *
from cc3d.twedit5.Messaging import dbgMsg, setDebugging


# this globally enables/disables debug statements
setDebugging(0)

if sys.platform.startswith('win'):
    # this takes care of the need to distribute qwindows.dll with the qt5 application
    # it needs to be located in the directory <library_path>/platforms

    QCoreApplication.addLibraryPath("./bin/")

# instaling message handler to suppres spurious qt messages
if sys.platform == 'darwin':
    mac_ver = platform.mac_ver()
    mac_ver_float = float('.'.join(mac_ver[0].split('.')[:2]))

    if mac_ver_float == 10.11:
        def handler(msg_type, msg_log_context, msg_string=None):

            if msg_log_context.startswith('QCocoaView handleTabletEvent'):
                return

            print(msg_log_context)

    elif mac_ver_float == 10.10:
        def handler(msg_type, msg_log_context, msg_string=None):
            if msg_log_context.startswith('Qt: qfontForThemeFont:'):
                return

            print(msg_log_context)



class Twedit(object):

    def __init__(self):

        self.fileList = []

    def getFileList(self):
        return self.fileList

    def processCommandLineOptions(self):

        print("processCommandLineOptions\n\n\n\n")

        opts = None
        args = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "p", ["file=", "port=", "socket="])
            print("opts=", opts)
            print("args=", args)

        except getopt.GetoptError as err:

            # print help information and exit:

            print(str(err))  # will print something like "option -a not recognized"

            # self.usage()

            sys.exit(2)

        port = 47406

        for o, a in opts:

            print("o=", o)
            print("a=", a)
            if o in ("--port"):
                port = a
                print("THIS IS PORT=", port)

            if o in ("--file"):
                file = a
                print("THIS IS file=", file)

        for a in args:
            self.fileList.append(a)

        print("FILE LIST=", self.fileList)

    def main(self, argv):

        app = CQApplication(argv)

        QApplication.setWindowIcon(QIcon(':/icons/twedit-icon.png'))

        qt_version = str(QT_VERSION_STR).split('.')

        if platform.mac_ver()[0] != '' and int(qt_version[1]) >= 2:  # style sheets may not work properly for qt < 4.2

            app.setStyleSheet("QDockWidget::close-button, QDockWidget::float-button { padding: 0px;icon-size: 24px;}")

        pixmap = QPixmap("icons/lizard-at-a-computer-small.png")

        print("pixmap=", pixmap)

        splash = QSplashScreen(pixmap)

        splash.showMessage("Please wait.\nLoading Twedit++5 ...", Qt.AlignLeft, Qt.black)

        splash.show()

        app.processEvents()

        # app.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))

        self.mainWindow = EditorWindow(False)

        self.mainWindow.setArgv(argv)  # passing command line to the code

        self.mainWindow.show()

        splash.finish(self.mainWindow)

        # self.mainWindow.processCommandLine()

        self.mainWindow.openFileList(self.fileList)

        self.mainWindow.raise_()  # to make sure on OSX window is in the foreground

        if sys.platform.startswith('win'):
            import win32process
            self.mainWindow.setProcessId(win32process.GetCurrentProcessId())
            # showTweditWindowInForeground()

        app.exec_()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    sys.exit(1)


if __name__ == '__main__':

    if __name__ == '__main__':
        # enable it during debugging in pycharm
        sys.excepthook = except_hook

    try:

        twedit = Twedit()

        twedit.processCommandLineOptions()

    except OSError as e:

        dbgMsg("GOT OS ERROR")

        # argvSendSocket=QUdpSocket()

        fileList = twedit.getFileList()

        print("\n\n\n\n FILE LIST=", fileList)

        for fileName in fileList:
            datagram = fileName

            # argvSendSocket.writeDatagram(datagram,QHostAddress.LocalHost,47405)

            fileSender = FileNameSender(datagram)

            fileSender.send()

        if sys.platform == 'win32':

            showTweditWindowInForeground()

        else:

            # notice, on linux you may have to change "focus stealing prevention level" setting to None in
            # window behavior settings , to enable bringing window to foreground

            dbgMsg("NON-WINDOWS PLATFORM - TRY TO ACTIVATE WINDOW")

    twedit.main(sys.argv[1:])
