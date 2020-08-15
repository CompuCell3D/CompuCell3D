"""
todo fix Socket - rewrite signals to use qt5 style
"""

SIZEOF_UINT16 = 2
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.core.SystemUtils import getCC3DPlayerRunScriptPath
from cc3d.twedit5.windowsUtils import *
import sys
import os
from os import environ

# this class runs inside Qt event loop we can use slots and signals to handle communication
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging


class Socket(QTcpSocket):

    def __init__(self, parent=None):

        super(Socket, self).__init__(parent)

        self.editorWindow = parent.editorWindow

        self.listener = parent

        self.readyRead.connect(self.readRequest)

        self.disconnected.connect(self.listener.maybeCloseEditor)

        self.disconnected.connect(self.deleteLater)

        # self.connect(self, SIGNAL("readyRead()"), self.readRequest)

        # self.connect(self, SIGNAL("disconnected()"), self.listener.maybeCloseEditor)

        # self.connect(self, SIGNAL("disconnected()"), self.deleteLater)

        self.nextBlockSize = 0

        self.line = 0

        self.col = 0

        self.errorLine = -1

        self.bringupTweditPath = None

        if sys.platform.startswith('win'):
            self.bringupTweditPath = os.path.join(environ['PREFIX_CC3D'], 'Twedit++5/bringupTwedit.py')

            self.bringupTweditPath = os.path.abspath(self.bringupTweditPath)

    def disconnectDisconnectedSignal(self):

        self.disconnected.disconnect(self.listener.maybeCloseEditor)

    def connectDisconnectedSignal(self):

        self.disconnected.connect(self.listener.maybeCloseEditor)

    def disconnectReadyReadSignal(self):

        self.readyRead.disconnect(self.readRequest)

    def readRequest(self):

        dbgMsg("INSIDE READ REQUEST")

        stream = QDataStream(self)

        stream.setVersion(QDataStream.Qt_5_2)

        dbgMsg("BYTES AVAILABLE:", self.bytesAvailable())

        if self.nextBlockSize == 0:

            if self.bytesAvailable() < SIZEOF_UINT16:
                return

        self.nextBlockSize = stream.readUInt16()

        if self.bytesAvailable() < self.nextBlockSize:
            return

        action = ""

        fileName = ""

        line = 0

        column = 0

        # date = QDate()

        # stream >> action

        action = stream.readQString()

        dbgMsg("ACTION=", action)

        if str(action) in ("FILEOPEN"):

            fileName = stream.readQString()

            line = stream.readUInt16()

            col = stream.readUInt16()

            self.editorWindow.loadFile(str(fileName))

            currentEditor = self.editorWindow.getCurrentEditor()

            # currentEditor.setCursorPosition(line-1,col)

            currentEditor.setCursorPosition(line - 1, 0)

            dbgMsg("THIS IS FILENAME READ FROM CLIENT=", str(fileName), " line=", line, " col=", col)

            dbgMsg("currentEditor=", " line=", line, " col=", col)

            self.setCurrentLineBackgroundColor(currentEditor)

            self.line = line - 1

            self.col = 0

            # bring up the window

            if sys.platform.startswith('win'):

                # self.editorWindow.activateWindow()

                # aparently

                # showTweditWindowInForeground() will not work because we are trying to set current window in the foreground using win32Api

                # doing the same from separate process works fine

                # have to construct full path from env vars

                # have to get python path here as well

                # from subprocess import Popen

                # p = Popen(["python", self.bringupTweditPath,str(self.editorWindow.getProcessId())])

                print("calling script")



            else:

                self.editorWindow.showNormal()

                self.editorWindow.activateWindow()

                self.editorWindow.raise_()

                self.editorWindow.setFocus(True)



        elif str(action) in ('NEWCONNECTION'):

            print("\n\n\n \t\t\t NEW CONNECTION")

            self.sendEditorOpen()

            # self.sendEditorClosed()

            self.flush()



        elif str(action) in ("CONNECTIONESTABLISHED"):

            print("CONNECTION ESTABLISHED - LISTENER ACKNOWLEDGED")

            self.flush()



        elif str(action) in ("NEWSIMULATIONRECEIVED"):

            print("NEWSIMULATIONRECEIVED SIMULATION NAME SENT SUCCESFULLY")

            self.flush()

            # self.disconnectDisconnectedSignal()

            # self.sendEditorClosed()

            # import time

            # time.sleep(3)

            # self.disconnectFromHost()

            # self.close()

            # self.listener.getOpenPort()

            # dbgMsg("\n\n\n SENDING EDITOR OPEN"    )

            # self.sendEditorOpen()

    def setCurrentLineBackgroundColor(self, currentEditor):

        print("SETTING CARET LINE BACKGROUND")

        print("position=", currentEditor.getCursorPosition())

        line, col = currentEditor.getCursorPosition()

        lineLen = currentEditor.lineLength(line)

        currentEditor.setCaretLineVisible(True)

        # color=currentEditor.SendScintilla(QsciScintilla.SCI_GETCARETLINEBACK)

        # print "COLOR=",color

        # newColor=255 | (0 << 8) | (0 << 16)

        # color=currentEditor.SendScintilla(QsciScintilla.SCI_GETCARETLINEBACK)

        # print "COLOR=",color

        # currentEditor.SendScintilla(QsciScintilla.SCI_SETCARETLINEBACK, newColor)

        # currentEditor.setSelection(line,0,line,lineLen-1)

        # currentEditor.setSelection(line,0,line,0)

        currentEditor.setCaretLineBackgroundColor(QColor('#FE2E2E'))  # current line has this color

        currentEditor.setCaretLineVisible(True)

        currentEditor.hide()

        currentEditor.show()

        # for i in range(1000):

        # currentEditor.setCaretLineBackgroundColor(QtGui.QColor('#FE2E2E')) #current line has this color

        # currentEditor.setCaretLineVisible(True)

        # currentEditor.show()

        # errorBookmark = currentEditor.markerDefine(QsciScintilla.SC_MARK_ARROWS) #All editors tab share same markers

        # errorBookmark = currentEditor.markerDefine(QsciScintilla.SC_MARK_BACKGROUND) #All editors tab share same markers

        # errorBookmark = currentEditor.markerDefine(QsciScintilla.SC_MARK_SHORTARROW) #All editors tab share same markers

        errorBookmark = self.editorWindow.lineBookmark  # All editors tab share same markers

        currentEditor.setMarkerBackgroundColor(QColor("red"), errorBookmark)

        self.errorLine = line

        marker = currentEditor.markerAdd(self.errorLine, errorBookmark)

        print("currentEditor.markersAtLine(self.errorLine)=", currentEditor.markersAtLine(self.errorLine))

        # print self.errorLine

        currentEditor.cursorPositionChanged.connect(self.cursorPositionChangedHandler)

    def cursorPositionChangedHandler(self, line, col):

        print("\n\n\n\n\n\n\n ERROR LINE: ", self.errorLine)

        # dbgMsg("self.line=",self.line, " self.col=",self.col," line=",line," col=",col)

        if line != self.line or col != self.col:

            # dbgMsg("GOING OVER LIST OF EDITORS")

            for editor in self.editorWindow.getEditorList():

                try:  # in case signal is not connected exception is thrown - we simply ignore it

                    self.editorWindow.setEditorProperties(editor)  # restoring original styling for the editor

                    editor.markerDelete(self.errorLine)

                    # # # editor.setCaretLineBackgroundColor(QColor('#E0ECF8'))

                    editor.cursorPositionChanged.disconnect(self.cursorPositionChangedHandler)

                    self.errorLine = -1



                except:

                    pass

    # IMPORTANT: whenever you send message composed of e.g. int, Qstring, Qstring, int  you have to read all of these items otherwise socket state will be corrupted and result in undefined behavior during subsequent reads

    def sendError(self, msg):

        reply = QByteArray()

        stream = QDataStream(reply, QIODevice.WriteOnly)

        stream.setVersion(QDataStream.Qt_5_2)

        stream.writeUInt16(0)

        # stream << QString("ERROR") << QString(msg)

        stream.writeQString("ERROR")

        stream.writeQString(msg)

        stream.device().seek(0)

        stream.writeUInt16(reply.size() - SIZEOF_UINT16)

        self.write(reply)

    def sendEditorClosed(self):

        reply = QByteArray()

        stream = QDataStream(reply, QIODevice.WriteOnly)

        stream.setVersion(QDataStream.Qt_5_2)

        stream.writeUInt16(0)

        # stream << QString("EDITORCLOSED")

        stream.writeQString("EDITORCLOSED")

        stream.device().seek(0)

        print("EDITOR CLOSED SIGNAL SIZE=", reply.size() - SIZEOF_UINT16)

        stream.writeUInt16(reply.size() - SIZEOF_UINT16)

        print("EDITOR CLOSED reply=", reply)

        self.write(reply)

    def sendEditorOpen(self):

        reply = QByteArray()

        stream = QDataStream(reply, QIODevice.WriteOnly)

        stream.setVersion(QDataStream.Qt_5_2)

        stream.writeUInt16(0)

        # stream << QString("EDITOROPEN")

        stream.writeQString("EDITOROPEN")

        stream.writeUInt16(self.editorWindow.getProcessId())

        stream.device().seek(0)

        stream.writeUInt16(reply.size() - SIZEOF_UINT16)

        self.write(reply)

    def sendNewSimulation(self, _simulationName=""):

        reply = QByteArray()

        stream = QDataStream(reply, QIODevice.WriteOnly)

        stream.setVersion(QDataStream.Qt_5_2)

        stream.writeUInt16(0)

        # stream << QString("NEWSIMULATION") <<QString(_simulationName)

        # stream << QString("NEWSIMULATION") << QString(_simulationName)

        stream.writeQString("NEWSIMULATION")

        stream.writeQString(_simulationName)

        stream.device().seek(0)

        print("NEW SIMULATION reply=", reply)

        stream.writeUInt16(reply.size() - SIZEOF_UINT16)

        self.write(reply)

    def sendReply(self, action, room, date):

        reply = QByteArray()

        stream = QDataStream(reply, QIODevice.WriteOnly)

        stream.setVersion(QDataStream.Qt_5_2)

        stream.writeUInt16(0)

        stream << action << room << date

        stream.device().seek(0)

        stream.writeUInt16(reply.size() - SIZEOF_UINT16)

        self.write(reply)


class CC3DListener(QTcpServer):
    newlyReadFileName = QtCore.pyqtSignal(('char*',))

    def __init__(self, parent=None):

        super(CC3DListener, self).__init__(parent)

        self.editorWindow = parent

        # self.port=47406 # initial port - might be reassigned by calling program - vial --port=... command line option

        self.port = -1  # initial port - might be reassigned by calling program - vial --port=... command line option

        self.socketId = -1

        self.port, self.socketId = self.getPortFromCommandLine()

        dbgMsg("PORT=", self.port)

        print("PORT=", self.port)

        self.clientSocket = None

        # on some linux distros QHostAddress.LocalHost does not work

        # if not self.tcpServer.listen(QHostAddress.LocalHost,47405):

        dbgMsg("\n\n\n LISTENING ON PORT ", self.port)

        self.nextBlockSize = 0

        self.socket = None

        self.socketSender = None

        self.nextBlockSize = 0

        self.cc3dPath = getCC3DPlayerRunScriptPath()

        # # # if sys.platform.startswith('win'):

        # # # self.cc3dPath=os.path.join(environ['PREFIX_CC3D'],'compucell3d.bat')

        # # # elif sys.platform.startswith('darwin'):

        # # # self.cc3dPath=os.path.join(environ['PREFIX_CC3D'],'compucell3d.command')

        # # # else : # linux/unix

        # # # self.cc3dPath=os.path.join(environ['PREFIX_CC3D'],'compucell3d.sh')

        # # # self.cc3dPath=os.path.abspath(self.cc3dPath)

        self.pluginObj = None

        self.cc3dProcess = None

        if self.port > 0 and not self.listen(QHostAddress("127.0.0.1"), self.port):
            QMessageBox.critical(None, "FileNameReceiver",

                                 "CONSTRUCTOR Unable to start the server: %s." % str(self.errorString()))

            # self.getOpenPort()

            return

    def setPluginObject(self, plugin):

        self.pluginObj = plugin

    def maybeCloseEditor(self):

        # ret=QtGui.QMessageBox.information(self.editorWindow, "CompuCell3D has been closed","Close editor as well? ",QMessageBox.Yes|QMessageBox.No)

        # if ret==QMessageBox.Yes:

        # self.editorWindow.close()

        if self.socket:
            self.socket.disconnectDisconnectedSignal()

            print("CLOSING LOCAL SOCKET")

            self.socket.disconnectFromHost()

            self.socket = None

            # self.socket=None

            # self.getOpenPort()

        if self.pluginObj:
            self.pluginObj.enableStartCC3DAction(True)

            # self.close()

            # self.startServer()

            # if self.cc3dProcess:

            # print "self.cc3dProcess=",self.cc3dProcess

            # print "dir(self.cc3dProcess)=",dir(self.cc3dProcess)

            # self.cc3dProcess.wait()

            # if self.cc3dProcess:

            # # print "self.cc3dProcess=",self.cc3dProcess

            # print "dir(self.cc3dProcess)=",dir(self.cc3dProcess)

            # self.cc3dProcess.send_signal(SIGTERM)

            # print "self.cc3dProcess.poll()=",self.cc3dProcess.poll()

    def startServer(self):

        port = self.getOpenPort()

        if self.port > 0 and not self.listen(QHostAddress("127.0.0.1"), self.port):

            ret = QMessageBox.critical(None, "FileNameReceiver",

                                       "STARTSERVER Unable to start the server: %s." % str(self.errorString()))

            if ret == QMessageBox.Ok:
                print("\n\n\n START SERVER: SERVER STARTED")

        return port

    def getOpenPort(self):

        print("TRY TO FIGURE OUT PORT\n\n\n\n\n\n")

        port = -1

        for port in range(47406, 47506):

            print("CHECKING PORT=", port)

            tcpServer = QTcpServer(self)

            if tcpServer.listen(QHostAddress("127.0.0.1"), port):
                self.port = port

                tcpServer.close()

                print("established empty port=", self.port)

                break

        return self.port

    def startCC3D(self, _simulationName=""):

        from subprocess import Popen

        print("self.cc3dPath=", self.cc3dPath)

        popenArgs = [self.cc3dPath, "--port=%s" % self.port]

        if _simulationName != "":
            popenArgs.append("-i")

            popenArgs.append(_simulationName)

        # popenArgs.append("-i")

        # popenArgs.append("D:\\Program Files\\COMPUCELL3D_3.5.1_install2\\examples_PythonTutorial\\infoPrinterDemo\\infoPrinterDemo.cc3d" )

        print('Executing Popen command with following arguments=', popenArgs)

        self.cc3dProcess = Popen(popenArgs)

        # self.cc3dProcess = Popen([self.cc3dPath,"--port=%s"%self.port])

        # ,"--tweditPID=%s"%self.editorWindow.getProcessId()

    def getPortFromCommandLine(self):

        import getopt

        import sys

        opts = None

        args = None

        try:

            opts, args = getopt.getopt(sys.argv[1:], "p", ["file=", "port=", "socket="])

            dbgMsg("opts=", opts)

            dbgMsg("args=", args)

        except getopt.GetoptError as err:

            # dbgMsg(help information and exit:)

            dbgMsg(str(err))  # will print something like "option -a not recognized")

            sys.exit(2)

        port = -1

        socketId = 1

        for o, a in opts:

            dbgMsg("o=", o)

            dbgMsg("a=", a)

            if o in ("--port"):
                port = a

                dbgMsg("THIS IS PORT=", port)

            if o in ("--socket"):
                socketId = a

                dbgMsg("THIS IS SOCKET=", socketId)

            if o in ("--file"):
                file = a

                dbgMsg("THIS IS file=", file)

        return int(port), int(socketId)

    def incomingConnection(self, socketId):

        dbgMsg("GOT INCOMMING CONNECTION self.socket=", self.socket)

        sendEditorOpenFlag = False

        if not self.socket:
            sendEditorOpenFlag = True

        self.socket = Socket(self)

        self.socket.setSocketDescriptor(socketId)

        # once we get connection we disable start CC3D action on the tool bar to prevent additional copies of CC3D being open

        if self.pluginObj:
            self.pluginObj.enableStartCC3DAction(False)

        dbgMsg("\n\n\n\n\n socket ID = ", socketId)

    def deactivate(self):

        # print "\n\n\n DEACTIVATING LISTENER"

        # print "listening=",self.isListening()

        if self.socket:
            self.socket.disconnectDisconnectedSignal()

        self.close()

        self.getOpenPort()

        # return

        # self.socket.close()

        if self.socket and self.socket.state() == QAbstractSocket.ConnectedState:
            print("SENDING EDITOR CLOSED SIGNAL")

            self.socket.sendEditorClosed()

            self.socket.waitForReadyRead(3000)

            # self.socket.waitForDisconnected(3000)

        self.close()

