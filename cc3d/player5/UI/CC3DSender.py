# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
# from PyQt4.QtNetwork import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtNetwork import *
from PyQt5 import *

import sys
import os
from os import environ
import time
from weakref import ref

SIZEOF_UINT16 = 2


class CC3DSender(QObject):
    def __init__(self, _parent=None):
        super(CC3DSender, self).__init__(_parent)

        self.errorConsole = _parent

        self.socket = QtNetwork.QTcpSocket()

        self.socket.connected.connect(self.sendRequest)
        self.socket.readyRead.connect(self.readResponse)
        self.socket.disconnected.connect(self.serverHasStopped)
        self.socket.error.connect(self.serverHasError)

        self.request = "ABC"

        self.port = 47406  # initial port - might be reassigned if this particular value is unavailable'
        # self.port=47407 # initial port - might be reassigned if this particular value is unavailable'
        self.readyToSend = True
        self.editorStarted = False

        # try to figure out the port to communicate on with editor

        for port in range(47406, 47506):

            tcpServer = QTcpServer(self)
            if tcpServer.listen(QHostAddress("127.0.0.1"), port):
                self.port = port
                tcpServer.close()
                break

        self.editorMessageBox = None

        self.tweditCC3DPath = None

        if sys.platform.startswith('win'):
            self.tweditCC3DPath = os.path.join(environ['PREFIX_CC3D'], 'twedit++.bat')
        elif sys.platform.startswith('darwin'):
            self.tweditCC3DPath = os.path.join(environ['PREFIX_CC3D'], 'twedit++.command')
        else:  # linux/unix
            self.tweditCC3DPath = os.path.join(environ['PREFIX_CC3D'], 'twedit++.sh')

        self.tweditCC3DPath = os.path.abspath(self.tweditCC3DPath)
        # checking inf file exists
        try:
            open(self.tweditCC3DPath)
        except:
            self.tweditCC3DPath = None

        self.tweditPID = -1

        self.connectionEstablished = False

        self.fileName_Request = ""
        self.line_Request = 0
        self.column_Request = 0

        self.editorOpenedBySender = False

    @property
    def errorConsole(self):
        try:
            o = self._errorConsole()
        except TypeError:
            o = self._errorConsole
        return o

    @errorConsole.setter
    def errorConsole(self, _i):
        try:
            self._errorConsole = ref(_i)
        except TypeError:
            self._errorConsole = _i

    def setServerPort(self, port):
        self.port = port
        # port is set externally only when editor is already started -
        # although will have to find better solution than this...
        self.editorStarted = True
        self.connectionEstablished = self.establishConnection()

        # def setTweditPID(self,pid):
        # self.tweditPID=pid

    def establishConnection(self):

        print("CALL establishConnection")
        if self.connectionEstablished:
            print("THE CONNECTION HAS ALREADY BEEN ESTABLISHED")
            return
        self.request = QByteArray()
        stream = QDataStream(self.request, QIODevice.WriteOnly)
        stream.setVersion(QDataStream.Qt_5_2)
        stream.writeUInt16(0)

        # stream << "NEWCONNECTION"
        stream.writeQString("NEWCONNECTION")

        stream.device().seek(0)
        stream.writeUInt16(self.request.size() - SIZEOF_UINT16)

        if self.socket.state() != QAbstractSocket.ConnectedState:
            if not self.editorStarted:
                print("BEFORE STARTING EDITOR - port=", self.port)

                self.startEditor()
                self.editorOpenedBySender = True

            connectionEstablished = False
            # On non-windows platforms we will try 10 times to establish connection -
            # sleeping 0.5 seconds between consecutive attempts
            if not sys.platform.startswith('win'):
                for i in range(10):
                    self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)
                    if self.socket.waitForConnected(1000):
                        print("CONNECTED TO HOST")
                        connectionEstablished = True
                        break
                    else:
                        time.sleep(0.5)
                if not connectionEstablished:
                    QMessageBox.warning(None, "Connection Problems",
                                        "Could not connect to Twedit++5. You may try starting CC3D from Twedit++5. <br> twedit++.bat (Windows) or twedit++.sh linux/OSX")
            else:

                self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)

        self.connectionEstablished = True
        return self.connectionEstablished

        # return True    

    def issueOpenFileRequest(self, fileName, line=0, column=0):
        # in case twedit is not installed do not issue any requests
        self.fileName_Request = fileName
        self.line_Request = line
        self.column_Request = column

        print("self.connectionEstablished=", self.connectionEstablished)
        if not self.tweditCC3DPath:
            return

        if not self.connectionEstablished:
            self.connectionEstablished = self.establishConnection()
            return

        print("OPEN FILE REQUEST file=", fileName)

        self.request = QByteArray()
        stream = QDataStream(self.request, QIODevice.WriteOnly)
        stream.setVersion(QDataStream.Qt_5_2)
        stream.writeUInt16(0)

        # stream << QString("FILEOPEN") << QString(fileName)
        stream.writeQString("FILEOPEN")
        stream.writeQString(fileName)
        stream.writeUInt16(line)
        stream.writeUInt16(column)

        stream.device().seek(0)
        stream.writeUInt16(self.request.size() - SIZEOF_UINT16)

        print("TRYING TO CONNECT TO HOST (issueRequest): current state= ", self.socket.state())

        if self.socket.state() != QAbstractSocket.ConnectedState:
            self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)
            print("\n\n\n SOCKET IN THE NON-CONNECTED STATE -  showing splash screen")
            self.editorMessageBox = QtGui.QMessageBox(QMessageBox.Information, "Connecting to Twedit++5.Please wait...",
                                                      "", QMessageBox.Ok)
            pixmap = QPixmap("icons/lizard-at-a-computer-small.png")
            self.editorMessageBox.setIconPixmap(pixmap)
            self.editorMessageBox.show()


        else:
            print("WRITING USING SOCKET WRITE")
            self.socket.flush()
            self.socket.write(self.request)

        import sys
        self.bringupTweditPath = None
        if sys.platform.startswith('win'):
            self.bringupTweditPath = os.path.join(environ['PREFIX_CC3D'], 'Twedit++5/bringupTwedit.py')
            self.bringupTweditPath = os.path.abspath(self.bringupTweditPath)
            from subprocess import Popen

            p = Popen(["python", self.bringupTweditPath, str(self.tweditPID)])
            print("\n\n\n\n\n\n\n\n\n\n tweditPID=", self.tweditPID)

        print("\n\n\n\n\n\n\n\n\n\n SENDER(issueRequest): self.socket.socketDescriptor()=",
              self.socket.socketDescriptor())

    def sendToEditor(self, _request):
        self.request = _request
        self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)
        print("SENDING TO EDITOR MESSAGE: ", self.request)

    def sendRequest(self):

        print("SENDING REQUEST")
        self.nextBlockSize = 0
        self.socket.write(self.request)
        self.request = None

    def readResponse(self):
        print("READING RESPONSE CC3D SENDER")
        stream = QDataStream(self.socket)
        stream.setVersion(QDataStream.Qt_5_2)

        # print "BEFORE INTERCEPTING self.socket.bytesAvailable()=",self.socket.bytesAvailable()
        # msgStr=QString()
        # msgStr1=QString()
        # size=stream.readUInt16()
        # print "SIZE MESSAGE=",size
        # # stream.skipRawData(2)

        # stream>>msgStr>>msgStr1
        # print "msgStr=",msgStr
        # print "msgStr1=",msgStr1

        # byteArray=QByteArray()
        # byteArray.append(msgStr)
        # if str(msgStr)=="EDITOROPEN":
        # self.tweditPID=stream.readUInt16()

        # print "self.tweditPID=",self.tweditPID

        # stream = QDataStream(byteArray)
        # stream.setVersion(QDataStream.Qt_4_2)

        # reply = QByteArray()
        # stream1 = QDataStream(reply, QIODevice.WriteOnly)
        # stream1.setVersion(QDataStream.Qt_4_2)            
        # stream1.writeUInt16(0)
        # stream1 <<QString("CONNECTIONESTABLISHED")             
        # stream1.device().seek(0)
        # stream1.writeUInt16(reply.size() - SIZEOF_UINT16)            

        # self.socket.write(reply)
        # self.socket.flush()

        # return

        print("self.socket.bytesAvailable()=", self.socket.bytesAvailable())
        if self.nextBlockSize == 0:
            if self.socket.bytesAvailable() < SIZEOF_UINT16:
                return

        self.nextBlockSize = stream.readUInt16()
        print("self.nextBlockSize=", self.nextBlockSize)
        if self.socket.bytesAvailable() < self.nextBlockSize:
            msg = ''
            # stream >> msg
            msg = stream.readQString()
            print("message=", msg)
            return

        print("self.socket.bytesAvailable()=", self.socket.bytesAvailable())
        messageType = ''

        # stream >> messageType
        messageType = stream.readQString()
        print("\n\n\n\n messageType=", messageType)

        if messageType == "EDITORCLOSED":

            if self.editorOpenedBySender:
                # do not close CC3D if this instance opened the editor
                self.socket.close()
                return

            print("EDITOR WAS CLOSED")
            self.socket.close()
            self.errorConsole.emitCloseCC3D()
            return

            # import sys

            # sys.exit()
            # self.errorConsole.closeCC3D()
            # self.socket.flush()
            # self.socket.disconnectFromHost()
            # self.socket.close()
            # self.socket=None
            # return

            QMessageBox.information(None, "EDITOR WAS CLOSED", "EDITOR CLOSED: ")
            reply = QByteArray()
            stream1 = QDataStream(reply, QIODevice.WriteOnly)
            stream1.setVersion(QDataStream.Qt_5_2)
            stream1.writeUInt16(0)
            # self.socket.close()
            self.socket.write(reply)


        elif messageType == "EDITOROPEN":
            print("GOT EDITOROPEN MESSAGE")
            if self.editorMessageBox:
                self.editorMessageBox.close()

            self.tweditPID = stream.readUInt16()

            print("self.tweditPID=", self.tweditPID)

            # self.socket.disconnectFromHost()
            # return

            reply = QByteArray()
            stream1 = QDataStream(reply, QIODevice.WriteOnly)
            stream1.setVersion(QDataStream.Qt_5_2)
            stream1.writeUInt16(0)
            # stream1 << QString("CONNECTIONESTABLISHED")
            stream1.writeQString('CONNECTIONESTABLISHED')
            stream1.device().seek(0)
            stream1.writeUInt16(reply.size() - SIZEOF_UINT16)

            self.socket.write(reply)
            if not sys.platform.startswith('win'):
                import time
                time.sleep(0.5)

            if self.fileName_Request != "":
                print("WILL TRY RESENDING OPEN FILE REQUEST")
                self.issueOpenFileRequest(self.fileName_Request, self.line_Request, self.column_Request)
                self.fileName_Request = ""
                self.editorStarted = True

        elif messageType == "NEWSIMULATION":
            newSimulation = ''
            # stream >> newSimulation
            newSimulation = stream.readQString()
            reply = QByteArray()
            stream1 = QDataStream(reply, QIODevice.WriteOnly)
            stream1.setVersion(QDataStream.Qt_5_2)
            stream1.writeUInt16(0)
            # stream1 << QString("NEWSIMULATIONRECEIVED")
            stream1.writeQString('NEWSIMULATIONRECEIVED')
            stream1.device().seek(0)
            stream1.writeUInt16(reply.size() - SIZEOF_UINT16)
            print("self.errorConsole.playerMainWidget=", self.errorConsole.playerMainWidget)
            if self.errorConsole.playerMainWidget:
                if self.errorConsole.playerMainWidget.simulationIsRunning or self.errorConsole.playerMainWidget.simulationIsStepping:
                    message = "Current simulation is still running.<br>Do you want to stop it immediately and run new simulation?<br>If you choose not to stop new simulation will be queued"
                    ret = QMessageBox.warning(self.errorConsole.playerMainWidget, "Simulation still running", message,
                                              QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        self.errorConsole.playerMainWidget.processIncommingSimulation(newSimulation, True)
                    else:
                        self.errorConsole.playerMainWidget.processIncommingSimulation(newSimulation, False)

                else:
                    self.errorConsole.playerMainWidget.processIncommingSimulation(newSimulation, True)

            else:
                print("self.errorConsole.playerMainWidget was not initialized")

            self.socket.write(reply)
            if not sys.platform.startswith('win'):
                import time
                time.sleep(0.5)

            # self.socket.flush()
            print("GOT NEW SIMULATION ", newSimulation)

        self.socket.flush()
        return

    def serverHasStopped(self):
        print("SERVER HAS STOPPED")
        self.editorStarted = False
        self.connectionEstablished = False

        print("\t\t\t\t SERVER HAS STOPPED: self.editorStarted=", self.editorStarted, "\n\n\n")
        self.socket.close()
        # self.connect(self.socket,SIGNAL("error(QAbstractSocket::SocketError)"),self.serverHasError)         

        # import sys
        # sys.exit()

    def startEditor(self):
        self.socket.abort()
        from subprocess import Popen
        print("self.socket.socketDescriptor()=", self.socket.socketDescriptor())
        # p = Popen([self.tweditCC3DPath, "--port=%s " % self.port, "--socket=%s" % self.socket.socketDescriptor()])

        # turns out socket descriptor is not used anywhere
        # sending -1 for now but should eliminate this extra argument altogether

        p = Popen([self.tweditCC3DPath, "--port=%s " % self.port, "--socket=%s" % str(-1)])

        # p = Popen(["python", self.tweditCC3DPath,"--port=%s "%self.port,"--socket=%s"%self.socket.socketDescriptor()])
        # p = Popen(["python", "D:\\Project_SVN_CC3D\\branch\\twedit++\\twedit_plus_plus_cc3d.py","--port=%s "%self.port,"--socket=%s"%self.socket.socketDescriptor()])
        print("\n\n\n\n\STARTED TWEDIT++\n\n\n\n\n")

        # self.editorStarted=True

    def serverHasError(self, error):

        print("SERVER ERROR")
        print("error=", error)

        self.socket.abort()
        return

        print("\t\t\t\t self.editorStarted=", self.editorStarted, "\n\n\n")
        if not self.editorStarted:

            self.socket.error.disconnect(self.serverHasError)
            self.startEditor()
            self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)
            self.socket.error.conect(self.serverHasError)
        else:
            self.socket.close()

        # if not self.editorStarted:
        #     self.disconnect(self.socket, SIGNAL("error(QAbstractSocket::SocketError)"), self.serverHasError)
        #     self.startEditor()
        #     self.socket.connectToHost(QHostAddress("127.0.0.1"), self.port)
        #     self.connect(self.socket, SIGNAL("error(QAbstractSocket::SocketError)"), self.serverHasError)
        # else:
        #     self.socket.close()
