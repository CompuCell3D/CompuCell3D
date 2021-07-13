from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, setDebugging


# This clazs does not use Qt event loop so we have to use blocking statements waitForConnected and waitForDisconnected to make sure

# data was delivered succesfully within given time frame - we give it 3 sec waiting time

class FileNameSender:

    def __init__(self, _fileName):

        self.tcpSocket = QtNetwork.QTcpSocket()

        self.fileName = _fileName

    def send(self):

        self.tcpSocket.abort()

        # on some linux distros QHostAddress.LocalHost does not work

        # self.tcpSocket.connectToHost(QHostAddress.LocalHost,47405)

        self.tcpSocket.connectToHost(QHostAddress("127.0.0.1"), 47405)

        if self.tcpSocket.waitForConnected(3000):

            self.tcpSocket.writeData(self.fileName)

        else:

            dbgMsg("Connection timed out")

        # wait here for tcp server to read fileName

        if self.tcpSocket.waitForDisconnected(3000):

            pass

        else:

            dbgMsg("server busy - did not respond within 3 secs")


# this class runs inside Qt event loop we can use slots and signals to handle communication   


class FileNameReceiver(QObject):
    newlyReadFileName = QtCore.pyqtSignal(('char*',))

    def __init__(self, parent=None):
        super(FileNameReceiver, self).__init__(parent)

        self.tcpServer = QTcpServer(self)

        self.clientSocket = None

        # on some linux distros QHostAddress.LocalHost does not work

        # if not self.tcpServer.listen(QHostAddress.LocalHost,47405):

        if not self.tcpServer.listen(QHostAddress("127.0.0.1"), 47405):
            QtGui.QMessageBox.critical(None, "FileNameReceiverr",

                                       "Unable to start the server: %s." % str(self.tcpServer.errorString()))

            # self.close()

            return

        self.tcpServer.newConnection.connect(self.acceptConnection)

    def acceptConnection(self):
        dbgMsg("ACCEPTING NEW CONNECTION")

        self.clientSocket = self.tcpServer.nextPendingConnection()  # this is connecting tcp socket from the client

        self.clientSocket.disconnected.connect(self.clientSocket.deleteLater)

        self.clientSocket.readyRead.connect(self.readFileName)

    def readFileName(self):
        # fileName=self.clientSocket.readData(bytesInSocket)

        fileName = self.clientSocket.read(self.clientSocket.bytesAvailable())

        dbgMsg("THIS IS FILENAME READ FROM CLIENT=", fileName)

        self.clientSocket.disconnectFromHost()

        self.newlyReadFileName.emit(fileName)
