#!/usr/bin/env python

from PyQt4.QtCore import *

from PyQt4.QtGui import *

from PyQt4.QtNetwork import *

from PyQt4.Qsci import *

from .CQt.CQApplication import CQApplication

from .EditorWindow import EditorWindow



from .DataSocketCommunicators import FileNameSender



import sys, os, tempfile



from .Messaging import dbgMsg, setDebugging

# this globally enables/disables debug statements

setDebugging(0)





from .windowsUtils import *



class Twedit(object):

    def __init__(self):

   

        import sys

        self.lockfile = os.path.normpath(tempfile.gettempdir() + '/' + os.path.basename(__file__) + '.lock')

        if sys.platform == 'win32':

            try:

                # file already exists, we try to remove (in case previous execution was interrupted)

                if(os.path.exists(self.lockfile)):

                    os.unlink(self.lockfile)

                self.fd =  os.open(self.lockfile, os.O_CREAT|os.O_EXCL|os.O_RDWR)

            except OSError as e:

                if e.errno == 13:

                    dbgMsg("Another instance is already running, quitting.")

                    raise

                    

                dbgMsg(e.errno)

                raise

        else: # non Windows

            import fcntl, sys

            

            try:

                self.fp = open(self.lockfile, 'w')

                fcntl.lockf(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)                

            except IOError:

                dbgMsg("NON-WINDOWS PLATFORM Another instance is already running, quitting.")                

                raise OSError

    

    def __del__(self):

        import sys

        if sys.platform == 'win32':

            if hasattr(self, 'fd'):

                os.close(self.fd)

                os.unlink(self.lockfile)





    

    def main(self,argv):

        



        #global mainWindow

        app = CQApplication(argv)

        #app.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))

        self.mainWindow = EditorWindow()

        self.mainWindow.setArgv(argv) # passing command line to the code



        self.mainWindow.show()

        self.mainWindow.processCommandLine()

        self.mainWindow.raise_() # to make sure on OSX window is in the foreground

        if sys.platform.startswith('win'):    

            import win32process

            self.mainWindow.setProcessId(win32process.GetCurrentProcessId())

            

        

        app.exec_()



        





if __name__ == '__main__':



    try:

        twedit=Twedit()

    except OSError as e:

        dbgMsg("GOT OS ERROR")

        

           

        # argvSendSocket=QUdpSocket()

        for fileName in sys.argv[1:]: 

            datagram=fileName

            # argvSendSocket.writeDatagram(datagram,QHostAddress.LocalHost,47405)        

            fileSender=FileNameSender(datagram)  

            fileSender.send()

            

        if sys.platform.startswith('win'):    

            showTweditWindowInForeground()

        else:

            # notice, on linux you may have to change "focus stealing prevention level" setting to None in window behavior settings , to enable bringing windo to foreground 

            dbgMsg("NON-WINDOWS PLATFORM - TRY TO ACTIVATE WINDOW")

        

            

        sys.exit()    



    

    twedit.main(sys.argv[1:])

    

    