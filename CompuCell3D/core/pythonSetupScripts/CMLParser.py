import sys
class CMLParser:
    def __init__(self):
        self.__screenshotDescriptionFileName=""
        self.customScreenshotDirectoryName=""
        self.__fileName=""
    def getSimulationFileName(self):
        return self.__fileName
        
    def processCommandLineOptions(self):
        import getopt
        self.__screenshotDescriptionFileName=""
        self.customScreenshotDirectoryName=""
        self.outputFrequency=1
        self.outputFileCoreName="Step"
        self.exitWhenDone=False
        self.maxNumberOfRuns=-1
        self.push_address = None
        self.return_value_tag = 'generic_label'
        startSimulation=False
        
        opts=None
        args=None
        try:
            opts, args = getopt.getopt(sys.argv[1:], "i:s:o:f:p:l:c:h", ["help","noOutput","exitWhenDone","currentDir=","outputFrequency=","pushAddress=","returnValueTag","maxNumberOfRuns="])
            print "opts=",opts
            print "args=",args
        except getopt.GetoptError, err:
            # print help information and exit:
            print str(err) # will print something like "option -a not recognized"
            # self.usage()
            sys.exit(1)
        output = None
        verbose = False
        currentDir=""
        for o, a in opts:
            print "o=",o
            print "a=",a
            if o in ("-i"):
                self.__fileName=a
                startSimulation=True
                
            elif o in ("-h", "--help"):
                self.usage()
                return True  # help only
            elif o in ("-s"):
                self.__screenshotDescriptionFileName=a
            elif o in ("-c"):
                self.outputFileCoreName=a                
            elif o in ("-o"):    
                self.customScreenshotDirectoryName=a
                self.__noOutput=False
            elif o in ("--noOutput"):             
                self.__noOutput=True
                self.outputFrequency=0                
            elif o in ("-f","--outputFrequency"):             
                self.outputFrequency=int(a)

            elif o in ("-p","--pushAddress"):
                self.push_address = str(a)
                # print 'GOT pushAddress = ',self.push_address

            elif o in ("-l","--returnValueTag"):
                self.return_value_tag = str(a)
                print 'GOT return_value_tag = ',self.return_value_tag

            elif o in ("--currentDir"):
                currentDir=a
                print "currentDirectory=",currentDir
            elif o in ("--exitWhenDone"):
                self.exitWhenDone=True                    
            elif o in ("--maxNumberOfRuns"):
                self.maxNumberOfRuns=int(a)
                
            
            # elif o in ("--exitWhenDone"):             
                # self.closePlayerAfterSimulationDone=True 
                
            else:
                assert False, "unhandled option"
                
        return False
                
                
    def usage(self):
        print "USAGE: ./runScript.sh -i <simulation file>  -c <outputFileCoreName> "
        print "-o <customVtkDirectoryName>  -f,--frequency=<frequency> "
        print "--noOutput will ensure that no output is stored"
        print "-h or --help will print help message"
