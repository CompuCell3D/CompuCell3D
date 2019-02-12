"""
=================================
Batch Testing Script
=================================
This script runs all the .cc3d files in a directory.
Use this file in the command line with your 2.7.* python command.
You can use the command line args "-o  [Path To Directory]" for directing output, 
"-i [Path To Directory]" for directing input, and "-player" to determine if you 
want to use the player. "-o" and "-i" are expected to be given directories
For "-o" the program will send your test result as well as your normal CompuCell3D 
data to that directory. The default location for your test output should be found 
in a folder in this directory.
Ex.
python batch_test_init.py -i [Input Directory] -o [Output Directory]
Uses Demos directory (one folder up from the location of the batch_test_init.py file)
python batch_test_init.py
"""
import os
import time
import platform
import sys
import shutil

#get the current time DD-MM-YYYY_H-MM
timeStart = time.strftime("%d-%m-%Y_%H-%M")
#default output directory for test results
testOutput = "Demos/Testing/"
#keeps track which demo the program is on
demoNumber = 0

#Int Boolean -> Void
#appends the return code and whether the demo produced a performance report, End Of Simulation or an error.
def appendReturnCode(returnCode, hasPerformanceReport, hasEndOfSimulation, error):
    if hasPerformanceReport:
        reportTag = "PR"
    elif hasEndOfSimulation:
        reportTag = "EOS"
    else:
        reportTag = "ERROR"
    f = open(testOutput + "ProcessedDemos" + timeStart + platform.system() + ".txt", "r")
    contents = ""
    # looks for the current demo and inserts the return value
    for line in f:
        if path in line:
            contents += "[Return Code: " + str(returnCode) + "]\t" + line[:-1] + "\t[" + reportTag + "] " + error + "\n"
        else:
            contents += line
    f.close()

    f = open(testOutput + "ProcessedDemos" + timeStart + platform.system() + ".txt", "w")
    f.writelines(contents)
    f.close()


print("Processing...  This could take a VERY long time...")

#List[String] -> Void
#Sends the command to the shell stores the output depending on the outcome of the demo
def callInShell(arguments):
    command = ""
    #adds a space for each command
    for _arg in arguments:
        command += _arg + " "
    commandOutput = []
    try:
        for line in os.popen(command, "r"):
            commandOutput.append(line)
    except os.error as e:
        print("#Error code:", e.returncode, e.output)
        f = open(testOutput + tail[0:-5] + ".txt", "w")
        f.writelines("Error Code:" + "\n" + str(e.returncode) + "\n" + e.output)
        f.close()

    numberOfLinesInOutput = 0
    #creates a list that stores a performance report if there is one
    demoOutputToList = []
    performanceReportExists = False
    hasEndofSimulation = False
    #adds a path to it as an identifier
    demoOutputToList.append("SUCCESSFUL DEMO OUTPUT[DEMO #" + str(demoNumber) + "]: " + path + "\n")
    #doesn't add lines until it encounters a performance report
    for line in commandOutput:
        numberOfLinesInOutput += 1
        if "------------------PERFORMANCE REPORT:----------------------" in line or performanceReportExists:
            performanceReportExists = True
            demoOutputToList.append(line)
        if "END OF SIMULATION" in line:
            hasEndofSimulation = True
    # if the demo had a performance report
    if performanceReportExists:
        appendReturnCode(0, True, False, "")
        #writes the performance report
        with open(testOutput + "SuccessfulResults" + timeStart + platform.system() + ".txt", "a") as f:
            for line in demoOutputToList:
                f.write(line)
            f.write("\n\n")
        f.close()
    elif hasEndofSimulation:
        appendReturnCode(0, False, True, "")
        with open(testOutput + "SuccessfulResults" + timeStart + platform.system() + ".txt", "a") as f:
            f.write("--------------------------------------------------------------------------------------------------\n")
            f.write("[DEMO #" + str(demoNumber) + "]: " + path + "\n")
            f.write("The demo ran to completion but did not produce an performance report.\n")
            f.write("--------------------------------------------------------------------------------------------------\n")
            f.write("\n\n")
        f.close()
    #if the demo did not have a performance report
    else:

        if not os.path.exists(testOutput + "UnexpectedResults" + timeStart + platform.system()  + ".txt"):
            f = open(testOutput + "UnexpectedResults" + timeStart + platform.system() + ".txt", "w")

            f.write(platform.system() + " version: " + platform.version() + "  " + platform.machine() + "\n")
            if os.path.isfile("../../../CompuCell3D-64bit website.url"):
                f.write("CompuCell3D-64bit\n")
            elif os.path.isfile("../../../CompuCell3D-32bit website.url"):
                f.write("CompuCell3D-32bit\n")
            f.write("These are demos that have given an error or have ran to completion and did not provide a performance report.\n")
            f.close()

        #opens the UnexpectedResults file and writes the last 20 lines of the output from the demo
        with open(testOutput + "UnexpectedResults" + timeStart + platform.system() + ".txt", "a") as testResult:
            #tags location in UnexpectedResults with an identifier
            testResult.write("--------------------------------------------------------------------------------------------------\n")
            testResult.write("UNEXPECTED OUTPUT[DEMO #" + str(demoNumber) + "]: " + path + "\n")
            testResult.write("--------------------------------------------------------------------------------------------------\n")
            testResult.write("OUTPUT: \n")
            #writes last 20 lines of the demo file
            linesToBeRemoved = numberOfLinesInOutput - 10
            error = ""
            for line in commandOutput:
                if linesToBeRemoved <= 0:
                    testResult.write(line)
                    linesToBeRemoved -= 1
                    if linesToBeRemoved == -8 and platform.system() == "Windows":
                        error = line[:-1]
                    elif linesToBeRemoved == -9:
                        error = line[:-1]
                else:
                    linesToBeRemoved -= 1
            testResult.write("\n\n")
            testResult.close()
        appendReturnCode(1, False, False, error)
    #if the user specifies no output ensure that if CC3D creates output to delete it
    if noOutput and os.path.isdir(outputDirectory + "/" + tail[0:-5] + timeStart):
        shutil.rmtree(outputDirectory + "/" + tail[0:-5] + timeStart)


#Void -> Void
#Creates a command to send the CallinShell function
def createCommand():
    #checks if the player is going to be used
    if player:
        arguments = [setup + "compucell3d" + ext, "--exitWhenDone"]
    else:
        arguments = [setup + "runScript" + ext]
    #Takes the Demos directory by default unless specified with -i
    arguments.append("-i")
    arguments.append(path)
    #Sends output to CC3DWorkspace by default unless specified with -o
    arguments.append("-o")
    #Check the output directory exists if not create it
    if not os.path.isdir(outputDirectory + "/" + tail[0:-5] + timeStart):
        os.makedirs(outputDirectory + "/" + tail[0:-5] + timeStart)
    arguments.append(outputDirectory + "/" + tail[0:-5] + timeStart)

    #deletes output from CC3D
    if noOutput:
        arguments.append("--noOutput")

    #creates the arguments to be called
    for arg in args:
        arguments.append(arg)

    callInShell(arguments)
#String -> Void

def createNewInstance(commandForInstance):
    os.system(commandForInstance)

# String -> Void
# writes the used to demos to a file
def storeProcessedDemos(demoPath):
    with open(testOutput + "ProcessedDemos" + timeStart + platform.system()  + ".txt", 'a') as processedDemos:
        processedDemos.writelines(demoPath + "\n")
    processedDemos.close()


# String -> Boolean
# checks if a path has been processed yet
# TODO: this was mean't for multi-instancing
def notInProcessedDemos(demoPath):
    with open(testOutput + "ProcessedDemos" + timeStart + platform.system() + ".txt", 'r') as processedDemos:
        for line in processedDemos:
            if demoPath in line:
                return False
        return True


#defaults
player = False
customOutput = False
noOutput = False
#assign indicated args to a list
args = sys.argv
del args[0]
inputDirectory = "Demos"
outputDirectory = "../CC3DWorkspace"
#allows the initial instance run through
if "__batchTestStartTime__" in args:
    time.sleep(2)
#help command args for this file
i = 0
#goes through the given arguments
while i < len(args):
    #outputs help menu
    if "-h" in args or "--help" in args:
        print("""
This script tests all the .cc3d scripts in a directory and collects output
and determines whether the .cc3d files failed or ran to completion. See the 
documentation located in this directory for more details.
        -h shows a list of commands
        -i [Path to Directory] Determines which directory gets searched for .cc3d files
        -o [Path to Directory] Specifies where vtk/screenshot files will be written 
        -testOutput [Path to Directory] Specifies where the test results file will be located
        --noOutput Instructs CC3D to not store any project output, but tests will be saved
        --player Uses the player if you want to have screenshots
Ex.
    python batch_test_init.py -i ../../Demos -o ../../../CC3DWorkspace --noOutput  
    python batch_test.init.py          
        """)
        exit(0)
    #takes a directory as opposed to a file location
    elif args.count("-i") == 1 and args[i] == "-i":
        inputDirectory = os.path.abspath(args[args.index("-i") + 1])
        #delete their locations for convenience later
        del args[args.index("-i") + 1]
        del args[args.index("-i")]
    #sets a custom output for Demo data
    elif args.count("-o") == 1 and args[i] == "-o":
        customOutput = True
        outputDirectory = os.path.abspath(args[args.index("-o") + 1])
        #delete their locations for convenience later
        del args[args.index("-o") + 1]
        del args[args.index("-o")]
    #sets a custom output for test data
    elif args.count("-testOutput") == 1 and args[i] == "-testOutput":
        testOutput = os.path.abspath(args[args.index("-testOutput") + 1]) + "/"
        #delete their locations for convenience later
        del args[args.index("-testOutput") + 1]
        del args[args.index("-testOutput")]

    #signals that the user wants to use the player and removes any --exitWhenDone commands
    elif args.count("--player") == 1 and args[i] == "--player":
        player = True
        del args[args.index("--player")]
        #makes sure --exitWhenDone isn't in the arguments they gave
        if "--exitWhenDone" in args:
            del args[args.index("--exitWhenDone")]

    #sets up the tests not to send any output to the CC3DWorkspace
    elif args.count("--noOutput") == 1 and args[i] == "--noOutput":
        noOutput = True
        del args[args.index("--noOutput")]
    #throw an error if the arg isn't found
    else:
        print("Error: near " + args[i] + " is an invalid command or sequence!")
        print("Use the --help command to see a list of commands")
        exit(1)

#changes the working directory to the top directory
os.chdir("../../")

#stores system information in SuccessfulResults
if not os.path.exists(testOutput + "SuccessfulResults" + timeStart + platform.system()  + ".txt"):
    f = open(testOutput + "SuccessfulResults" + timeStart + platform.system() + ".txt", "w")
    f.write(platform.system() + " version: " + platform.version() + "  " + platform.machine() + "\n")
    if os.path.isfile("../../../CompuCell3D-64bit website.url"):
        f.write("CompuCell3D-64bit\n")
    elif os.path.isfile("../../../CompuCell3D-32bit website.url"):
        f.write("CompuCell3D-32bit\n")
    f.close()

#creates a place to store used demos so that multiple instances of this program be ran
#also gets the initially used time if its been created already
if not os.path.isfile(testOutput + "ProcessedDemos" + timeStart + platform.system() + ".txt"):
    f = open(testOutput + "ProcessedDemos" + timeStart + platform.system() + ".txt", "w")
    f.write(timeStart + "\n")
    f.write("    EOS: (End Of Simulation) The demo did not produce a performance report, but ran successfully.\n")
    f.write("     PR: (Performance Report) The demo produced a performance report and ran successfully.\n")
    f.write("  ERROR: The demo produced an error.\n\n")
    f.close()
else:
    f = open(testOutput +"ProcessedDemos" + timeStart + platform.system() + ".txt", "r")
    timeStart = file.readline()
    timeStart = timeStart[:-1]
    f.close()

#Collects all .cc3d files
demos = [os.path.join(dp, file) for dp, dn, fileNames in os.walk(inputDirectory) for file in fileNames if os.path.splitext(file)[1] == ".cc3d"]

#Check for OS
if platform.system() == "Linux":
    for path in demos:
        demoNumber += 1
        #Gets filename
        if notInProcessedDemos(path):
            storeProcessedDemos(path)
            head, tail = os.path.split(path)
            ext = ".sh"
            setup = "./"
            createCommand()
elif platform.system() == "Windows":
    for path in demos:
        print("Running " + path)
        print("Don't kill the process!")
        demoNumber += 1
        #Gets filename
        if notInProcessedDemos(path):
            storeProcessedDemos(path)
            head, tail = os.path.split(path)
            ext = ".bat"
            setup = os.getcwd() + "/"
            createCommand()
        print("Finished")
elif platform.system() == "Darwin":
    for path in demos:
        demoNumber += 1
        #Gets filename
        if notInProcessedDemos(path):
            storeProcessedDemos(path)
            head, tail = os.path.split(path)
            ext = ".command"
            setup = "./"
            createCommand()
