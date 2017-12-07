
import ProjectFileStore
import os
import csv
import numpy as np
import shutil
import subprocess

class ParameterScanner:

    parameters = {}
    parameterSetDictionaryList = []

    compuCell3DExecutable = "CC3D_CLI"

    # Used to create slurm script for each simulation
    inputFilePath = ""
    outputDirectory = ""
    cc3dOutputFilePath = ""

    slurmScriptName =  "parameter_scan.sh"
    stdOuputFileName = "stdout.txt"
    isSlurmScriptCreated = False


    def readParameters(self):
        '''
        Read parameters from the file to scan over
        :return:
        '''
        parameterFile = ProjectFileStore.parameterScanFile
        if os.path.isfile(parameterFile) == False:
            print parameterFile, "Parameter File does not exists. Please specify valid file"
            exit(1)

        with open(parameterFile, 'r') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV, None)
            for row in readCSV:
                #print row
                parameterName = row[0]
                minValue = float(row[1])
                maxValue = float(row[2])
                step = float(row[3])
                values = np.arange(minValue,maxValue,step)
                self.parameters[parameterName] = values

        self.parameterSetDictionaryList = self.buildParameterSetDictionary(self.parameters)
        print "There are ", len(self.parameterSetDictionaryList), " number of parameter combinations."

    def buildParameterSetDictionary(self, parameterDict):
        parameterName, values = parameterDict.popitem()
        if len(parameterDict) == 0:
            parmeterSetList = []
            for value in values:
                parmeterSetList.append({parameterName: value})
            return parmeterSetList
        else:
            childParmeterSetList = self.buildParameterSetDictionary(parameterDict)
            parmeterSetList = []
            for value in values:
                for childParameterSet in childParmeterSetList:
                    parameterSet = {parameterName: value}
                    parameterSet.update(childParameterSet)
                    parmeterSetList.append(parameterSet)
        return parmeterSetList


    def createParameterFile(self, parameterFilePath, parameterSet):
        #print parameterSet
        with open(parameterFilePath, 'w') as f:
            for parameterName, value in parameterSet.iteritems():
                f.write(parameterName + " = " + str(value) + "\n")

    def copyCC3DProject(self, targetDirectory):

        projectDirectory = os.path.dirname(ProjectFileStore.projectFilePath)
        projectSimulationDirectory = os.path.join(os.path.sep, projectDirectory, "Simulation")
        targetSimulationDirectory = os.path.join(os.path.sep, targetDirectory, "Simulation")
        shutil.copytree(projectSimulationDirectory, targetSimulationDirectory)

        shutil.copy(ProjectFileStore.projectFilePath, targetDirectory)

        self.inputFilePath = os.path.join(os.path.sep, targetDirectory, os.path.basename(ProjectFileStore.projectFilePath))
        self.outputDirectory = os.path.join(os.path.sep, targetDirectory, "output")
        self.cc3dOutputFilePath = os.path.join(os.path.sep, targetDirectory, "output.txt")

    def createSimulationFolders(self):
        '''
        This will create multiple simulation folders with individual parameters
        :return:
        '''
        parameterScriptName = "Parameters.py"
        outputDirectory = ProjectFileStore.outputDirectoryPath
        count = 0
        for parameterSet in self.parameterSetDictionaryList:
            simulationDirectory = os.path.join(os.path.sep, outputDirectory, str(count))
            #os.makedirs(simulationDirectory)
            self.copyCC3DProject(simulationDirectory)


            parameterFilePath = os.path.join(os.path.sep, simulationDirectory, "Simulation", parameterScriptName)
            os.remove(parameterFilePath)
            self.createParameterFile(parameterFilePath, parameterSet)

            self.writeToSlurmScript()
            count += 1


    def writeToSlurmScript(self):
        if self.isSlurmScriptCreated == False:
            self.createBatchScriptForSlurm()

        slumScriptPath = os.path.join(os.path.sep, ProjectFileStore.outputDirectoryPath, self.slurmScriptName)
        with open(slumScriptPath, "a") as script:
            script.write("srun -o " +  self.cc3dOutputFilePath + " " + \
                         self.compuCell3DExecutable + " -i " + self.inputFilePath + " -o " + self.outputDirectory + " -f " + str(ProjectFileStore.outputFrequency) + "\n")

    def createBatchScriptForSlurm(self):
        '''
        This will create a batch script for Slurm job manager
        :return:
        '''
        slumScriptPath = os.path.join(os.path.sep, ProjectFileStore.outputDirectoryPath, self.slurmScriptName)
        stdOutputFilePath = os.path.join(os.path.sep, ProjectFileStore.outputDirectoryPath, self.stdOuputFileName)
        with open(slumScriptPath, 'w') as f:
            f.write("#! /usr/bin/sh\n")
            f.write("#SBATCH --mail-type=ALL\n")
            f.write("#SBATCH --mail-user=anshaikh@iu.edu\n")
            f.write("#SBATCH -o " + stdOutputFilePath + "\n")
            f.write("#SBATCH --nodes=8 \n")
        self.isSlurmScriptCreated = True


    def executeSimulations(self):
        '''
        This function will execute simulations
        :return:
        '''
        slumScriptPath = os.path.join(os.path.sep, ProjectFileStore.outputDirectoryPath, self.slurmScriptName)
        #subprocess.call("chmod +x " + slumScriptPath)
        subprocess.call(["sbatch", slumScriptPath])

    def scan(self):
        print "Running Parameter Scan"
        print "Reading Parameter File"
        self.readParameters()
        print "Creating Required directories..."
        self.createSimulationFolders()
        print "Submitting Jobs to Scheduler..."
        self.executeSimulations()