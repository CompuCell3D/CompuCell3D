
import ProjectFileStore
import os
import csv
import numpy as np
import shutil
import subprocess
import pandas as pd

class ParameterScannerExtensiveInput:

    '''
    This is a parameter scanner module which takes input in the form of .csv which is extensive (i.e. more than just
    ranges). The format for the input csv is mentioned as below

    || simulation_id | param_1 | param_2 | param_3 | ... ||
    ||      1        |  2.5    |    4    |  100    | ... ||

    Where each row of the csv is an individual instance of CompuCell3D simulation.
    '''

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

        self.parameter_data = pd.read_csv(parameterFile)
        print "There are ", len(self.parameter_data.index), " number of parameter combinations."

    def is_number(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def createParameterFile(self, parameterFilePath, parameterRow):
        #print parameterSet
        with open(parameterFilePath, 'w') as f:
            for parameterName, value in parameterRow.iteritems():
                if not self.is_number(value):
                    value = "'" + str(value) + "'"
                f.write(parameterName + " = " + str(value) + "\n")

    def copyCC3DProject(self, targetDirectory):

        projectDirectory = os.path.dirname(ProjectFileStore.projectFilePath)
        projectSimulationDirectory = os.path.join(os.path.sep, projectDirectory, "Simulation")
        targetSimulationDirectory = os.path.join(os.path.sep, targetDirectory, "Simulation")
        shutil.copytree(projectSimulationDirectory, targetSimulationDirectory)

        shutil.copy(ProjectFileStore.projectFilePath, targetDirectory)

        # Set the variables to write in Slurm job script
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
        for index, parameterRow in self.parameter_data.iterrows():
            simulationDirectory = os.path.join(os.path.sep, outputDirectory, parameterRow[0])
            #os.makedirs(simulationDirectory)
            self.copyCC3DProject(simulationDirectory)


            parameterFilePath = os.path.join(os.path.sep, simulationDirectory, "Simulation", parameterScriptName)
            os.remove(parameterFilePath)
            self.createParameterFile(parameterFilePath, parameterRow)

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