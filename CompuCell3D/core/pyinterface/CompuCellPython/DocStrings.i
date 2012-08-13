//Simulator.h
%define Simulator_class
"Simulation class used for getting information about the simulation"
%enddef
%define getNumSteps_func
"Returns the number of Monte Carlo steps the simulation will perform"
%enddef
%define getStep_func
"Returns the current Monte Carlo step"
%enddef
%define isStepping_func
"Returns a bool value on the condition of whether the simulation is currently running"
%enddef
%define getPotts_func
"Returns a pointer to the Potts class"
%enddef

//Potts.h
%define Potts3D_class
"Potts3D class used for getting information about the lattice and cells"
%enddef
%define getNumberOfAttempts_func
"Returns the number of attempted flips in a Monte Carlo step"
%enddef
%define getNumberOfAcceptedSpinFlips_func
"Returns the number of accepted flips in a Monte Carlo step"
%enddef
%define getNumberOfAttemptedEnergyCalculations_func
"I don't know"
%enddef
%define getDepth_func
"Return the Depth"
%enddef
%define setDepth_func
"Function to set the Depth (neighbor distance)"
%enddef
%define setDebugOutputFrequency_func
"Function to set output frequency of debug statements"
%enddef
%define getCellInventory_func
"Returns a list of all the cells in the simulation"
%enddef

//PottsParseData.h
%define PottsParseData_Class
"Class used for defining lattice and simulation.  Gives access to public member variables: Frequency, acceptanceFunctionName, 
algorithmName, anneal, boundary_x, boundary_y, boundary_z, debugOutputFrequency, depth, depthFlag, dim, energyFcnParseDataPtr, 
flip2DimRatio, frequency, getEnergyFunctionCalculatorStatisticsParseData, kBoltzman, latticeType, moduleName, neighborOrder, 
numSteps, offset, seed, shapeAlgorithm, shapeFlag,   shapeIndex, shapeInputfile, shapeReg, shapeSize, temperature"
%enddef

//VolumeParseData.h
%define VolumeParseData_Class
"Class used for Defining Lambda and Target Volumes for Cells"
%enddef
%define LambdaVolume_func
"Function to set the LambdaVolume"
%enddef
%define TargetVolume_func
"Function to set the TargetVolume"
%enddef
