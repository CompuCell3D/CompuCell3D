
import os
import sys
import math

from PySteppables import *
from XMLUtils import dictionaryToMapStrStr as d2mss

import bionetAPI

class OscillatingContactEnergiesSteppable( SteppableBasePy ):
    def __init__( self, _simulator, _frequency = 1 ):
        SteppableBasePy.__init__( self, _simulator,_frequency )
        
        bionetAPI.initializeBionetworkManager( self.simulator )
    
    def modifyContactEnergies( self, BB_offset, GG_offset, steadyStateOffset,
        BB_scale = -30.0,  GG_scale = -10.0,
        MM_scale =  30.0,  MB_scale = -10.0,
        MG_scale =   0.0,  BG_scale =  10.0 ):
        
        contactXMLData = self.simulator.getCC3DModuleData( "Plugin", "Contact" )
        BB = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Blue", "Type2":"Blue"} ) )
        GG = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Green", "Type2":"Green"} ) )
        MM = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Medium", "Type2":"Medium"} ) )
        MB = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Medium", "Type2":"Blue"} ) )
        MG = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Medium", "Type2":"Green"} ) )
        BG = contactXMLData.getFirstElement( "Energy", d2mss( {"Type1":"Blue", "Type2":"Green"} ) )
        
        BB.updateElementValue( str( BB_scale + BB_offset  - steadyStateOffset ) )
        GG.updateElementValue( str( GG_scale + GG_offset - steadyStateOffset ) )
        MM.updateElementValue( str( MM_scale ) )
        MB.updateElementValue( str( MB_scale ) )
        MG.updateElementValue( str( MG_scale ) )
        BG.updateElementValue( str( BG_scale ) )
        
        self.simulator.updateCC3DModule(contactXMLData)
 
    
    def writeContactEnergies( self, mcs, fileName = "contactEnergies.txt", append = "w" ):
        contactXMLData = self.simulator.getCC3DModuleData( "Plugin", "Contact" )
        BB = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Blue", "Type2":"Blue"}) ).getText()
        GG = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Green", "Type2":"Green"}) ).getText()
        MM = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Medium", "Type2":"Medium"}) ).getText()
        MB = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Medium", "Type2":"Blue"}) ).getText()
        MG = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Medium", "Type2":"Green"}) ).getText()
        BG = contactXMLData.getFirstElement( "Energy", d2mss({"Type1":"Blue", "Type2":"Green"}) ).getText()
        
        outputFile = open( fileName, append )
        if( append == "w" ):
            outputFile.write( "\t%s\t%s\t%s\t%s\t%s\t%s\n" % (   "BB", "GG", "MM", "MB", "MG", "BG" ) )
        outputFile.write( "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ( mcs, BB,   GG,   MM,   MB,   MG,   BG  ) )
        outputFile.close()
        
    
    def start(self):
        
        # #################### Create SBML models ######################
        
        ## Load a Bionetwork SBML model named "SimpleExample"
        sbmlModelPath =  "Simulation/SimpleExample.sbml"
        bionetAPI.loadSBMLModel( "SimpleExample", sbmlModelPath, "SE", 0.5 )
        
        
        # ################ Add SBML models to Bionetwork Template Libraries ##################
        
        ## Add SBML model to non-cell bionetwork template libaries called "ExternalOscillatorA" and "ExternalOscillatorB"
        bionetAPI.addSBMLModelToTemplateLibrary( "SimpleExample", "ExternalOscillatorA" )
        bionetAPI.addSBMLModelToTemplateLibrary( "SimpleExample", "ExternalOscillatorB" )
        
        
        # ####### Set initial conditions for Bionetwork properties #########
        
        ## Set initial conditions for "ExternalOscillatorA" and "ExternalOscillatorB" bionetwork template libraries
        self.k1_init = 0.1; self.k2_init = 0.1; self.mcsPeriod = 500
        
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorA", "SE_k1", self.k1_init )
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorA", "SE_k2", self.k2_init )
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorA", "SE_X0", 10.0 )
        
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorB", "SE_k1", self.k1_init )
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorB", "SE_k2", self.k2_init )
        bionetAPI.setBionetworkInitialCondition( "ExternalOscillatorB", "SE_X0", 10.0 )
        
        
        # ######## Create Bionetworks (integrators) and initialize their states ##########
        bionetAPI.initializeBionetworks()
        
        
        # ######## Perform initialization tasks and calculations ##########
        
        k1 = bionetAPI.getBionetworkValue( "SE_k1", "ExternalOscillatorA" )
        k2 = bionetAPI.getBionetworkValue( "SE_k2", "ExternalOscillatorA" )
        X0 = bionetAPI.getBionetworkValue( "SE_X0", "ExternalOscillatorA" )
        print "k1=",k1," k2=",k2," X0=",X0 
        self.steadyState = k1 * X0 / k2
        
        
        # #### Write bionetwork state variable names of external oscillators to output file
        self.oscillatorAFileName = "Demos/BionetSolverExamples/OscillatingContactEnergies/oscillatorStateA.txt"
        bionetAPI.writeBionetworkStateVarNamesToFile( "ExternalOscillatorA", "SimpleExample", self.oscillatorAFileName, "w" )
        
        self.oscillatorBFileName = "Demos/BionetSolverExamples/OscillatingContactEnergies/oscillatorStateB.txt"
        bionetAPI.writeBionetworkStateVarNamesToFile( "ExternalOscillatorB", "SimpleExample", self.oscillatorBFileName, "w" )
        
        
        # ############## Write adhesion data to an output file ################
        self.contactEnergyOutputFile = "Demos/BionetSolverExamples/OscillatingContactEnergies/contactEnergies.txt"
        self.writeContactEnergies( 0, self.contactEnergyOutputFile, "w" )
        
    def step(self, mcs):
        
        # #### Write bionetwork state variable names of external oscillators to output file
        bionetAPI.writeBionetworkStateToFile( mcs, "ExternalOscillatorA", "SimpleExample", self.oscillatorAFileName, "a" )
        bionetAPI.writeBionetworkStateToFile( mcs, "ExternalOscillatorB", "SimpleExample", self.oscillatorBFileName, "a" )
        
        
        # ######### Update integrator(s) for all bionetworks ###########
        bionetAPI.timestepBionetworks()
        
        
        # ######## Modify k1 and k2 as a function of time (MCS) ##########
        k1 = self.k1_init * ( 1 + math.sin( 2*math.pi*mcs/self.mcsPeriod ) )
        k2 = self.k2_init * ( 1 - math.sin( 2*math.pi*mcs/self.mcsPeriod ) )
        bionetAPI.setBionetworkValue( "SE_k1", 0.2*k1, "ExternalOscillatorA" )
        bionetAPI.setBionetworkValue( "SE_k2", 0.2*k2, "ExternalOscillatorA" )
        
        k1 = self.k1_init * ( 1 - math.sin( 2*math.pi*mcs/self.mcsPeriod ) )
        k2 = self.k2_init * ( 1 + math.sin( 2*math.pi*mcs/self.mcsPeriod ) )
        bionetAPI.setBionetworkValue( "SE_k1", 0.2*k1, "ExternalOscillatorB" )
        bionetAPI.setBionetworkValue( "SE_k2", 0.2*k2, "ExternalOscillatorB" )
        
        
        # ####### Set "Blue-Blue" and "Green-Green" adhesion energies #########
        BB_offset = bionetAPI.getBionetworkValue( "SE_S1", "ExternalOscillatorA" )
        GG_offset = bionetAPI.getBionetworkValue( "SE_S1", "ExternalOscillatorB" )
        self.modifyContactEnergies( BB_offset, GG_offset, self.steadyState )
        
        
        # ############## Write adhesion data to an output file ################
        self.writeContactEnergies( mcs, self.contactEnergyOutputFile, "a" )
        
        
        
        
        
        
        
        








