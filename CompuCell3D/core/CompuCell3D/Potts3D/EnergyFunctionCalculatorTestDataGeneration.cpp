#include "EnergyFunctionCalculatorTestDataGeneration.h"
#include "EnergyFunction.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cmath>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <CompuCell3D/Simulator.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <sstream>
#include <CompuCell3D/PottsParseData.h>
#include <XMLUtils/CC3DXMLElement.h>
#include "PottsTestData.h"


using namespace CompuCell3D;
using namespace std;


EnergyFunctionCalculatorTestDataGeneration::EnergyFunctionCalculatorTestDataGeneration():EnergyFunctionCalculator(){

}

EnergyFunctionCalculatorTestDataGeneration::~EnergyFunctionCalculatorTestDataGeneration(){

}

double EnergyFunctionCalculatorTestDataGeneration::changeEnergy(Point3D &pt, const CellG *newCell,const CellG *oldCell,const unsigned int _flipAttempt){

    double change = 0;
    double energy = 0;
    for (unsigned int i = 0; i < energyFunctions.size(); i++) {
        
        energy = energyFunctions[i]->changeEnergy(pt, newCell, oldCell);
        change += energy;
        energyFunctionNameToValueMap[energyFunctionsNameVec[i]] = energy;
    }
    return change;    
}

std::string EnergyFunctionCalculatorTestDataGeneration::get_output_file_name() {
    return sim->getOutputDirectory() + "/" + "potts_data_output.csv";
}

std::string EnergyFunctionCalculatorTestDataGeneration::get_input_file_name() {
    return potts->get_simulation_input_dir() + "/" + "potts_data_output.csv";
}


void EnergyFunctionCalculatorTestDataGeneration::log_output(PottsTestData & potts_test_data) {

    potts_test_data.energyFunctionNameToValueMap = energyFunctionNameToValueMap;
    std::string file_name = get_output_file_name();

    if (!header_written) {        
        header_written = potts_test_data.write_header(file_name);
        ASSERT_OR_THROW(" Could not write header to " + file_name, header_written);        
    }

    bool write_ok = potts_test_data.serialize(file_name);
    ASSERT_OR_THROW(" Could not write data to " + file_name, write_ok);

}

