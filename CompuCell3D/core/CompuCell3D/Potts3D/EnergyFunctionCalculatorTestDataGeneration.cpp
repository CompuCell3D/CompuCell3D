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
        energyFuctionNametoValueMap[energyFunctionsNameVec[i]] = energy;
        cerr<<"CHANGE FROM ACCEPTANCE FUNCTION"<<change<<" FCNNAME="<<energyFunctionsNameVec[i]<<endl;
    }
    return change;    
}

std::string EnergyFunctionCalculatorTestDataGeneration::get_output_file_name() {
    return sim->getOutputDirectory() + "/" + "potts_data_output.csv";
}

void EnergyFunctionCalculatorTestDataGeneration::write_header() {
    ofstream out(get_output_file_name(), std::ofstream::out);
    if (out) {
        out << "change_pixel_x,";
        out << "change_pixel_y,";
        out << "change_pixel_z,";
        out << "neighbor_change_pixel_x,";
        out << "neighbor_change_pixel_y,";
        out << "neighbor_change_pixel_z,";
        out << "motility,";
        out << "pixel_copy_accepted,";
        out << "acceptance_function_probability";

    }

    header_written = true;
}

void EnergyFunctionCalculatorTestDataGeneration::log_output(PottsTestData & potts_test_data) {
    if (!header_written) {
        write_header();

    }
    ofstream out(get_output_file_name(), std::ofstream::app);
    if (out) {
        out << potts_test_data.changePixel.x<<",";
        out << potts_test_data.changePixel.y << ",";
        out << potts_test_data.changePixel.z << ",";
        out << potts_test_data.changePixelNeighbor.x << ",";
        out << potts_test_data.changePixelNeighbor.y << ",";
        out << potts_test_data.changePixelNeighbor.z << ",";
        out << potts_test_data.motility<<",";
        out << potts_test_data.pixelCopyAccepted << ",";
        out << potts_test_data.pixelCopyAccepted << ",";
        out << potts_test_data.acceptanceFunctionProbability<<endl;

    }
    //cerr << "logging output" << endl;
    //cerr << " changePixel=" << potts_test_data.changePixel ;
    //cerr << " changePixelNeighbor=" << potts_test_data.changePixelNeighbor;
    //cerr << " motility=" << potts_test_data.motility;
    //cerr << " pixelCopyAccepted=" << potts_test_data.pixelCopyAccepted << endl;
    //cerr << "energyFuctionNametoValueMap.size()=" << energyFuctionNametoValueMap.size() << endl;
    //cerr << "energyFunctions=" << energyFunctions.size() << endl;    

    //for (auto mitr = energyFuctionNametoValueMap.begin(); mitr != energyFuctionNametoValueMap.end(); ++mitr) {
    //    cerr << "energy function name = " << mitr->first << " value=" << mitr->second << endl;
    //}
}