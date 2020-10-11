#include "EnergyFunctionCalculatorTestDataGeneration.h"
#include "EnergyFunction.h"
#include <iterator>
#include <iostream>
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
//void EnergyFunctionCalculatorTestDataGeneration::registerEnergyFunction(EnergyFunction *_function) {
//    EnergyFunctionCalculator::registerEnergyFunction(_function);
//}
//
//void EnergyFunctionCalculatorTestDataGeneration::registerEnergyFunctionWithName(EnergyFunction *_function, std::string _functionName) {
//    EnergyFunctionCalculator::registerEnergyFunctionWithName(_function, _functionName);
//}

//void EnergyFunctionCalculatorTestDataGeneration::registerEnergyFunction(EnergyFunction *_function) {
//
//    ASSERT_OR_THROW("registerEnergyFunction() function cannot be NULL!",
//        _function);
//
//    ASSERT_OR_THROW("Potts3D Pointer  cannot be NULL!",
//        potts);
//
//    //_function->registerPotts3D(potts);
//
//    ostringstream automaticNameStream;
//    automaticNameStream << "EnergyFuction_" << energyFunctions.size() - 1;
//    string functionName;
//
//    functionName = automaticNameStream.str();
//    nameToEnergyFuctionMap.insert(make_pair(functionName, _function));
//
//    energyFunctions.push_back(_function);
//    energyFunctionsNameVec.push_back(functionName);
//
//}
//
//void EnergyFunctionCalculatorTestDataGeneration::registerEnergyFunctionWithName(EnergyFunction *_function, std::string _functionName) {
//
//    cerr << "potts=" << potts << endl;
//    ASSERT_OR_THROW("registerEnergyFunction() function cannot be NULL!",
//        _function);
//
//    ASSERT_OR_THROW("Potts3D Pointer  cannot be NULL!",
//        potts);
//
//    cerr << "registering " << _functionName << endl;
//    //_function->registerPotts3D(potts);
//    ostringstream automaticNameStream;
//    automaticNameStream << "EnergyFuction_" << energyFunctions.size() - 1;
//    string functionName;
//    if (_functionName.empty()) {
//        functionName = automaticNameStream.str();
//    }
//    else {
//        functionName = _functionName;
//    }
//    nameToEnergyFuctionMap.insert(make_pair(functionName, _function));
//
//    energyFunctions.push_back(_function);
//    energyFunctionsNameVec.push_back(functionName);
//
//
//}


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

void EnergyFunctionCalculatorTestDataGeneration::log_output(PottsTestData & potts_test_data) {
    cerr << "logging output" << endl;
    cerr << " changePixel=" << potts_test_data.changePixel ;
    cerr << " changePixelNeighbor=" << potts_test_data.changePixelNeighbor;
    cerr << " motility=" << potts_test_data.motility;
    cerr << " pixelCopyAccepted=" << potts_test_data.pixelCopyAccepted << endl;
    cerr << "energyFuctionNametoValueMap.size()=" << energyFuctionNametoValueMap.size() << endl;
    cerr << "energyFunctions=" << energyFunctions.size() << endl;    

    for (auto mitr = energyFuctionNametoValueMap.begin(); mitr != energyFuctionNametoValueMap.end(); ++mitr) {
        cerr << "energy function name = " << mitr->first << " value=" << mitr->second << endl;
    }
}