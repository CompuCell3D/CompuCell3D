#include "EnergyFunctionCalculator.h"
#include <CompuCell3D/CC3DExceptions.h>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "EnergyFunction.h"
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;
using namespace std;

EnergyFunctionCalculator::EnergyFunctionCalculator() {
    potts = 0;
}

EnergyFunctionCalculator::~EnergyFunctionCalculator() {

}

void EnergyFunctionCalculator::registerEnergyFunction(EnergyFunction *_function) {

    checkEnergyFunction(_function);

    //_function->registerPotts3D(potts);

    ostringstream automaticNameStream;
    automaticNameStream << "EnergyFuction_" << energyFunctions.size() - 1;
    string functionName;

    functionName = automaticNameStream.str();
    nameToEnergyFunctionMap.insert(make_pair(functionName, _function));

    energyFunctions.push_back(_function);
    energyFunctionsNameVec.push_back(functionName);

}

void EnergyFunctionCalculator::registerEnergyFunctionWithName(EnergyFunction *_function, std::string _functionName) {

    checkEnergyFunction(_function);

    ostringstream automaticNameStream;
    automaticNameStream << "EnergyFuction_" << energyFunctions.size() - 1;
    string functionName;
    if (_functionName.empty()) {
        functionName = automaticNameStream.str();
    } else {
        functionName = _functionName;
    }
    nameToEnergyFunctionMap.insert(make_pair(functionName, _function));

    energyFunctions.push_back(_function);
    energyFunctionsNameVec.push_back(functionName);


}


void EnergyFunctionCalculator::unregisterEnergyFunction(std::string _functionName) {

    map<string, EnergyFunction *>::iterator mitr;
    mitr = nameToEnergyFunctionMap.find(_functionName);

  if(mitr==nameToEnergyFunctionMap.end()){
    CC3D_Log(LOG_DEBUG) << "Sorry, Could not find "<<_functionName<<" energy Function";
        return; //plugin name not found
    }
    energyFunctions.erase(remove(energyFunctions.begin(), energyFunctions.end(), mitr->second), energyFunctions.end());
    energyFunctionsNameVec.erase(remove(energyFunctionsNameVec.begin(), energyFunctionsNameVec.end(), _functionName),
                                 energyFunctionsNameVec.end());

}

double EnergyFunctionCalculator::changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell,
                                              const unsigned int _flipAttempt) {


    double change = 0;
    for (unsigned int i = 0; i < energyFunctions.size(); i++) {
        change += energyFunctions[i]->changeEnergy(pt, newCell, oldCell);
    }
    return change;

}

void EnergyFunctionCalculator::checkEnergyFunction(EnergyFunction *_function) {
    if (!_function) throw CC3DException("registerEnergyFunction() function cannot be NULL!");
    if (!potts) throw CC3DException("Potts3D Pointer cannot be NULL!");
}
