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
        //cerr<<"CHANGE FROM ACCEPTANCE FUNCTION"<<change<<" FCNNAME="<<energyFunctionsNameVec[i]<<endl;
    }
    return change;    
}

void EnergyFunctionCalculatorTestDataGeneration::log_output(Point3D pt, Point3D nPt, bool accepted, float motility) {
    cerr << "logging output" << endl;
}