#include "Integrator.h"
#include "SimulatorCM.h"
#include <iostream>

using namespace CenterModel;
using namespace std;

       
void Integrator::init(SimulatorCM *_simulator){
    if (!_simulator)
        return;
    
    simulator=_simulator;
    //simulator->registerIntegrator(this);
    sbPtr=simulator->getSimulationBoxPtr();
    ciPtr=simulator->getCellInventoryPtr();
    boxMax=sbPtr->getDim();
    boxMin=Vector3(0.,0.,0.);
}

