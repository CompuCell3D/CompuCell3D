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
}

