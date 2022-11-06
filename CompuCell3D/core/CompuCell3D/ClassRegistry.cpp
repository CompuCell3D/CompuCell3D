#include "Simulator.h"
#include "CC3DExceptions.h"
#include "ClassRegistry.h"
#include <string>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;

ClassRegistry::ClassRegistry(Simulator *simulator) : simulator(simulator) {
}


Steppable *ClassRegistry::getStepper(string id) {
    Steppable *stepper = activeSteppersMap[id];
    if (!stepper) throw CC3DException(string("Stepper '") + id + "' not found!");
    return stepper;
}

void ClassRegistry::extraInit(Simulator *simulator) {
    ActiveSteppers_t::iterator it;
    for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
        (*it)->extraInit(simulator);
}


void ClassRegistry::start() {
    ActiveSteppers_t::iterator it;
    for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
        (*it)->start();
}

void ClassRegistry::step(const unsigned int currentStep) {
    ActiveSteppers_t::iterator it;
    for (it = activeSteppers.begin(); it != activeSteppers.end(); it++) {
        if ((*it)->frequency && (currentStep % (*it)->frequency) == 0)
            (*it)->step(currentStep);
    }
}

void ClassRegistry::finish() {
    ActiveSteppers_t::iterator it;
    for (it = activeSteppers.begin(); it != activeSteppers.end(); it++)
        (*it)->finish();
}


void ClassRegistry::addStepper(std::string _type, Steppable *_steppable) {
    activeSteppers.push_back(_steppable);
    activeSteppersMap[_type] = _steppable;

}

void ClassRegistry::initModules(Simulator *_sim) {

    std::vector < CC3DXMLElement * > steppableCC3DXMLElementVectorRef = _sim->ps.steppableCC3DXMLElementVector;


    PluginManager<Steppable> &steppableManagerRef = Simulator::steppableManager;

    CC3D_Log(LOG_DEBUG) << " INSIDE INIT MODULES:" << endl;

    for (int i = 0; i < steppableCC3DXMLElementVectorRef.size(); ++i) {

        string type = steppableCC3DXMLElementVectorRef[i]->getAttribute("Type");

      Steppable *steppable = steppableManagerRef.get(type);
      CC3D_Log(LOG_DEBUG) << "CLASS REGISTRY INITIALIZING "<<type;
steppable->init(_sim, steppableCC3DXMLElementVectorRef[i]);
        addStepper(type, steppable);

    }

    for (ActiveSteppers_t::iterator litr = activeSteppers.begin(); litr != activeSteppers.end(); ++litr) {
        CC3D_Log(LOG_DEBUG) << "HAVE THIS STEPPER : "<<(*litr)->getParseData()->moduleName;
    }

}

