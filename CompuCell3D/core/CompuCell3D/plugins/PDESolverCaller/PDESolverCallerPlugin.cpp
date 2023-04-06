#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "PDESolverCallerPlugin.h"

PDESolverCallerPlugin::PDESolverCallerPlugin() : sim(0), potts(0), xmlData(0) {}

PDESolverCallerPlugin::~PDESolverCallerPlugin() {}

void PDESolverCallerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;
    sim = simulator;
    potts = sim->getPotts();

    potts->registerFixedStepper(this);

    sim->registerSteerableObject(this);

}

void PDESolverCallerPlugin::extraInit(Simulator *simulator) {

    update(xmlData, true);

}

void PDESolverCallerPlugin::step() {

    unsigned int currentStep;
    unsigned int currentAttempt;
    unsigned int numberOfAttempts;


    currentStep = sim->getStep();
    currentAttempt = potts->getCurrentAttempt();
    numberOfAttempts = potts->getNumberOfAttempts();

    for (int i = 0; i < solverDataVec.size(); ++i) {
        if (!solverDataVec[i].extraTimesPerMC) //when user specifies 0 extra calls per MCS we don't execute the rest of the loop 
            continue;
        int reminder = (numberOfAttempts % (solverDataVec[i].extraTimesPerMC + 1));
        int ratio = (numberOfAttempts / (solverDataVec[i].extraTimesPerMC + 1));
        if (!((currentAttempt - reminder) % ratio) && currentAttempt > reminder) {
            solverPtrVec[i]->step(currentStep);

        }

    }

}

void PDESolverCallerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    solverPtrVec.clear();
    solverDataVec.clear();
    ClassRegistry *classRegistry = sim->getClassRegistry();
    Steppable *steppable;


    CC3DXMLElementList pdeSolversXMLList = _xmlData->getElements("CallPDE");
    for (unsigned int i = 0; i < pdeSolversXMLList.size(); ++i) {
        solverDataVec.push_back(SolverData(pdeSolversXMLList[i]->getAttribute("PDESolverName"), pdeSolversXMLList[i]->getAttributeAsUInt("ExtraTimesPerMC")));
        SolverData & sd = solverDataVec[solverDataVec.size() - 1];

        steppable = classRegistry->getStepper(sd.solverName);
        solverPtrVec.push_back(steppable);

    }

}

std::string PDESolverCallerPlugin::toString() {

    return "PDESolverCaller";

}

std::string PDESolverCallerPlugin::steerableName() {

    return toString();

}
