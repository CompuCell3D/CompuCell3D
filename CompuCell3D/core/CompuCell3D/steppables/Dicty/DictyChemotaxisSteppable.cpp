

#include <CompuCell3D/CC3D.h>

#include <CompuCell3D/plugins/SimpleClock/SimpleClockPlugin.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>


using namespace CompuCell3D;
using namespace std;


#include "DictyChemotaxisSteppable.h"
#include <Logger/CC3DLogger.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DictyChemotaxisSteppable::DictyChemotaxisSteppable() : potts(0), cellInventoryPtr(0) {

    clockReloadValue = 0;
    chemotactUntil = 0;
    chetmotaxisActivationThreshold = 0.0;
    chemotactingCellsCounter = 0;
    ignoreFirstSteps = 0;


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DictyChemotaxisSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    if (_xmlData->findElement("ClockReloadValue"))
        clockReloadValue = _xmlData->getFirstElement("ClockReloadValue")->getUInt();

    if (_xmlData->findElement("ChemotactUntil"))
        chemotactUntil = _xmlData->getFirstElement("ChemotactUntil")->getUInt();

    if (_xmlData->findElement("ChemotactUntil"))
        chemotactUntil = _xmlData->getFirstElement("ChemotactUntil")->getUInt();

    if (_xmlData->findElement("ChetmotaxisActivationThreshold"))
        chetmotaxisActivationThreshold = _xmlData->getFirstElement("ChetmotaxisActivationThreshold")->getDouble();

    if (_xmlData->findElement("IgnoreFirstSteps"))
        ignoreFirstSteps = _xmlData->getFirstElement("IgnoreFirstSteps")->getUInt();

    if (_xmlData->findElement("ChemicalField")) {
        chemicalFieldName = _xmlData->getFirstElement("ChemicalField")->getText();
        chemicalFieldSource = _xmlData->getFirstElement("ChemicalField")->getAttribute("Source");
    }

    if (chemotactUntil >= clockReloadValue)
        throw CC3DException("ChemotactUntil has to be smaller than Clock Reload Value!");

}

void DictyChemotaxisSteppable::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

    update(_xmlData, true);

    simulator = _simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    cellInventoryPtr = &potts->getCellInventory();

    bool pluginAlreadyRegisteredFlag;
    SimpleClockPlugin *simpleClockPlugin = (SimpleClockPlugin *) Simulator::pluginManager.get("SimpleClock",
                                                                                              &pluginAlreadyRegisteredFlag); //this will load PlasticityTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        simpleClockPlugin->init(simulator);

    simpleClockAccessorPtr = simpleClockPlugin->getSimpleClockAccessorPtr();

    simulator->registerSteerableObject(this);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void DictyChemotaxisSteppable::extraInit(Simulator *_simulator) {

    ClassRegistry *classRegistry = simulator->getClassRegistry();
    Steppable *steppable = classRegistry->getStepper(chemicalFieldSource);

    field = ((DiffusableVector<float> *) steppable)->getConcentrationField(chemicalFieldName);

    if (!field) throw CC3DException("No chemical field has been loaded!");
    CC3D_Log(LOG_DEBUG) << "GOT FIELD INTO CHEMOTAXIS STEPPABLE: "<<field;

    fieldDim = field->getDim();

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DictyChemotaxisSteppable::start() {
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DictyChemotaxisSteppable::step(const unsigned int currentStep){
	CC3D_Log(LOG_DEBUG) <<"ignoreFirstSteps="<<ignoreFirstSteps;

    if (currentStep < ignoreFirstSteps)
        return;

    Point3D pt;
    CellG *currentCellPtr;
    int *currentClockPtr;
    char *clockFlagPtr;
    float currentConcentration;

    for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
        for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
            for (pt.x = 0; pt.x < fieldDim.x; ++pt.x) {

                currentCellPtr = cellFieldG->get(pt);
                if(!currentCellPtr) continue;

                currentClockPtr = &(simpleClockAccessorPtr->get(
                        currentCellPtr->extraAttribPtr)->clock);///to avoid costly too many costly accesses
                clockFlagPtr = &(simpleClockAccessorPtr->get(currentCellPtr->extraAttribPtr)->flag);
                currentConcentration = field->get(pt);

                if (currentConcentration > chetmotaxisActivationThreshold && *currentClockPtr == 0) {
                    //simpleClockAccessorPtr->get(currentCellPtr->extraAttribPtr)->clock = clockReloadValue;
                    *currentClockPtr = clockReloadValue; ///will reload the clock if any pixel belonging to a cel has concentration > activeThreshold
                    *clockFlagPtr = 1;
                    // CC3D_Log(LOG_DEBUG) <<"\t\treloading clock and activating chemotaxis";
                    ++chemotactingCellsCounter;
                }

            }

    ///decrementing clock
    CellInventory::cellInventoryIterator cInvItr;
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        currentCellPtr = cellInventoryPtr->getCell(cInvItr);
        //currentCellPtr=*cInvItr;
        currentClockPtr = &(simpleClockAccessorPtr->get(currentCellPtr->extraAttribPtr)->clock);
        clockFlagPtr = &simpleClockAccessorPtr->get(currentCellPtr->extraAttribPtr)->flag;

        if (*currentClockPtr >= 1) {
            --(*currentClockPtr);///decrementring clock but only until zero - non negative clock values allowed here
        }

        if (*currentClockPtr < chemotactUntil && *clockFlagPtr) {
            *clockFlagPtr = 0;
            // CC3D_Log(LOG_DEBUG) <<"Deactivating chemotaxis";
            --chemotactingCellsCounter;
        }

			}
			// CC3D_Log(LOG_DEBUG) <<"\n\n \t\t THERE ARE "<<chemotactingCellsCounter<<" CELLS CHEMOTACTING\n";
}

std::string DictyChemotaxisSteppable::toString() {
    return "DictyChemotaxisSteppable";
}


std::string DictyChemotaxisSteppable::steerableName() {
    return toString();
}


