#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
#include <CompuCell3D/plugins/SimpleClock/SimpleClockPlugin.h>

using namespace CompuCell3D;
using namespace std;


#include "ChemotaxisDictyPlugin.h"


ChemotaxisDictyPlugin::ChemotaxisDictyPlugin() : field(0), potts(0), lambda(lambda), gotChemicalField(false),
                                                 xmlData(0) {
}

ChemotaxisDictyPlugin::~ChemotaxisDictyPlugin() {

}


void ChemotaxisDictyPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();

    potts->registerEnergyFunctionWithName(this, "ChemotaxisDicty");

    bool pluginAlreadyRegisteredFlagNeighbor;
    Plugin *plugin = Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlagNeighbor);
    if (!pluginAlreadyRegisteredFlagNeighbor)
        plugin->init(sim);


    bool pluginAlreadyRegisteredFlag;
    //this will load SurfaceTracker plugin if it is not already loaded
    SimpleClockPlugin *simpleClockPlugin = (SimpleClockPlugin *) Simulator::pluginManager.get("SimpleClock",
                                                                                              &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        simpleClockPlugin->init(sim);


    simpleClockAccessorPtr = simpleClockPlugin->getSimpleClockAccessorPtr();
    simulator->registerSteerableObject(this);

}

///will initialize chemotactic field here - need to deffer this after all steppables which contain field had been pre-initialzied
void ChemotaxisDictyPlugin::extraInit(Simulator *simulator) {

    update(xmlData, true);


}

void ChemotaxisDictyPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
    if (!newCell) {
        concentrationField->set(pt, 0.0);/// in medium we assume concentration 0
    }
}


void ChemotaxisDictyPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //if(potts->getDisplayUnitsFlag()){
    //	Unit energyUnit=potts->getEnergyUnit();




    //	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units");
    //	if (!unitsElem){ //add Units element
    //		unitsElem=_xmlData->attachElement("Units");
    //	}

    //	if(unitsElem->getFirstElement("LambdaUnit")){
    //		unitsElem->getFirstElement("LambdaUnit")->updateElementValue(energyUnit.toString());
    //	}else{
    //		CC3DXMLElement * energyElem = unitsElem->attachElement("LambdaUnit",energyUnit.toString());
    //	}
    //}

    nonChemotacticTypeVector.clear();

    lambda = _xmlData->getFirstElement("Lambda")->getDouble();

    if (_xmlData->findElement("NonChemotacticType")) {

        nonChemotacticTypeVector.push_back(_xmlData->getFirstElement("NonChemotacticType")->getByte());
    }
    chemicalFieldName = _xmlData->getFirstElement("ChemicalField")->getText();
    chemicalFieldSource = _xmlData->getFirstElement("ChemicalField")->getAttribute("Source");
    initializeField();
}

double ChemotaxisDictyPlugin::changeEnergy(const Point3D &pt,
                                           const CellG *newCell,
                                           const CellG *oldCell) {


    if (!gotChemicalField)
        return 0.0;

///cells move up the concentration gradient
    float concentration = field->get(pt);
    float neighborConcentration = field->get(potts->getFlipNeighbor());
    float energy = 0.0;
    unsigned char type;
    bool chemotaxisDone = false;

    /// new cell has to be different than medium i.e. only non-medium cells can chemotact
    ///e.g. in chemotaxis only non-medium cell can move a pixel that either belonged to other cell or to medium
    ///but situation where medium moves to a new pixel is not considered a chemotaxis
    if (newCell && simpleClockAccessorPtr->get(newCell->extraAttribPtr)->flag) {
        energy += (neighborConcentration - concentration) * lambda;
        chemotaxisDone = true;

    }

    if (!chemotaxisDone && oldCell && simpleClockAccessorPtr->get(oldCell->extraAttribPtr)->flag) {
        energy += (neighborConcentration - concentration) * lambda;
        chemotaxisDone = true;
    }

    return energy;
}

double ChemotaxisDictyPlugin::getConcentration(const Point3D &pt) {
    if (!field) throw CC3DException("No chemical field has been initialized!");
    return field->get(pt);
}

void ChemotaxisDictyPlugin::initializeField() {

    if (!gotChemicalField) {///this is only temporary solution will have to come up with something better
        ClassRegistry *classRegistry = sim->getClassRegistry();
        Steppable *steppable = classRegistry->getStepper(chemicalFieldSource);

        field = ((DiffusableVector<float> *) steppable)->getConcentrationField(chemicalFieldName);
        gotChemicalField = true;

        if (!field) throw CC3DException("No chemical field has been initialized!");
    }

}


std::string ChemotaxisDictyPlugin::toString() {
    return "ChemotaxisDicty";
}


std::string ChemotaxisDictyPlugin::steerableName() {
    return toString();
}

