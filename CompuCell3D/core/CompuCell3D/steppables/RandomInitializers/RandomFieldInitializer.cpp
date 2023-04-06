
#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>

using namespace CompuCell3D;

#include "RandomFieldInitializer.h"

using namespace std;

RandomFieldInitializer::RandomFieldInitializer() :
        potts(0),
        simulator(0),
        rand(0),
        cellField(0),
        builder(0) {

    ncells, growsteps = 0;
    borderTypeID = -1;
    showStats = false;
}

RandomFieldInitializer::~RandomFieldInitializer() {
    delete builder;
    if (rand) {

        delete rand;
        rand = nullptr;
    }
}

void RandomFieldInitializer::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
    simulator = _simulator;
    potts = _simulator->getPotts();
    cellField = (WatchableField3D<CellG *> *)
            potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field G cannot be null!");
    dim = cellField->getDim();
    builder = new FieldBuilder(_simulator);
    // setParameters(_simulator,_xmlData);
    auto randomSeed = simulator->getRNGSeed();
    rand = simulator->generateRandomNumberGenerator(randomSeed);

    update(_xmlData, true);
}


void RandomFieldInitializer::extraInit(Simulator *simulator) {}

void RandomFieldInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    setParameters(simulator, _xmlData);
}

void RandomFieldInitializer::setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData) {
    // initiate random generator
    rand = _simulator->generateRandomNumberGenerator();
    builder->setRandomGenerator(rand);
    // set builder boxes
    Dim3D boxMin = Dim3D(0, 0, 0);
    Dim3D boxMax = cellField->getDim();
    if (_xmlData->getFirstElement("offset")) {
        int offsetX = _xmlData->getFirstElement("offset")->getAttributeAsUInt("x");
        int offsetY = _xmlData->getFirstElement("offset")->getAttributeAsUInt("y");
        int offsetZ = _xmlData->getFirstElement("offset")->getAttributeAsUInt("z");
        boxMin = Dim3D(offsetX, offsetY, offsetZ);
        boxMax.x = dim.x - offsetX;
        boxMax.y = dim.y - offsetY;
        boxMax.z = dim.z - offsetZ;
    }
    builder->setBoxes(boxMin, boxMax);
    int order = 1;
    if (_xmlData->getFirstElement("order"))
        order = _xmlData->getFirstElement("order")->getInt();
    CC3D_Log(LOG_DEBUG) << "order = " << order << endl;
    if (order == 2) { builder->setNeighborListSO(); }
    else { builder->setNeighborListFO(); }
    // read types and set bias
    vector<string> typeNames;
    vector<string> biasVec;
    if (_xmlData->getFirstElement("types")) {
        string typeNamesString = _xmlData->getFirstElement("types")->getText();
        parseStringIntoList(typeNamesString, typeNames, ",");
    }
    // read number of growsteps
    if (_xmlData->getFirstElement("growsteps"))
        growsteps = _xmlData->getFirstElement("growsteps")->getInt();
    // read number of cells
    if (_xmlData->getFirstElement("ncells"))
        ncells = _xmlData->getFirstElement("ncells")->getInt();
    bool biasSet = false;
    if (_xmlData->getFirstElement("bias")) {
        string biasString = _xmlData->getFirstElement("bias")->getText();
        parseStringIntoList(biasString, biasVec, ",");
        if (biasVec.size() == typeNames.size()) {
            builder->setTypeVec(ncells, typeNames, biasVec);
            biasSet = true;
        }
    }
    if (!biasSet)
        builder->setTypeVec(ncells, typeNames);
    if (ncells > (boxMax.x * boxMax.y * boxMax.z)) {
        ncells = boxMax.x * boxMax.y * boxMax.z;
        growsteps = 1;
        CC3D_Log(LOG_DEBUG) << "#########################";
        CC3D_Log(LOG_DEBUG) << "Too much cells!" << endl << "ncells is set to " << ncells;
        CC3D_Log(LOG_DEBUG) << "growsteps is set to 0";
        CC3D_Log(LOG_DEBUG) << "#########################";
    }
    // get border type
    Automaton *automaton = potts->getAutomaton();
    if (_xmlData->getFirstElement("borderType")) {
        borderTypeID = automaton->getTypeId(_xmlData->getFirstElement("borderType")->getText());
    }
    // check showstats
    if (_xmlData->getFirstElement("showStats"))
        showStats = true;
}

void RandomFieldInitializer::start() {
    int i;
    for (i = 0; i < ncells; i++) {
        builder->addCell();
    }
    builder->growCells(growsteps);
    if (borderTypeID >= 0) {
        builder->addBorderCell(borderTypeID);
    }
    if (showStats) { builder->showCellStats(borderTypeID); }
}

std::string RandomFieldInitializer::toString() {
    return "RandomFieldInitializer";
}


std::string RandomFieldInitializer::steerableName() {
    return toString();
}
