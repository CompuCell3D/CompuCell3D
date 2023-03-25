//Author: Margriet Palm CWI, Netherlands

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "RandomBlobInitializer.h"

using namespace std;

RandomBlobInitializer::RandomBlobInitializer() :
        mit(0),
        potts(0),
        simulator(0),
        rand(0),
        cellField(0),
        pixelTrackerAccessorPtr(0),
        builder(0),
        cellInventoryPtr(0) {

    ndiv, growsteps = 0;
    borderTypeID = -1;
    showStats = false;


}

RandomBlobInitializer::~RandomBlobInitializer() {
    delete builder;
    if (rand) {

        delete rand;
        rand = nullptr;
    }

}

void RandomBlobInitializer::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {
    CC3D_Log(LOG_DEBUG) << "START randomblob";
    simulator = _simulator;
    potts = _simulator->getPotts();
    cellField = (WatchableField3D<CellG *> *)
            potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field G cannot be null!");
    dim = cellField->getDim();
    cellInventoryPtr = &potts->getCellInventory();
    builder = new FieldBuilder(_simulator);

    auto randomSeed = simulator->getRNGSeed();
    rand = simulator->generateRandomNumberGenerator(randomSeed);


    update(_xmlData, true);


}

void RandomBlobInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    setParameters(simulator, _xmlData);
}

void RandomBlobInitializer::setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData) {

    builder->setRandomGenerator(rand);
    // set builder boxes
    Dim3D boxMin = Dim3D(0, 0, 0);
    Dim3D boxMax = cellField->getDim();
    if (_xmlData->getFirstElement("offset")) {
        int offsetX = _xmlData->getFirstElement("offset")->getAttributeAsUInt("x");
        int offsetY = _xmlData->getFirstElement("offset")->getAttributeAsUInt("y");
        int offsetZ = _xmlData->getFirstElement("offset")->getAttributeAsUInt("z");
        boxMin = Dim3D(offsetX, offsetY, offsetZ);
        boxMax = Dim3D(dim.x - offsetX, dim.y - offsetY, dim.z - offsetZ);
    }
    builder->setBoxes(boxMin, boxMax);
    int order = 1;
    if (_xmlData->getFirstElement("order"))
        order = _xmlData->getFirstElement("order")->getInt();
    if (order == 2) { builder->setNeighborListSO(); }
    else { builder->setNeighborListFO(); }
    // read types and set bias
    vector<string> typeNames;
    vector<string> biasVec;
    // read number of divisions
    if (_xmlData->getFirstElement("ndiv"))
        ndiv = _xmlData->getFirstElement("ndiv")->getInt();
    if (_xmlData->getFirstElement("types")) {
        string typeNamesString = _xmlData->getFirstElement("types")->getText();
        parseStringIntoList(typeNamesString, typeNames, ",");
    }
    bool biasSet = false;
    if (_xmlData->getFirstElement("bias")) {
        string biasString = _xmlData->getFirstElement("bias")->getText();
        parseStringIntoList(biasString, biasVec, ",");
        if (biasVec.size() == typeNames.size()) {
            builder->setTypeVec(pow((double) 2, (int) ndiv), typeNames, biasVec);
            biasSet = true;
        }
    }
    if (!biasSet)
        builder->setTypeVec(pow((double) 2, (int) ndiv), typeNames);
    // read number of growsteps
    if (_xmlData->getFirstElement("growsteps"))
        growsteps = _xmlData->getFirstElement("growsteps")->getInt();

    // get initial blobsize (before eden growth)
    blobsize = Dim3D(0, 0, 0);
    if (_xmlData->getFirstElement("initBlobSize")) {
        blobsize.x = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("x");
        blobsize.y = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("y");
        blobsize.z = _xmlData->getFirstElement("initBlobSize")->getAttributeAsUInt("z");
    }
    blobpos = Dim3D(dim.x / 2, dim.y / 2, dim.z / 2);
    if (_xmlData->getFirstElement("blobPos")) {
        blobpos.x = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("x");
        blobpos.y = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("y");
        blobpos.z = _xmlData->getFirstElement("blobPos")->getAttributeAsUInt("z");
    }
    // get   type
    Automaton *automaton = potts->getAutomaton();
    if (_xmlData->getFirstElement("borderType")) {
        borderTypeID = automaton->getTypeId(_xmlData->getFirstElement("borderType")->getText());
    }
    // check showstats
    if (_xmlData->getFirstElement("showStats"))
        showStats = true;
}

void RandomBlobInitializer::extraInit(Simulator *simulator) {
    bool pluginAlreadyRegisteredFlag;
    mit = (MitosisSteppable *) (Simulator::steppableManager.get("Mitosis", &pluginAlreadyRegisteredFlag));
    if (!pluginAlreadyRegisteredFlag) {
        mit->init(simulator);
    }
    if (!mit) throw CC3DException("MitosisSteppable not initialized!");
}

void RandomBlobInitializer::start() {
    Dim3D pos;
    if ((blobsize.x * blobsize.y * blobsize.z) == 1)
        pos = blobpos;
    else {
        pos.x = (blobpos.x > blobsize.x / 2) ? blobpos.x - blobsize.x / 2 : 0;
        pos.y = (blobpos.y > blobsize.y / 2) ? blobpos.y - blobsize.y / 2 : 0;
        pos.z = (blobpos.z > blobsize.z / 2) ? blobpos.z - blobsize.z / 2 : 0;
    }
    builder->addCell(pos, blobsize);

    builder->growCells(growsteps);

    for (int i = 0; i < ndiv; i++)
        divide();
    if (borderTypeID >= 0)
        builder->addBorderCell(borderTypeID);
    if (showStats) { builder->showCellStats(borderTypeID); }
}

void RandomBlobInitializer::divide() {
    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;
    CellG *child;
    PixelTracker *pixelTracker;
    set<PixelTrackerData>::iterator pixelItr;
    Point3D pt;
    vector<CellG *> cells;
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        if (cellInventoryPtr->getCell(cInvItr)->volume > 2)
            cells.push_back(cellInventoryPtr->getCell(cInvItr));
    }
    if ((int) cells.size() > 0) {
        vector<CellG *>::iterator it;
        for (it = cells.begin(); it < cells.end(); it++) {
            mit->doDirectionalMitosisAlongMinorAxis(*it);
            if (mit->childCell)
                builder->setType(mit->childCell);
        }
    } else { CC3D_Log(LOG_DEBUG) << "cells are too small, not dividing"; }
}

std::string RandomBlobInitializer::toString() {
    return "RandomBlobInitializer";
}


std::string RandomBlobInitializer::steerableName() {
    return toString();
}




