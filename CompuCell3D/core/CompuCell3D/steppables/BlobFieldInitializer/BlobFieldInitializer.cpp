
#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>

using namespace CompuCell3D;

using namespace std;


#include "BlobFieldInitializer.h"
#include <Logger/CC3DLogger.h>

std::string BlobFieldInitializer::steerableName() {
    return toString();
}

std::string BlobFieldInitializer::toString() {
    return "BlobInitializer";
}


BlobFieldInitializer::BlobFieldInitializer() :
        potts(0), sim(0) {}

void BlobFieldInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    sim = simulator;
    potts = simulator->getPotts();
    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellFieldG) throw CC3DException("initField() Cell field G cannot be null!");
    Dim3D dim = cellFieldG->getDim();


    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    if (_xmlData->getFirstElement("Radius")) {
        oldStyleInitData.radius = _xmlData->getFirstElement("Radius")->getUInt();
        CC3D_Log(LOG_DEBUG) << "Got FE This Radius: " << oldStyleInitData.radius;
        if (!(oldStyleInitData.radius > 0 && 2 * (oldStyleInitData.radius) < (dim.x - 2)))
            throw CC3DException(
                    "Radius has to be greater than 0 and 2*radius cannot be bigger than lattice dimension x");
    }

    if (_xmlData->getFirstElement("Width")) {
        oldStyleInitData.width = _xmlData->getFirstElement("Width")->getUInt();
        CC3D_Log(LOG_DEBUG) << "Got FE This Width: " << oldStyleInitData.width;
    }
    if (_xmlData->getFirstElement("Gap")) {
        oldStyleInitData.gap = _xmlData->getFirstElement("Gap")->getUInt();
        CC3D_Log(LOG_DEBUG) << "Got FE This Gap: " << oldStyleInitData.gap;
    }


    if (_xmlData->getFirstElement("CellSortInit")) {
        if (_xmlData->getFirstElement("CellSortInit")->getText() == "yes" ||
            _xmlData->getFirstElement("CellSortInit")->getText() == "Yes") {
            cellSortInit = true;
            CC3D_Log(LOG_DEBUG) << "SET CELLSORT INIT";
        }
    }


    CC3DXMLElement *elem = _xmlData->getFirstElement("Engulfment");
    if (elem) {
        engulfmentData.engulfment = true;
        engulfmentData.bottomType = elem->getAttribute("BottomType");
        engulfmentData.topType = elem->getAttribute("TopType");
        engulfmentData.engulfmentCutoff = elem->getAttributeAsUInt("EngulfmentCutoff");
        engulfmentData.engulfmentCoordinate = elem->getAttribute("EngulfmentCoordinate");
    }


    //clearing vector storing BlobFieldInitializerData (region definitions)
    blobInitializerData.clear();

    CC3DXMLElementList regionVec = _xmlData->getElements("Region");


    for (int i = 0; i < regionVec.size(); ++i) {
        BlobFieldInitializerData initData;
        if (!regionVec[i]->getFirstElement("Radius"))
            throw CC3DException(
                    "BlobInitializer requires Radius element inside Region section.See manual for details.");
        initData.radius = regionVec[i]->getFirstElement("Radius")->getUInt();
        if (regionVec[i]->getFirstElement("Gap")) {
            initData.gap = regionVec[i]->getFirstElement("Gap")->getUInt();
        }

        if (regionVec[i]->getFirstElement("Width")) {
            initData.width = regionVec[i]->getFirstElement("Width")->getUInt();
        }

        if (!regionVec[i]->getFirstElement("Types"))
            throw CC3DException("BlobInitializer requires Types element inside Region section.See manual for details.");
        initData.typeNamesString = regionVec[i]->getFirstElement("Types")->cdata;

        parseStringIntoList(initData.typeNamesString, initData.typeNames, ",");

        if (!regionVec[i]->getFirstElement("Center"))
            throw CC3DException(
                    "BlobInitializer requires Center element inside Region section.See manual for details.");

        initData.center.x = regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("x");
        initData.center.y = regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("y");
        initData.center.z = regionVec[i]->getFirstElement("Center")->getAttributeAsUInt("z");
		CC3D_Log(LOG_DEBUG) << "radius="<<initData.radius<<" gap="<<initData.gap<<" types="<<initData.typeNamesString;
		blobInitializerData.push_back(initData);
	}

	CC3D_Log(LOG_DEBUG) << "GOT HERE BEFORE EXIT";

}


double BlobFieldInitializer::distance(double ax, double ay, double az, double bx, double by, double bz) {

    return sqrt((double) (ax - bx) * (ax - bx) +
                (double) (ay - by) * (ay - by) +
                (double) (az - bz) * (az - bz));
}

void BlobFieldInitializer::layOutCells(const BlobFieldInitializerData &_initData) {

    int size = _initData.gap + _initData.width;
    int cellWidth = _initData.width;

    WatchableField3D < CellG * > *cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");

    Dim3D dim = cellField->getDim();

    if (!(_initData.radius > 0 && 2 * _initData.radius < (dim.x - 2)))
        throw CC3DException("Radius has to be greater than 0 and 2*radius cannot be bigger than lattice dimension x");


	Dim3D itDim = getBlobDimensions(dim, size);
	CC3D_Log(LOG_DEBUG) << "itDim="<<itDim;


    Point3D pt;
    Point3D cellPt;
    CellG *cell;

    for (int z = 0; z < itDim.z; z++)
        for (int y = 0; y < itDim.y; y++)
            for (int x = 0; x < itDim.x; x++) {
                pt.x = x * size;
                pt.y = y * size;
                pt.z = z * size;


                if (!(distance(pt.x, pt.y, pt.z, _initData.center.x, _initData.center.y, _initData.center.z) <
                      _initData.radius)) {
                    continue; //such cell will not be inside spherical region
                }

                if (BoundaryStrategy::getInstance()->isValid(pt)) {
                    cell = potts->createCellG(pt);
                    cell->type = initCellType(_initData);
                    potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                    //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                    // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                    //inventory unless you call steppers(VolumeTrackerPlugin) explicitly


                } else {
                    continue;
                }

                for (cellPt.z = pt.z; cellPt.z < pt.z + cellWidth &&
                                      cellPt.z < dim.z; cellPt.z++)
                    for (cellPt.y = pt.y; cellPt.y < pt.y + cellWidth &&
                                          cellPt.y < dim.y; cellPt.y++)
                        for (cellPt.x = pt.x; cellPt.x < pt.x + cellWidth &&
                                              cellPt.x < dim.x; cellPt.x++) {

                            if (BoundaryStrategy::getInstance()->isValid(pt))
                                cellField->set(cellPt, cell);

                        }
                potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                //inventory unless you call steppers(VolumeTrackerPlugin) explicitly

            }


}

unsigned char BlobFieldInitializer::initCellType(const BlobFieldInitializerData &_initData) {
    Automaton *automaton = potts->getAutomaton();
    if (_initData.typeNames.size() == 0) {//by default each newly created type will be 1
        return 1;
    }else { //user has specified more than one cell type - will pick randomly the type
        RandomNumberGenerator *randGen = sim->getRandomNumberGeneratorInstance();
        int index = randGen->getInteger(0, _initData.typeNames.size() - 1);
        return automaton->getTypeId(_initData.typeNames[index]);
    }

}

void BlobFieldInitializer::start() {
    if (sim->getRestartEnabled()) {
        return;  // we will not initialize cells if restart flag is on
    }
    // TODO: Chage this code so it write the 0 spins too.  This will make it
    //       possible to re-initialize a previously used field.

    /// I am changing here so that now I will work with cellFieldG - the field of CellG
    /// - this way CompuCell will have more functionality


    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellFieldG) throw CC3DException("initField() Cell field G cannot be null!");

    CC3D_Log(LOG_DEBUG) << "********************BLOB INIT***********************";
    Dim3D dim = cellFieldG->getDim();
    if (blobInitializerData.size() != 0) {
        for (int i = 0; i < blobInitializerData.size(); ++i) {
            CC3D_Log(LOG_DEBUG) << "GOT HERE";
            layOutCells(blobInitializerData[i]);
            //          exit(0);
        }
    } else {
        oldStyleInitData.center = Point3D(dim.x / 2, dim.y / 2, dim.z / 2);
        layOutCells(oldStyleInitData);

        if (cellSortInit) {
            initializeCellTypesCellSort();
        }

        if (engulfmentData.engulfment) {
            initializeEngulfment();
        }
    }


}

Dim3D BlobFieldInitializer::getBlobDimensions(const Dim3D &dim, int size) {
    Dim3D itDim;

    itDim.x = dim.x / size;
    if (dim.x % size) itDim.x += 1;
    itDim.y = dim.y / size;
    if (dim.y % size) itDim.y += 1;
    itDim.z = dim.z / size;
    if (dim.z % size) itDim.z += 1;

    blobDim = itDim;

    return itDim;

}


void BlobFieldInitializer::initializeEngulfment() {

    unsigned char topId, bottomId;
    CellTypePlugin *cellTypePluginPtr = (CellTypePlugin * )(Simulator::pluginManager.get("CellType"));
    if (!cellTypePluginPtr) throw CC3DException("CellType plugin not initialized!");

    EngulfmentData &enData = engulfmentData;

    topId = cellTypePluginPtr->getTypeId(enData.topType);
    bottomId = cellTypePluginPtr->getTypeId(enData.bottomType);
    CC3D_Log(LOG_DEBUG) << "topId="<<(int)topId<<" bottomId="<<(int)bottomId<<" enData.engulfmentCutoff="<<enData.engulfmentCutoff<<" enData.engulfmentCoordinate="<<enData.engulfmentCoordinate;


    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    Dim3D dim = cellFieldG->getDim();

    CellInventory *cellInventoryPtr = &potts->getCellInventory();
    ///will initialize cell type to be 1
    CellInventory::cellInventoryIterator cInvItr;

    CellG *cell;
    Point3D pt;

    ///loop over all the cells in the inventory
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);

        cell->type = 1;
    }

    for (int x = 0; x < dim.x; ++x) {
        for (int y = 0; y < dim.y; ++y) {
            for (int z = 0; z < dim.z; ++z) {
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cell = cellFieldG->get(pt);

                if (enData.engulfmentCoordinate == "x" || enData.engulfmentCoordinate == "X") {
                    if (cell && pt.x < enData.engulfmentCutoff) {
                        cell->type = bottomId;
                    } else if (cell && pt.x >= enData.engulfmentCutoff) {
                        cell->type = topId;
                    }

                }
                if (enData.engulfmentCoordinate == "y" || enData.engulfmentCoordinate == "Y") {
                    if (cell && pt.y < enData.engulfmentCutoff) {
                        cell->type = bottomId;
                    } else if (cell && pt.y >= enData.engulfmentCutoff) {
                        cell->type = topId;
                    }
                }
                if (enData.engulfmentCoordinate == "z" || enData.engulfmentCoordinate == "Z") {
                    if (cell && pt.z < enData.engulfmentCutoff) {
                        cell->type = bottomId;
                    } else if (cell && pt.z >= enData.engulfmentCutoff) {
                        cell->type = topId;
                    }
                }

            }

        }
    }


}


void BlobFieldInitializer::initializeCellTypesCellSort() {
    //Note that because cells are ordered by physical address in the memory you get additional
    //randomization of the cell types assignment. Assuming that randiom number generator is fixed i.e. it produces
    //same sequence of numbers every run, you still get random initial configuration and it comes from the fact that
    // in general ordering of cells in the inventory is not repetitive between runs

    RandomNumberGenerator *rand = sim->getRandomNumberGeneratorInstance();
    CellInventory *cellInventoryPtr = &potts->getCellInventory();

    ///will initialize cell type here depending on the position of the cells
    CellInventory::cellInventoryIterator cInvItr;

    CellG *cell;

    ///loop over all the cells in the inventory
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {

        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;

        if (rand->getRatio() < 0.5) { /// randomly assign types for cell sort
            cell->type = 1;
        } else {
            cell->type = 2;
        }

    }

}
