

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

using namespace std;


#include "DictyFieldInitializer.h"
#include <Logger/CC3DLogger.h>

DictyFieldInitializer::DictyFieldInitializer() :
        potts(0), gotAmoebaeFieldBorder(false), presporeRatio(0.5), gap(1), width(2), amoebaeFieldBorder(10) {}


void DictyFieldInitializer::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    if (_xmlData->findElement("Gap"))
        gap = _xmlData->getFirstElement("Gap")->getUInt();

    if (_xmlData->findElement("Width"))
        width = _xmlData->getFirstElement("Width")->getUInt();

    if (_xmlData->findElement("AmoebaeFieldBorder"))
        amoebaeFieldBorder = _xmlData->getFirstElement("AmoebaeFieldBorder")->getUInt();

    if (_xmlData->findElement("ZonePoint")) {
        zonePoint.x = _xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("x");
        zonePoint.y = _xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("y");
        zonePoint.z = _xmlData->getFirstElement("ZonePoint")->getAttributeAsUInt("z");
        zoneWidth = _xmlData->getFirstElement("ZonePoint")->getUInt();
    }

    if (_xmlData->findElement("PresporeRatio")) {
        presporeRatio = _xmlData->getFirstElement("PresporeRatio")->getDouble();
        if (!(0 <= presporeRatio && presporeRatio <= 1.0)) throw CC3DException("Ratio must belong to [0,1]!");
    }


}

void DictyFieldInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    update(_xmlData, true);

    sim = simulator;
    potts = simulator->getPotts();
    automaton = potts->getAutomaton();
    cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!cellField) throw CC3DException("initField() Cell field cannot be null!");
    if (!Simulator::pluginManager.get("CenterOfMass")) throw CC3DException("Could not find Center of Mass plugin");

    dim = cellField->getDim();

    if (!gotAmoebaeFieldBorder) {
        amoebaeFieldBorder = dim.x;
    }


}

void DictyFieldInitializer::start() {

    // TODO: Chage this code so it write the 0 spins too.  This will make it
    //       possible to re-initialize a previously used field.

    int size = gap + width;



    //  CenterOfMassPlugin * comPlugin=(CenterOfMassPlugin*)(Simulator::pluginManager.get("CenterOfMass"));
    //  Cell *c;

    //  comPlugin->getCenterOfMass(c);

    Dim3D itDim;

    itDim.x = dim.x / size;
    if (dim.x % size) itDim.x += 1;
    itDim.y = dim.y / size;
    if (dim.y % size) itDim.y += 1;
    itDim.z = dim.z / size;
    if (dim.z % size) itDim.z += 1;

    Point3D pt;
    Point3D cellPt;
    CellG *cell;

    ///this is only temporary initializer. Later I will rewrite it so that the walls and ground are thiner
    ///preparing ground layer
    pt.x = 0;
    pt.y = 0;
    pt.z = 0;

    cell = potts->createCellG(pt);
    groundCell = cell;
    for (cellPt.z = pt.z; cellPt.z <= pt.z + 1 * (width + gap) - 1 && cellPt.z < dim.z; cellPt.z++)
        for (cellPt.y = pt.y; cellPt.y < pt.y + dim.y && cellPt.y < dim.y; cellPt.y++)
            for (cellPt.x = pt.x; cellPt.x < pt.x + dim.y && cellPt.x < dim.x; cellPt.x++)
                cellField->set(cellPt, cell);


    ///walls
    cell = potts->createCellG(pt);
    wallCell = cell;
    for (cellPt.z = pt.z; cellPt.z < dim.z; cellPt.z++)
        for (cellPt.y = pt.y; cellPt.y < dim.y; cellPt.y++)
            for (cellPt.x = pt.x; cellPt.x < dim.x; cellPt.x++) {

                if (
                        (int) fabs(1.0 * cellPt.z - dim.z) % dim.z <= 1.0 ||
                        (int) fabs(1.0 * cellPt.y - dim.y) % dim.y <= 1.0 ||
                        (int) fabs(1.0 * cellPt.x - dim.x) % dim.x <= 1.0
                        ) {
                    cellField->set(cellPt, cell);
				}

                /*         if(cellPt.z<width)///additionally thick wall at the bottom to prevent the diffusion into too large area
                cellField->set(cellPt, cell);*/
            }



    ///laying down layer of aboebae - keeping the distance from the walls
    for (int z = 1; z < 2; z++)
        for (int y = 1; y < itDim.y - 1; y++)
            for (int x = 1; x < itDim.x - 1; x++) {

                pt.x = x * size;
                pt.y = y * size;
                pt.z = z * size;


                if (pt.x < amoebaeFieldBorder && pt.y < amoebaeFieldBorder) {
                    cell = potts->createCellG(pt);


                    for (cellPt.z = pt.z; cellPt.z < pt.z + width && cellPt.z < dim.z; cellPt.z++)
                        for (cellPt.y = pt.y; cellPt.y < pt.y + width && cellPt.y < dim.y; cellPt.y++)
                            for (cellPt.x = pt.x; cellPt.x < pt.x + width && cellPt.x < dim.x; cellPt.x++)
                                cellField->set(cellPt, cell);
                }
            }


    //Now will initialize types of cells
    initializeCellTypes();

    ///preparing water layer
    /*   pt.x = 0;
    pt.y = 0;
    pt.z = width+gap;

    cell = 0;

    bool initializedWater=false;
    //groundCell=cell;
    for (cellPt.z = pt.z; cellPt.z <= pt.z + width && cellPt.z < dim.z; cellPt.z++)
    for (cellPt.y = pt.y; cellPt.y < pt.y + dim.y && cellPt.y < dim.y; cellPt.y++)
    for (cellPt.x = pt.x; cellPt.x < pt.x + dim.y && cellPt.x < dim.x; cellPt.x++){


    if(!cellField->get(cellPt)){///check if medium
    if( ! initializedWater){ ///put water
    // CC3D_Log(LOG_DEBUG) << CREATING WATER CELL "<<cellPt;
    cell=potts->createCellG(pt);
    cell->type=automaton->getTypeId("Water");
    cellField->set(cellPt,cell);
    initializedWater=true;

    }else{///put water
    // CC3D_Log(LOG_DEBUG) << "SETTING WATER CELL "<<cellPt;
    cellField->set(cellPt,cell);
    }
    }

    }*/



}


void DictyFieldInitializer::initializeCellTypes() {

    RandomNumberGenerator *rand = sim->getRandomNumberGeneratorInstance();
    cellInventoryPtr = &potts->getCellInventory();

    ///will initialize cell type here depending on the position of the cells
    CellInventory::cellInventoryIterator cInvItr;
    ///loop over all the cells in the inventory
    Point3D com;
    CellG *cell;


    float x, y, z;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {

        cell = cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        if (cell == groundCell)
            cell->type = automaton->getTypeId("Ground");
        else if (cell == wallCell) {
            cell->type = automaton->getTypeId("Wall");
        } else {

			com.x=cell->xCM/ cell->volume ;
			com.y=cell->yCM/ cell->volume;
			com.z=cell->zCM/ cell->volume;
			CC3D_Log(LOG_DEBUG) << "belongToZone(com)="<<belongToZone(com)<<" com="<<com;
            if (belongToZone(com)) {
                cell->type = automaton->getTypeId("Autocycling");
                CC3D_Log(LOG_DEBUG) << "setting autocycling type="<<(int)cell->type;
            } else {
                if (rand->getRatio() < presporeRatio) {
                    cell->type = automaton->getTypeId("Prespore");
                } else {
                    cell->type = automaton->getTypeId("Prestalk");
                }

            }


        }


    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool DictyFieldInitializer::belongToZone(Point3D com) {

    if (
            com.x > zonePoint.x && com.x < (zonePoint.x + zoneWidth) &&
            com.y > zonePoint.y && com.y < (zonePoint.y + zoneWidth) &&
            com.z > zonePoint.z && com.z < (zonePoint.z + zoneWidth)
            )
        return true;
    else
        return false;

}


std::string DictyFieldInitializer::toString() {
    return "DictyInitializer";
}

std::string DictyFieldInitializer::steerableName() {
    return toString();
}

