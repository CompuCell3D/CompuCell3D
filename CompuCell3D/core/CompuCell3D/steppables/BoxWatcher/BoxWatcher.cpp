
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;


#include "BoxWatcher.h"

BoxWatcher::BoxWatcher() : cellFieldG(0), sim(0), potts(0), xMargin(0), yMargin(0), zMargin(0) {}

BoxWatcher::~BoxWatcher() {
}

Point3D BoxWatcher::getMinCoordinates() { return minCoordinates; }

Point3D BoxWatcher::getMaxCoordinates() { return maxCoordinates; }

Point3D BoxWatcher::getMargins() { return Point3D(xMargin, yMargin, zMargin); }

void BoxWatcher::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    potts = simulator->getPotts();
    sim = simulator;
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();


    minCoordinates = Point3D(fieldDim.x, fieldDim.y, fieldDim.z);
    maxCoordinates = Point3D(0, 0, 0);

    simulator->registerSteerableObject(this);

    update(_xmlData, true);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxWatcher::extraInit(Simulator *simulator) {
    frozenTypeVector = potts->getFrozenTypeVector();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoxWatcher::start() {
    minCoordinates = Point3D(fieldDim.x, fieldDim.y, fieldDim.z);
    maxCoordinates = Point3D(0, 0, 0);


    adjustBox();


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxWatcher::step(const unsigned int currentStep) {

    minCoordinates = Point3D(fieldDim.x, fieldDim.y, fieldDim.z);
    maxCoordinates = Point3D(0, 0, 0);

    adjustBox();
}


void BoxWatcher::adjustBox() {
    Point3D pt;
    CellG *cell;
    for (int x = 0; x < fieldDim.x; ++x)
        for (int y = 0; y < fieldDim.y; ++y)
            for (int z = 0; z < fieldDim.z; ++z) {
                pt = Point3D(x, y, z);
                cell = cellFieldG->get(pt);
                if (!cell) continue;
                if (checkIfFrozen(cell->type)) continue;
                adjustCoordinates(pt);
            }

    int a;
    if (minCoordinates.x > maxCoordinates.x) {
        a = minCoordinates.x;
        minCoordinates.x = maxCoordinates.x;
        maxCoordinates.x = a;

    }
    if (minCoordinates.y > maxCoordinates.y) {
        a = minCoordinates.y;
        minCoordinates.y = maxCoordinates.y;
        maxCoordinates.y = a;

    }
    if (minCoordinates.z > maxCoordinates.z) {
        a = minCoordinates.z;
        minCoordinates.z = maxCoordinates.z;
        maxCoordinates.z = a;

    }


    minCoordinates.x = ((int) minCoordinates.x - (int) xMargin <= 0 ? 0 : minCoordinates.x - xMargin);
    minCoordinates.y = ((int) minCoordinates.y - (int) yMargin <= 0 ? 0 : minCoordinates.y - yMargin);
    minCoordinates.z = ((int) minCoordinates.z - (int) zMargin <= 0 ? 0 : minCoordinates.z - zMargin);

    //note fieldDim.x-1 is max x cocordinate for lattice pixel, similarly for y, and z
    maxCoordinates.x = (maxCoordinates.x + xMargin >= fieldDim.x - 1 ? fieldDim.x : maxCoordinates.x + xMargin + 1);
    maxCoordinates.y = (maxCoordinates.y + yMargin >= fieldDim.y - 1 ? fieldDim.y : maxCoordinates.y + yMargin + 1);
    maxCoordinates.z = (maxCoordinates.z + zMargin >= fieldDim.z - 1 ? fieldDim.z : maxCoordinates.z + zMargin + 1);

   potts->setMinCoordinates(minCoordinates);
   potts->setMaxCoordinates(maxCoordinates);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BoxWatcher::adjustCoordinates(Point3D _pt) {

    if (_pt.x > maxCoordinates.x)
        maxCoordinates.x = _pt.x;
    if (_pt.y > maxCoordinates.y)
        maxCoordinates.y = _pt.y;
    if (_pt.z > maxCoordinates.z)
        maxCoordinates.z = _pt.z;

    if (_pt.x < minCoordinates.x)
        minCoordinates.x = _pt.x;
    if (_pt.y < minCoordinates.y)
        minCoordinates.y = _pt.y;
    if (_pt.z < minCoordinates.z)
        minCoordinates.z = _pt.z;


}


bool BoxWatcher::checkIfFrozen(unsigned char _type) {

    for (unsigned int i = 0; i < frozenTypeVector.size(); ++i) {
        if (frozenTypeVector[i] == _type)
            return true;
    }
    return false;

}

void BoxWatcher::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    if (_xmlData->findElement("XMargin"))
        xMargin = _xmlData->getFirstElement("XMargin")->getUInt();
    if (_xmlData->findElement("YMargin"))
        yMargin = _xmlData->getFirstElement("YMargin")->getUInt();
    if (_xmlData->findElement("ZMargin"))
        zMargin = _xmlData->getFirstElement("ZMargin")->getUInt();

}

std::string BoxWatcher::toString() {
    return "BoxWatcher";
}


std::string BoxWatcher::steerableName() {
    return toString();
}


