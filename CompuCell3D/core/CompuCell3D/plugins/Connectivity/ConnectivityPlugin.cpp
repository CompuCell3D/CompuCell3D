#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "ConnectivityPlugin.h"


ConnectivityPlugin::ConnectivityPlugin() : penalty(0.0), potts(0), numberOfNeighbors(8) {

    offsetsIndex.assign(numberOfNeighbors, 0);//allocates memory for vector of offsetsIndexes
}

ConnectivityPlugin::~ConnectivityPlugin() {
}

void ConnectivityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    potts = simulator->getPotts();
    potts->registerEnergyFunction(this);
    simulator->registerSteerableObject(this);
    update(_xmlData, true);

    //here we initialize offsets for neighbors (up to second nearest) in 2D
    initializeNeighborsOffsets();
}

void ConnectivityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //if(potts->getDisplayUnitsFlag()){
    //	Unit energyUnit=potts->getEnergyUnit();




    //	CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units");
    //	if (!unitsElem){ //add Units element
    //		unitsElem=_xmlData->attachElement("Units");
    //	}

    //	if(unitsElem->getFirstElement("PenaltyUnit")){
    //		unitsElem->getFirstElement("PenaltyUnit")->updateElementValue(energyUnit.toString());
    //	}else{
    //		CC3DXMLElement * energyElem = unitsElem->attachElement("PenaltyUnit",energyUnit.toString());
    //	}
    //}

    if (_xmlData) {
        penalty = _xmlData->getFirstElement("Penalty")->getDouble();
    }
}

void ConnectivityPlugin::addUnique(CellG *cell, std::vector<CellG *> &_uniqueCells) {

    // Medium does not count
    for (unsigned int i = 0; i < _uniqueCells.size(); i++) {
        if (_uniqueCells[i] == cell)
            return;
    }
    _uniqueCells.push_back(cell);
}

//modified version of connectivity algorithm - Tinri Aegerter
double ConnectivityPlugin::changeEnergy(const Point3D &pt,
                                        const CellG *newCell,
                                        const CellG *oldCell) {

    if (!oldCell) return 0;
    std::vector < CellG * > uniqueCells;
    std::vector <Point3D> n(numberOfNeighbors, Point3D());
    // uniqueCells.clear();
    Neighbor neighbor;

    //rule 1: go over first neighrest neighbors and check if newCell has 1st neighrest neioghbotsbelongin to the same cell
    bool firstFlag = false;
    unsigned int firstMaxIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
    for (int i = 0; i <= firstMaxIndex; ++i) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), i);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }
        if (potts->getCellFieldG()->get(neighbor.pt) == newCell) {
            firstFlag = true;
            break;
        }
    }

    if (!firstFlag) {
        return penalty;
    }


    // Immediate neighbors

    for (int i = 0; i < numberOfNeighbors; ++i) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), offsetsIndex[i]);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }

        n[i] = neighbor.pt;
    }

    //rule 2: Count collisions between pixels belonging to different cells.
    // If the collision count is greater than 2 reject the spin flip
    // mediumFlag = false;
    int collisioncount = 0;
    for (unsigned int i = 0; i < numberOfNeighbors; i++) {
        if (
                ((potts->getCellFieldG()->get(n[i]) == oldCell) ||
                 (potts->getCellFieldG()->get(n[(i + 1) % numberOfNeighbors]) == oldCell))
                &&
                (potts->getCellFieldG()->get(n[i]) != potts->getCellFieldG()->get(n[(i + 1) % numberOfNeighbors]))
                ) {
            addUnique(potts->getCellFieldG()->get(n[i]), uniqueCells);
            addUnique(potts->getCellFieldG()->get(n[(i + 1) % numberOfNeighbors]), uniqueCells);
            collisioncount++;
        }
    }

    if (collisioncount == 2) return 0;  // Accept
    else { // Conditional rejection

        return penalty;  // Reject
    }

}

// Original Implementation - kept as a reference
// double ConnectivityPlugin::changeEnergy(const Point3D &pt,
//                                   const CellG *newCell,
//                                    const CellG *oldCell) {
//  
//    if (!oldCell) return 0;
//    uniqueCells.clear();
// 
//    // Immediate neighbors
//       Neighbor neighbor;
//       for(int i=0 ; i < numberOfNeighbors ; ++i ){
//          neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),offsetsIndex[i]);
//          if(!neighbor.distance){
//          //if distance is 0 then the neighbor returned is invalid
//          continue;
//          }
// 
//          n[i]=neighbor.pt;
//       }
// 
// 
// 
// 
//    mediumFlag = false;
//    int collisioncount = 0;
//    for (unsigned int i = 0; i < numberOfNeighbors; i++)
//    {
//       if (((potts->getCellFieldG()->get(n[i]) == oldCell) ||
//            (potts->getCellFieldG()->get(n[(i+1) % numberOfNeighbors]) == oldCell))
//               &&
//           (potts->getCellFieldG()->get(n[i]) !=
//            potts->getCellFieldG()->get(n[(i+1) % numberOfNeighbors])))
//       {
//         addUnique(potts->getCellFieldG()->get(n[i]));
//         addUnique(potts->getCellFieldG()->get(n[(i+1)%numberOfNeighbors]));
// 	collisioncount++;
//       }
//    }
//    //if (collisioncount == 0) {
      //    CC3D_Log(LOG_TRACE) << "COLLISION COUNT OF 0";
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[0]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[1]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[2]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[3]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[4]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[5]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[6]);
//          CC3D_Log(LOG_TRACE) << potts->getCellFieldG()->get(n[7]);
//        int aaa;
//        cin >> aaa;*/
//    //}
//    if (collisioncount == 2) return 0;  // Accept
//    else { // Conditional rejection
//       if (collisioncount > 2 && uniqueCells.size() == 2 && !mediumFlag) return 0; // Accept
//       else return penalty;  // Reject
//    }
// }

void ConnectivityPlugin::orderNeighborsClockwise
        (Point3D &_midPoint, const std::vector <Point3D> &_offsets, std::vector<int> &_offsetsIndex) {
    //this function maps indexes of neighbors as given by boundary strategy to indexes of neighbors which are in clockwise order
    //that is when iterating over list of neighbors using boundaryStrategy API you can retrieve neighbors in clocwise order if you use
    //neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),offsetsIndex[i]) call
    Neighbor neighbor;
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(_midPoint), nIdx);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }
        for (int i = 0; i < numberOfNeighbors; ++i) {
            if (_midPoint + _offsets[i] == neighbor.pt) {
                _offsetsIndex[i] = nIdx;
            }
        }
    }
}


void ConnectivityPlugin::initializeNeighborsOffsets() {

    Dim3D fieldDim = potts->getCellFieldG()->getDim();
    vector <Point3D> offsets;

    offsets.assign(numberOfNeighbors, Point3D(0, 0, 0));//allocates memory for vector of offsets

    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = 0;

    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(1.45);//second nearest neighbor

    if (fieldDim.x > 1 && fieldDim.y > 1 && fieldDim.z > 1)
        throw CC3DException(
                "This plugin will only work for 2D simulations i.e. one lattice dimansion must be equal to 1 Your simulations appears to be 3D");

    //here we define neighbors  offsets in the "clockwise order"
    if (fieldDim.x == 1) {
        //clockwise offsets
        offsets[0] = Point3D(0, 0, -1);
        offsets[1] = Point3D(0, -1, -1);
        offsets[2] = Point3D(0, -1, 0);
        offsets[3] = Point3D(0, -1, 1);
        offsets[4] = Point3D(0, 0, 1);
        offsets[5] = Point3D(0, 1, 1);
        offsets[6] = Point3D(0, 1, 0);
        offsets[7] = Point3D(0, 1, -1);

        Point3D midPoint(0, fieldDim.y / 2, fieldDim.z / 2);
        orderNeighborsClockwise(midPoint, offsets, offsetsIndex);

    }

    if (fieldDim.y == 1) {

        offsets[0] = Point3D(0, 0, -1);
        offsets[1] = Point3D(-1, 0, -1);
        offsets[2] = Point3D(-1, 0, 0);
        offsets[3] = Point3D(-1, 0, 1);
        offsets[4] = Point3D(0, 0, 1);
        offsets[5] = Point3D(1, 0, 1);
        offsets[6] = Point3D(1, 0, 0);
        offsets[7] = Point3D(1, 0, -1);

        Point3D midPoint(fieldDim.x / 2, 0, fieldDim.z / 2);
        orderNeighborsClockwise(midPoint, offsets, offsetsIndex);


    }

    if (fieldDim.z == 1) {
        offsets[0] = Point3D(0, -1, 0);
        offsets[1] = Point3D(-1, -1, 0);
        offsets[2] = Point3D(-1, 0, 0);
        offsets[3] = Point3D(-1, 1, 0);
        offsets[4] = Point3D(0, 1, 0);
        offsets[5] = Point3D(1, 1, 0);
        offsets[6] = Point3D(1, 0, 0);
        offsets[7] = Point3D(1, -1, 0);

        Point3D midPoint(fieldDim.x / 2, fieldDim.y / 2, 0);
        orderNeighborsClockwise(midPoint, offsets, offsetsIndex);
    }

}

std::string ConnectivityPlugin::toString() {
    return "Connectivity";
}

std::string ConnectivityPlugin::steerableName() {
    return toString();
}