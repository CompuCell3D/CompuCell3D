/*
 * FieldBuilder.cpp
 *
 *  Created on: 31 Jan 2011
 *      Author: palm
 */
//Author: Margriet Palm CWI, Netherlands

#include <CompuCell3D/CC3D.h>


#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>

#include "FieldBuilder.h"

using namespace CompuCell3D;
using namespace std;

//FieldBuilder::FieldBuilder(){}
FieldBuilder::FieldBuilder(Simulator *_simulator) {
    potts = _simulator->getPotts();
    cellField = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
}

void FieldBuilder::setRandomGenerator(RandomNumberGenerator *_rand) {
    rand = _rand;
}

void FieldBuilder::setBoxes(Dim3D _boxMin, Dim3D _boxMax) {
    boxMin = _boxMin;
    boxMax = _boxMax;
}

void FieldBuilder::addCell(Dim3D pos) {
    Dim3D size;
    addCell(pos, size);
}

void FieldBuilder::addCell() {
	CC3D_Log(LOG_TRACE) << "add cell";
    Point3D pt, pn;
    bool placed = false;
    bool hasNeighbor;
    int type, n;
    CellG *cell;
    map<double, int>::iterator it;
    while (!placed) {
        pt.x = rand->getInteger(boxMin.x, boxMax.x - 1);
        pt.y = rand->getInteger(boxMin.y, boxMax.y - 1);
        pt.z = rand->getInteger(boxMin.z, boxMax.z - 1);
        n = 0;
        hasNeighbor = false;
        if ((!cellField->get(pt)) && (!hasNeighbor)) {
            placed = true;
            cell = potts->createCellG(pt);
            setType(cell);
            potts->runSteppers();
        }

    }
}

void FieldBuilder::setNeighborListSO() {
	CC3D_Log(LOG_TRACE) << "build second order neighbor list";
    vector<int> subX;
    vector<int> subY;
    vector<int> subZ;
    if ((boxMax.x - boxMin.x) > 1)
        for (int i = -1; i < 2; i++) { subX.push_back(i); }
    else
        subX.push_back(0);
    if ((boxMax.y - boxMin.y) > 1)
        for (int i = -1; i < 2; i++) { subY.push_back(i); }
    else
        subY.push_back(0);
    if ((boxMax.z - boxMin.z) > 1)
        for (int i = -1; i < 2; i++) { subZ.push_back(i); }
    else
        subZ.push_back(0);
    for (int z = 0; z < subZ.size(); z++)
        for (int y = 0; y < subY.size(); y++)
            for (int x = 0; x < subX.size(); x++) {
                if (!((subX[x] == 0) && (subY[y] == 0) && (subZ[z] == 0))) {
                    xlist.push_back(subX[x]);
                    ylist.push_back(subY[y]);
                    zlist.push_back(subZ[z]);
                }
            }

}

void FieldBuilder::setNeighborListFO() {
    // create a list of neighbor pixels
    if ((boxMax.x - boxMin.x) > 1)
        for (int i = -1; i < 2; i += 2) {
            xlist.push_back(i);
            ylist.push_back(0);
            zlist.push_back(0);
        }
    if ((boxMax.y - boxMin.y) > 1)
        for (int i = -1; i < 2; i += 2) {
            ylist.push_back(i);
            xlist.push_back(0);
            zlist.push_back(0);
        }
    if ((boxMax.z - boxMin.z) > 1)
        for (int i = -1; i < 2; i += 2) {
            zlist.push_back(i);
            ylist.push_back(0);
            xlist.push_back(0);
        }

}


void FieldBuilder::addCell(Dim3D pos, Dim3D size) {

    CellG *cell;
    map<double, int>::iterator it;
    cell = potts->createCellG(pos);
    setType(cell);
    potts->runSteppers();
    int dx, dy, dz;
    Point3D p = Point3D(0, 0, 0);
    for (dx = 0; dx < (size.x); dx++)
        for (dy = 0; dy < (size.y); dy++)
            for (dz = 0; dz < (size.z); dz++) {
                if (!((dx == 0) && (dy == 0) && (dz == 0))) {
                    p.x = pos.x + dx;
                    p.y = pos.y + dy;
                    p.z = pos.z + dz;

                    cellField->set(p, cell);
                }
            }
    potts->runSteppers();
}

// cell growth based on eden growth
void FieldBuilder::growCells(int steps) {
    int n, r;
    Point3D pt, cellPt;
    for (n = 0; n < steps; n++) {

        map < Point3D, CellG * > growPoints;
        map<Point3D, CellG *>::iterator it;
        for (pt.z = boxMin.z; pt.z < boxMax.z; pt.z++) {
            for (pt.y = boxMin.y; pt.y < boxMax.y; pt.y++) {
                for (pt.x = boxMin.x; pt.x < boxMax.x; pt.x++) {
                    if (!(cellField->get(pt))) {
                        r = rand->getInteger(0, xlist.size() - 1);
                        cellPt = Point3D(pt.x + xlist[r], pt.y + ylist[r], pt.z + zlist[r]);
                        if (cellField->get(cellPt)) {
                            growPoints[pt] = cellField->get(cellPt);
                        }
                    }
                }
            }
        }
        for (it = growPoints.begin(); it != growPoints.end(); it++) {
            cellField->set((*it).first, (*it).second);
        }
        potts->runSteppers();
    }
}

void FieldBuilder::addBorderCell(int typeID) {
    Dim3D dim = cellField->getDim();
    Point3D pt = Point3D(0, 0, 0);
    CellG *cell = potts->createCellG(pt);
    potts->runSteppers();
    cell->type = typeID;
    int x, y, z;
    for (x = 1; x < dim.x; x++) {
        cellField->set(Point3D(x, 0, 0), cell);
        cellField->set(Point3D(x, dim.y - 1, dim.z - 1), cell);
    }
    for (y = 1; y < dim.y; y++) {
        cellField->set(Point3D(0, y, 0), cell);
        cellField->set(Point3D(dim.x - 1, y, dim.z - 1), cell);
    }
    for (z = 1; z < dim.z; z++) {
        cellField->set(Point3D(0, 0, z), cell);
        cellField->set(Point3D(dim.x - 1, dim.y - 1, z), cell);
    }
    potts->runSteppers();
}

void FieldBuilder::showCellStats(int borderID) {
    CellInventory::cellInventoryIterator cInvItr;
    CellInventory *cellInventoryPtr = &potts->getCellInventory();
    Automaton *automaton = potts->getAutomaton();
    CellG *cell;
    map<int, int> cellmap;
    float n = 0.0;
    float sumvol = 0.0;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        if (cellmap.find(cell->type) == cellmap.end())
            cellmap[cell->type] = 0;
        cellmap[cell->type]++;
        if (cell->type != borderID) {
            n++;
            sumvol += cell->volume;
        }
    }
    map<int, int>::iterator it;
    CC3D_Log(LOG_DEBUG) << "##### INITIAL CONFIGURATION #####";
    CC3D_Log(LOG_DEBUG) << "type\t#";
    for (it = cellmap.begin(); it != cellmap.end(); it++) {
        CC3D_Log(LOG_DEBUG) << automaton->getTypeName((*it).first) << "\t" << (*it).second << endl;
    }

    CC3D_Log(LOG_DEBUG) << "average volume:\t" << (sumvol / ((float) n)) << endl;
    CC3D_Log(LOG_DEBUG) << "#################################";
}

void FieldBuilder::setType(CellG *cell) {
    int r = rand->getInteger(0, typeVec.size() - 1);
    cell->type = typeVec[r];
    typeVec.erase(typeVec.begin() + r);
}

void FieldBuilder::setTypeVec(int ncells, vector <string> typeNames, vector <string> biasVec) {
    Automaton *automaton = potts->getAutomaton();
    vector<string>::size_type sz = biasVec.size();
    map<int, double> nt;
    int i, tid;
    double val, valsum;
    valsum = 0.0;
    for (i = 0; i < sz; i++) { valsum += atof(biasVec[i].c_str()); }
    for (i = 0; i < sz; i++) {
        val = atof(biasVec[i].c_str());
        tid = (int) automaton->getTypeId(typeNames[i]);
        if (val < 1) { typeVec.insert(typeVec.begin(), floor(ncells * val), tid); }
        else if ((val == 1) && (valsum == 1)) { typeVec.insert(typeVec.begin(), floor(ncells * val), tid); }
        else { typeVec.insert(typeVec.begin(), floor(val), tid); }
    }
    i = 0;
    while (typeVec.size() < ncells) {

        typeVec.push_back((int) automaton->getTypeId(typeNames[i]));
        i++;
        if (i > sz) { i = 0; }
    }
}

void FieldBuilder::setTypeVec(int ncells, vector <string> typeNames) {
    Automaton *automaton = potts->getAutomaton();
    vector<string>::size_type sz = typeNames.size();
    map<int, double> nt;
    int i, tid;
    for (i = 0; i < sz; i++) {
        tid = (int) automaton->getTypeId(typeNames[i]);

        typeVec.insert(typeVec.begin(), floor((double) ncells / (double) sz), tid);
    }
    i = 0;
    while ((int) typeVec.size() < ncells) {

        typeVec.push_back((int) automaton->getTypeId(typeNames[i]));
        i++;
        if (i >= sz) { i = 0; }
    }
}
