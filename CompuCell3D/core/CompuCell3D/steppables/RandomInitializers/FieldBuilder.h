/*
 * FieldBuilder.cpp
 *
 *  Created on: 31 Jan 2011
 *      Author: palm
 */

 //Author: Margriet Palm CWI, Netherlands
 
#ifndef FIELDBUILDER_H_
#define FIELDBUILDER_H_

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>

// // // #include <string>
// // // #include <map>
// // // #include <vector>

#include "RandomFieldInitializerDLLSpecifier.h"

using namespace CompuCell3D;

//class Potts3D;
class RANDOMINITIALIZERS_EXPORT FieldBuilder {

private:
	WatchableField3D<CellG *> *cellField;
	CellInventory * cellInventoryPtr;
//	BasicClassAccessor<PixelTracker> * pixelTrackerAccessorPtr;
    BasicRandomNumberGenerator * rand;
    Potts3D *potts;
    Dim3D boxMin,boxMax;
    void saveField(int n);
	std::vector<int> xlist;
	std::vector<int> ylist;
	std::vector<int> zlist;
    std::vector<int> typeVec;

public:
//    FieldBuilder();
    FieldBuilder(Simulator *_simulator);
    void setBoxes(Dim3D _boxMin,Dim3D _boxMax);
    void setNeighborListFO();
    void setNeighborListSO();
    void setRandomGenerator(BasicRandomNumberGenerator * _rand);
	void addCell();
    void addCell(Dim3D pos);
	void addCell(Dim3D pos,Dim3D size);
	void setType(CellG * cell);
    void setTypeVec(int ncells,std::vector<std::string> typeNames,std::vector<std::string> biasVec);
    void setTypeVec(int ncells,std::vector<std::string> typeNames);
	void growCells(int steps);
    void addBorderCell(int typeID);
    void showCellStats(int borderID);
};


#endif /* FIELDBUILDER_H_ */
