#include "CellVelocityData.h"
#include <iostream>

using namespace std;
namespace CompuCell3D {


    CellVelocityData::CellVelocityData() : enoughData(false), numberOfSamples(0) {

        cellCOMPtr=new cldeque<Coordinates3D<float> >();
   
   cellCOMPtr->assign(cldequeCapacity,Coordinates3D<float>(0.,0.,0.));
   velocity=Coordinates3D<float>(0.,0.,0.);

    }


    CellVelocityData::~CellVelocityData() {
        if (cellCOMPtr) delete cellCOMPtr;
        cellCOMPtr = 0;

    }

    void CellVelocityData::produceVelocityHistoryFromSource(const CellVelocityData *source) {

        ///assume that first place in deque is initialized to the correct COM
        cldeque<Coordinates3D < float> > &cellCOM = *cellCOMPtr;
        cldeque<Coordinates3D < float> > &sourceCellCOM = *(source->cellCOMPtr);

        enoughData = source->enoughData;
        numberOfSamples = source->numberOfSamples;

        unsigned int size = cellCOM.size();

        for (unsigned int i = 1; i < size; ++i) {

            cellCOM[i] = cellCOM[i - 1] - (sourceCellCOM[i - 1] - sourceCellCOM[i]);

        }


    }

};
