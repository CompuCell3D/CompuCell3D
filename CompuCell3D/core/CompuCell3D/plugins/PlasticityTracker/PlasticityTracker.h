
#ifndef REALPLASTICITYTRACKER_H
#define REALPLASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "PlasticityTrackerDLLSpecifier.h"


namespace CompuCell3D {

    class CellG;


    class PLASTICITYTRACKER_EXPORT PlasticityTrackerData {
    public:

        PlasticityTrackerData(CellG *_neighborAddress = 0, float _lambdaLength = 0.0, float _targetLength = 0.0)
                : neighborAddress(_neighborAddress), lambdaLength(_lambdaLength), targetLength(_targetLength) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const PlasticityTrackerData &_rhs) const {
            return neighborAddress < _rhs.neighborAddress;
        }

        ///members
        CellG *neighborAddress;
        float lambdaLength;
        float targetLength;

    };

    class PLASTICITYTRACKER_EXPORT PlasticityTracker {
    public:
        PlasticityTracker() {};

        ~PlasticityTracker() {};
        //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
        std::set <PlasticityTrackerData> plasticityNeighbors;

    };
};
#endif


