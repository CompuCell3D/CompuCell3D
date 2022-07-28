
#ifndef ELASTICITYTRACKER_H
#define ELASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "ElasticityTrackerDLLSpecifier.h"

#include <limits>

#if defined(_WIN32)
#undef max
#undef min
#endif

namespace CompuCell3D {

    class CellG;


    class ELASTICITYTRACKER_EXPORT ElasticityTrackerData {
    public:

        ElasticityTrackerData(CellG *_neighborAddress = 0, float _lambdaLength = 0.0, float _targetLength = 0.0,
                              float _maxLengthElasticity = std::numeric_limits<float>::max())
                : neighborAddress(_neighborAddress), lambdaLength(_lambdaLength), targetLength(_targetLength),
                  maxLengthElasticity(_maxLengthElasticity) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const ElasticityTrackerData &_rhs) const {
            return neighborAddress < _rhs.neighborAddress;
        }

        ///members
        CellG *neighborAddress;
        float lambdaLength;
        float targetLength;
        float maxLengthElasticity;

    };

    class ELASTICITYTRACKER_EXPORT ElasticityTracker {
    public:
        ElasticityTracker() {};

        ~ElasticityTracker() {};
        //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
        std::set <ElasticityTrackerData> elasticityNeighbors;

    };
};
#endif


