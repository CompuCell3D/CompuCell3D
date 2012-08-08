
#ifndef ELASTICITYTRACKER_H
#define ELASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "ElasticityDLLSpecifier.h"

#include <limits>
#undef max
#undef min


namespace CompuCell3D {

   class CellG;
   

   class ELASTICITY_EXPORT ElasticityTrackerData{
      public:

			ElasticityTrackerData(CellG * _neighborAddress=0,float _lambdaLength=0.0, float _targetLength=0.0,float _maxLengthElasticity=std::numeric_limits<float>::max())
         :neighborAddress(_neighborAddress),lambdaLength(_lambdaLength),targetLength(_targetLength),maxLengthElasticity(_maxLengthElasticity)
          {}

         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const ElasticityTrackerData & _rhs) const{
            return neighborAddress < _rhs.neighborAddress;
         }
         ///members
         CellG * neighborAddress;
         float lambdaLength;
         float targetLength;
         float maxLengthElasticity;

   };

   class ELASTICITY_EXPORT ElasticityTracker{
      public:
         ElasticityTracker(){};
         ~ElasticityTracker(){};
         std::set<ElasticityTrackerData> elasticityNeighbors; //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors

   };
};
#endif


