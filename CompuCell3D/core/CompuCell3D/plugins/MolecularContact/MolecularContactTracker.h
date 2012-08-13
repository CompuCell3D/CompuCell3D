#ifndef MOLECULARCONTACTTRACKER_H
#define MOLECULARCONTACTTRACKER_H


/**
@author m
*/
#include <set>

//#include <CompuCell3D/dllDeclarationSpecifier.h>
#include "MolecularContactDLLSpecifier.h"

namespace CompuCell3D {

   class CellG;
   

   class DECLSPECIFIER MolecularContactTrackerData{
      public:

         MolecularContactTrackerData(CellG * _neighborAddress=0,float _lambdaLength=0.0, float _targetLength=0.0)
         :neighborAddress(_neighborAddress),lambdaLength(lambdaLength),targetLength(_targetLength)
          {}

         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const MolecularContactTrackerData & _rhs) const{
            return neighborAddress < _rhs.neighborAddress;
         }
         ///members
         CellG * neighborAddress;
         float lambdaLength;
         float targetLength;

   };

   class DECLSPECIFIER MolecularContactTracker{
      public:
         MolecularContactTracker(){};
         ~MolecularContactTracker(){};
         std::set<MolecularContactTrackerData> molecularcontactNeighbors; //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors

   };
};
#endif


