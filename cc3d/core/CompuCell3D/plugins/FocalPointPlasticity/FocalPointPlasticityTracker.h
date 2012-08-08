
#ifndef FOCALPOINTPLASTICITYTRACKER_H
#define FOCALPOINTPLASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "FocalPointPlasticityDLLSpecifier.h"


namespace CompuCell3D {

   class CellG;



   class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTrackerData{
      public:

         FocalPointPlasticityTrackerData(CellG * _neighborAddress=0,float _lambdaDistance=0.0, float _targetDistance=0.0, float _maxDistance=100000.0,int _maxNumberOfJunctions=0, float _activationEnergy=0.0,int _neighborOrder=1)
         :neighborAddress(_neighborAddress),lambdaDistance(_lambdaDistance),targetDistance(_targetDistance),maxDistance(_maxDistance),maxNumberOfJunctions(_maxNumberOfJunctions),activationEnergy(_activationEnergy),neighborOrder(_neighborOrder)
          {}

         FocalPointPlasticityTrackerData(const FocalPointPlasticityTrackerData &fpptd) //copy constructor
         {
			  neighborAddress=fpptd.neighborAddress;
			  lambdaDistance=fpptd.lambdaDistance;
			  targetDistance=fpptd.targetDistance;
			  maxDistance=fpptd.maxDistance;
			  activationEnergy=fpptd.activationEnergy;			
			  maxNumberOfJunctions=fpptd.maxNumberOfJunctions;
			  neighborOrder=fpptd.neighborOrder;

		 }

         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const FocalPointPlasticityTrackerData & _rhs) const{
            return neighborAddress < _rhs.neighborAddress;
         }


         ///members
         CellG * neighborAddress;
         float lambdaDistance;
         float targetDistance;
		 float maxDistance;
		 int maxNumberOfJunctions;
		 float activationEnergy;
		 int neighborOrder;

   };

   class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTracker{
      public:
         FocalPointPlasticityTracker(){};
         ~FocalPointPlasticityTracker(){};
         std::set<FocalPointPlasticityTrackerData> focalPointPlasticityNeighbors; //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
		 std::set<FocalPointPlasticityTrackerData> internalFocalPointPlasticityNeighbors; //stores ptrs to cell neighbors withon a given cluster i.e. each cell keeps track of its neighbors
		 FocalPointPlasticityTrackerData fpptd;
   };

   class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityJunctionCounter{
	  unsigned char type;	  
	public:
		FocalPointPlasticityJunctionCounter(unsigned char _type){
			type=_type;
		}

		bool operator() (const FocalPointPlasticityTrackerData & _fpptd){
			return _fpptd.neighborAddress->type==type;
		}
		
   };

};
#endif

