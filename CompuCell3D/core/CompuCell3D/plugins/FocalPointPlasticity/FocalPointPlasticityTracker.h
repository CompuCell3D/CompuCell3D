
#ifndef FOCALPOINTPLASTICITYTRACKER_H
#define FOCALPOINTPLASTICITYTRACKER_H


/**
@author m
*/
#include <set>

#include "FocalPointPlasticityDLLSpecifier.h"
#include <vector>

namespace CompuCell3D {

   class CellG;



   class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTrackerData{
      public:

         FocalPointPlasticityTrackerData(CellG * _neighborAddress=0,float _lambdaDistance=0.0, float _targetDistance=0.0, float _maxDistance=100000.0,int _maxNumberOfJunctions=0, float _activationEnergy=0.0,int _neighborOrder=1, bool _isInitiator=true, int _initMCS=0)
         :neighborAddress(_neighborAddress),lambdaDistance(_lambdaDistance),targetDistance(_targetDistance),maxDistance(_maxDistance),maxNumberOfJunctions(_maxNumberOfJunctions),activationEnergy(_activationEnergy),neighborOrder(_neighborOrder),anchor(false),anchorId(0),isInitiator(_isInitiator), initMCS(_initMCS)
          {
			  
			  anchorPoint=std::vector<float>(3,0.);
		 }

         FocalPointPlasticityTrackerData(const FocalPointPlasticityTrackerData &fpptd) //copy constructor
         {
			  neighborAddress=fpptd.neighborAddress;
			  lambdaDistance=fpptd.lambdaDistance;
			  targetDistance=fpptd.targetDistance;
			  maxDistance=fpptd.maxDistance;
			  activationEnergy=fpptd.activationEnergy;			
			  maxNumberOfJunctions=fpptd.maxNumberOfJunctions;
			  neighborOrder=fpptd.neighborOrder;
			  anchor=fpptd.anchor;
			  anchorId=fpptd.anchorId;
			  anchorPoint=fpptd.anchorPoint;
			  //anchorPoint[0]=fpptd.anchorPoint[0];
			  //anchorPoint[1]=fpptd.anchorPoint[1];
			  //anchorPoint[2]=fpptd.anchorPoint[2];
			  isInitiator = fpptd.isInitiator;
			  initMCS = fpptd.initMCS;

		 }

         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const FocalPointPlasticityTrackerData & _rhs) const{
			// notice that anchor links will be listed first and the last of such links will have highest anchorId 
			 //return neighborAddress < _rhs.neighborAddress || (!(neighborAddress < _rhs.neighborAddress) && anchorId<_rhs.anchorId); //old and wrong implementation of comparison operator might give side effects on windows - it can crash CC3D or in some cases windows OS entirely
             return neighborAddress < _rhs.neighborAddress || (!(_rhs.neighborAddress <neighborAddress ) && anchorId<_rhs.anchorId);


			// return (neighborAddress && _rhs.neighborAddress) ? neighborAddress < _rhs.neighborAddress :
			//	 anchorPoint[0] < _rhs.anchorPoint[0] || (!(_rhs.anchorPoint[0] < anchorPoint[0])&& anchorPoint[1] < _rhs.anchorPoint[1])
			//||(!(_rhs.anchorPoint[0] < anchorPoint[0])&& !(_rhs.anchorPoint[1] <anchorPoint[1] )&& anchorPoint[2] < _rhs.anchorPoint[2]);
         }


         ///members
         CellG * neighborAddress;
         float lambdaDistance;
         float targetDistance;
		 float maxDistance;
		 int maxNumberOfJunctions;
		 float activationEnergy;
		 int neighborOrder;
		 bool anchor;
		 std::vector<float> anchorPoint;
		 bool isInitiator;
		 int initMCS;
	
		 int anchorId;

   };

   class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityTracker{
      public:
         FocalPointPlasticityTracker(){};
         ~FocalPointPlasticityTracker(){};
         std::set<FocalPointPlasticityTrackerData> focalPointPlasticityNeighbors; //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
		 std::set<FocalPointPlasticityTrackerData> internalFocalPointPlasticityNeighbors; //stores ptrs to cell neighbors withon a given cluster i.e. each cell keeps track of its neighbors

		 std::set<FocalPointPlasticityTrackerData> anchors;

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

