#ifndef DIFFUSABLEVECTORCOMMON_H
#define DIFFUSABLEVECTORCOMMON_H
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <vector>
#include <string>
#include <iostream>
#include <CompuCell3D/Field3D/Array3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>


namespace CompuCell3D {

	template <typename Y> class Field3DImpl;

	/**
	@author m
	*/
	template <typename precision, template <class> class Array_Type>
	class DiffusableVectorCommon//:public Steppable
	{
		
	public:
		typedef Array_Type<precision> Array_t; 

		DiffusableVectorCommon():
		 // Steppable(),
		  concentrationFieldVector(0),maxNeighborIndex(0),boundaryStrategy(0)
		  {
			using namespace std;
            boundaryStrategy=BoundaryStrategy::getInstance();
			cerr<<"Default constructor DiffusableVectorCommon"<<endl;

		};

		virtual ~DiffusableVectorCommon(){
			clearAllocatedFields();		
			//for(unsigned int i = 0 ; i< concentrationFieldVector.size() ; ++i){
			//   if(concentrationFieldVector[i]){
			//      delete concentrationFieldVector[i];
			//      concentrationFieldVector[i]=0;
			//   }
			//}


		}
		//Field3DImpl<precision> * getConcentrationField(unsigned int i){return concentrationFieldVector[i];};

		virtual Field3DImpl<precision> * getConcentrationField(const std::string & name){
			using namespace std;
			cerr<<"concentrationFieldNameVector.size()="<<concentrationFieldNameVector.size()<<endl;
			for(unsigned int i=0 ; i < concentrationFieldNameVector.size() ; ++i){
				cerr<<"THIS IS FIELD NAME "<<concentrationFieldNameVector[i]<<endl;
			}
			for(unsigned int i=0 ; i < concentrationFieldNameVector.size() ; ++i){
				if(concentrationFieldNameVector[i]==name){
					cerr<<"returning concentrationFieldVector[i]="<<concentrationFieldVector[i]<<endl;
					return concentrationFieldVector[i];  

				}
			}
			cerr<<"returning NULL="<<endl;
			return 0;

		};

		virtual void allocateDiffusableFieldVector(unsigned int numberOfFields,Dim3D fieldDim)
		{
			fieldDimLocal=fieldDim;			
			//       maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(1.1); 
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only
			//       const std::vector<Point3D> & offsetVecRef=BoundaryStrategy::getInstance()->getOffsetVec();
			//       for(int i = 0 ; i <= maxNeighborIndex ; ++i){
			//          offsetVec.push_back(offsetVecRef[i]);
			//       }
			clearAllocatedFields();
			for(unsigned int i = 0 ; i< numberOfFields ; ++i){
				precision val=precision();
				concentrationFieldVector.push_back(new Array_t(fieldDim, val));
			}
			concentrationFieldNameVector.assign(numberOfFields,std::string());
		}

		//std::vector<std::string> getConcentrationFieldNameVector(){ return concentrationFieldNameVector;}
		//std::vector<Array_t<precision> * > getConcentrationFieldVector(){ return concentrationFieldVector;}

		//     unsigned int getMaxNeighborIndex(){return maxNeighborIndex;}
	protected:
		void clearAllocatedFields(){
			for(unsigned int i = 0 ; i< concentrationFieldVector.size() ; ++i){
				if(concentrationFieldVector[i]){
					delete concentrationFieldVector[i];
					concentrationFieldVector[i]=0;
				}
			}
			concentrationFieldVector.clear();

		}

		std::vector<Array_t * > concentrationFieldVector;

		std::vector<std::string> concentrationFieldNameVector;
		unsigned int maxNeighborIndex;
		//    std::vector<Point3D> offsetVec;
		BoundaryStrategy *boundaryStrategy;

		Dim3D fieldDimLocal;

	public:
		BoundaryStrategy const *getBoundaryStrategy()const{return boundaryStrategy;}
		unsigned int getMaxNeighborIndex()const{return maxNeighborIndex;}
		Array_t * getConcentrationField(int n){return concentrationFieldVector[n];}
		void setConcentrationFieldName(int n, std::string const &name){concentrationFieldNameVector[n]=name;}
		std::string getConcentrationFieldName(int n){return concentrationFieldNameVector[n];}


	};

};

#endif
