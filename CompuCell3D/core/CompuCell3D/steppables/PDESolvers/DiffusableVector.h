#ifndef COMPUCELL3DDIFFUSABLEVECTOR_H
#define COMPUCELL3DDIFFUSABLEVECTOR_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Steppable.h>
#include <vector>
#include <string>
#include <iostream>
#include <CompuCell3D/Field3D/Array3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <Logger/CC3DLogger.h>

namespace CompuCell3D {

    template<typename Y>
    class Field3DImpl;

/**
@author m
*/
    template<typename precision>
    class DiffusableVector : public Steppable {
    public:
        DiffusableVector() : Steppable(), concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {
            using namespace std;
            CC3D_Log(LOG_DEBUG) << "Default constructor DiffusableVector";;

        };

        virtual ~DiffusableVector() {
            clearAllocatedFields();
            //for(unsigned int i = 0 ; i< concentrationFieldVector.size() ; ++i){
            //   if(concentrationFieldVector[i]){
            //      delete concentrationFieldVector[i];
            //      concentrationFieldVector[i]=0;
            //   }
            //}

        
    }
    //Field3DImpl<precision> * getConcentrationField(unsigned int i){return concentrationFieldVector[i];};
    
    virtual Field3D<precision> * getConcentrationField(const std::string & name){
      using namespace std;
      CC3D_Log(LOG_DEBUG) << "concentrationFieldNameVector.size()="<<concentrationFieldNameVector.size();
       for(unsigned int i=0 ; i < concentrationFieldNameVector.size() ; ++i){
         CC3D_Log(LOG_DEBUG) << "THIS IS FIELD NAME "<<concentrationFieldNameVector[i];
       }
      for(unsigned int i=0 ; i < concentrationFieldNameVector.size() ; ++i){
         if(concentrationFieldNameVector[i]==name){
            CC3D_Log(LOG_DEBUG) << "returning concentrationFieldVector[i]="<<concentrationFieldVector[i];
            return concentrationFieldVector[i];

         }
      }
      CC3D_Log(LOG_DEBUG) << "returning NULL=";
            return 0;

        };

        virtual void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) {
            boundaryStrategy = BoundaryStrategy::getInstance();
//       maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(1.1); 
            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only
//       const std::vector<Point3D> & offsetVecRef=BoundaryStrategy::getInstance()->getOffsetVec();
//       for(int i = 0 ; i <= maxNeighborIndex ; ++i){
//          offsetVec.push_back(offsetVecRef[i]);
//       }
            clearAllocatedFields();
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                precision val = precision();
                concentrationFieldVector.push_back(new Array3DBordersField3DAdapter<precision>(fieldDim, val));
            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }

        std::vector <std::string> getConcentrationFieldNameVector() { return concentrationFieldNameVector; }

        std::vector<Array3DBordersField3DAdapter < precision> * >

        getConcentrationFieldVector() { return concentrationFieldVector; }

//     unsigned int getMaxNeighborIndex(){return maxNeighborIndex;}
    protected:
        void clearAllocatedFields() {
            for (unsigned int i = 0; i < concentrationFieldVector.size(); ++i) {
                if (concentrationFieldVector[i]) {
                    delete concentrationFieldVector[i];
                    concentrationFieldVector[i] = 0;
                }
            }
            concentrationFieldVector.clear();

        }

        std::vector<Array3DBordersField3DAdapter < precision> * >
        concentrationFieldVector;

        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;
//    std::vector<Point3D> offsetVec;
        BoundaryStrategy *boundaryStrategy;
    };

};

#endif
