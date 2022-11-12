#ifndef COMPUCELL3DFIPYCONTIGUOUS_H
#define COMPUCELL3DFIPYCONTIGUOUS_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Steppable.h>
#include <vector>
#include <string>
#include <iostream>
#include <CompuCell3D/Field3D/Array3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <muParser/muParser.h>
#include <Logger/CC3DLogger.h>

namespace mu {

    class Parser; //mu parser class
};

namespace CompuCell3D {

    template<typename Y>
    class Field3DImpl;

    /**
    @author m
    */
    template<typename precision>
    class FiPyContiguous : public Steppable {
    public:
        FiPyContiguous() : Steppable(), concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {
            using namespace std;
            CC3D_Log(LOG_DEBUG) << "Default constructor FiPyContiguous";

        };

        virtual ~FiPyContiguous() {
            clearAllocatedFields();
            //for(unsigned int i = 0 ; i< concentrationFieldVector.size() ; ++i){
            //   if(concentrationFieldVector[i]){
            //      delete concentrationFieldVector[i];
            //      concentrationFieldVector[i]=0;
            //   }
            //}


        }
        //Field3DImpl<precision> * getConcentrationField(unsigned int i){return concentrationFieldVector[i];};

        virtual Field3DImpl<precision> *getConcentrationField(const std::string &name) {
            using namespace std;
            CC3D_Log(LOG_DEBUG) << "concentrationFieldNameVector.size()="<<concentrationFieldNameVector.size();
            for (unsigned int i = 0; i < concentrationFieldNameVector.size(); ++i) {
                CC3D_Log(LOG_DEBUG) << "THIS IS FIELD NAME "<<concentrationFieldNameVector[i];
            }
            for (unsigned int i = 0; i < concentrationFieldNameVector.size(); ++i) {
                if (concentrationFieldNameVector[i] == name) {
                    CC3D_Log(LOG_DEBUG) << "returning concentrationFieldVector[i]="<<concentrationFieldVector[i];
                    return concentrationFieldVector[i];

                }
            }
            CC3D_Log(LOG_DEBUG) << "returning NULL=";
            return 0;

        };

        virtual void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) {
            fieldDimLocal = fieldDim;
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
                concentrationFieldVector.push_back(new Array3DFiPy<precision>(fieldDim, val));
            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }

        std::vector <std::string> getConcentrationFieldNameVector() { return concentrationFieldNameVector; }

        std::vector<Array3DFiPy < precision> * >

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

        void initializeFieldUsingEquation(Field3DImpl<precision> *_field, std::string _expression) {
            Point3D pt;
            mu::Parser parser;
            double xVar, yVar, zVar; //variables used by parser
            try {
                parser.DefineVar("x", &xVar);
                parser.DefineVar("y", &yVar);
                parser.DefineVar("z", &zVar);

                parser.SetExpr(_expression);


                for (unsigned int x = 0; x < fieldDimLocal.x; ++x)
                    for (unsigned int y = 0; y < fieldDimLocal.y; ++y)
                        for (unsigned int z = 0; z < fieldDimLocal.z; ++z) {
                            pt.x = x;
                            pt.y = y;
                            pt.z = z;
                            //setting parser variables
                            xVar = x;
                            yVar = y;
                            zVar = z;
                            _field->set(pt, parser.Eval());
                        }

            } catch (mu::Parser::exception_type &e) {
                CC3D_Log(LOG_DEBUG) << e.GetMsg();
                ASSERT_OR_THROW(e.GetMsg(), 0);
            }
        }

        std::vector<Array3DFiPy < precision> * >
        concentrationFieldVector;

        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;
        //    std::vector<Point3D> offsetVec;
        BoundaryStrategy *boundaryStrategy;

        Dim3D fieldDimLocal;


    };

};

#endif
