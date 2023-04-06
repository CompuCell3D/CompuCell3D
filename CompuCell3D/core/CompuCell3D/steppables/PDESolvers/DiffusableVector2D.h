#ifndef COMPUCELL3DDIFFUSABLEVECTOR2D_H
#define COMPUCELL3DDIFFUSABLEVECTOR2D_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Steppable.h>
#include <vector>
#include <string>
#include <iostream>
#include <CompuCell3D/Field3D/Array3D.h>
#include <muParser/muParser.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
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
    class DiffusableVector2D : public Steppable {
    public:
        DiffusableVector2D() : Steppable(), concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {

        };

        virtual ~DiffusableVector2D() {
            clearAllocatedFields();



        }


        virtual Field3D <precision> *getConcentrationField(const std::string &name) {
            using namespace std;
            for (unsigned int i = 0; i < concentrationFieldNameVector.size(); ++i) {
                if (concentrationFieldNameVector[i] == name) {
                    return concentrationFieldVector[i];

                }
            }
            return 0;

        };

        virtual void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) {
            fieldDimLocal = fieldDim;
            boundaryStrategy = BoundaryStrategy::getInstance();
            //       maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(1.1); //for nearest neighbors only
            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only
            const std::vector <Point3D> &offsetVecRef = BoundaryStrategy::getInstance()->getOffsetVec();
            offsetVec.clear();
            for (int i = 0; i <= (int) maxNeighborIndex; ++i) {
                offsetVec.push_back(offsetVecRef[i]);
            }

            clearAllocatedFields();
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                precision val = precision();
                concentrationFieldVector.push_back(new Array2DContiguous<precision>(fieldDim, val));
            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }

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

        void initializeFieldUsingEquation(Field3D <precision> *_field, std::string _expression) {
            Point3D pt;
            mu::Parser parser;
            double xVar, yVar; //variables used by parser
            try {
                parser.DefineVar("x", &xVar);
                parser.DefineVar("y", &yVar);


                parser.SetExpr(_expression);


                for (int x = 0; x < fieldDimLocal.x; ++x)
                    for (int y = 0; y < fieldDimLocal.y; ++y) {
                        pt.x = x;
                        pt.y = y;
                        //setting parser variables
                        xVar = x;
                        yVar = y;
                        _field->set(pt, static_cast<float>(parser.Eval()));
                    }

            } catch (mu::Parser::exception_type &e) {
                CC3D_Log(LOG_DEBUG) << e.GetMsg();
                throw CC3DException(e.GetMsg());
            }
        }


        std::vector<Array2DContiguous < precision> * >
        concentrationFieldVector;
        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;
        std::vector <Point3D> offsetVec;
        BoundaryStrategy *boundaryStrategy;

        Dim3D fieldDimLocal;
    };

};

#endif
