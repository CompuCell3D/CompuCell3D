#ifndef COMPUCELL3DDIFFUSABLEVECTORFORTRAN_H
#define COMPUCELL3DDIFFUSABLEVECTORFORTRAN_H

#include <CompuCell3D/CC3DExceptions.h>
#include <CompuCell3D/Field3D/Dim3D.h>
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
    template<typename containerType>
    class DiffusableVectorFortran : public Steppable {
    public:
        typedef float precision;  //you have to use typename because otherwise C++ will assume precision_t is an object or function
        //typedef typename containerType::precision_t precision;  //you have to use typename because otherwise C++ will assume precision_t is an object or function

        DiffusableVectorFortran() : Steppable(), concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {

        };

        virtual ~DiffusableVectorFortran() {
            clearAllocatedFields();

        }

        virtual void start() {};

        virtual void step(const unsigned int currentStep) {};

        virtual void finish() {};

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

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only
            const std::vector <Point3D> &offsetVecRef = BoundaryStrategy::getInstance()->getOffsetVec();
            offsetVec.clear();
            for (unsigned int i = 0; i <= maxNeighborIndex; ++i) {
                offsetVec.push_back(offsetVecRef[i]);
            }

            clearAllocatedFields();
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                precision val = precision();

                concentrationFieldVector.push_back(new containerType(fieldDim, val));

            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }


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
            double xVar, yVar, zVar; //variables used by parser
            try {
                parser.DefineVar("x", &xVar);
                parser.DefineVar("y", &yVar);
                parser.DefineVar("z", &zVar);

                parser.SetExpr(_expression);


                for (unsigned int x = 0; x < (unsigned int) fieldDimLocal.x; ++x)
                    for (unsigned int y = 0; y < (unsigned int) fieldDimLocal.y; ++y)
                        for (unsigned int z = 0; z < (unsigned int) fieldDimLocal.z; ++z) {
                            pt.x = x;
                            pt.y = y;
                            pt.z = z;
                            //setting parser variables
                            xVar = x;
                            yVar = y;
                            zVar = z;
                            _field->set(pt, static_cast<float>(parser.Eval()));
                        }

            } catch (mu::Parser::exception_type &e) {
                CC3D_Log(LOG_DEBUG) << e.GetMsg();
                throw CC3DException(e.GetMsg());
            }
        }


        std::vector<containerType *> concentrationFieldVector;
        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;
        std::vector <Point3D> offsetVec;
        BoundaryStrategy *boundaryStrategy;

        Dim3D fieldDimLocal;

    };

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename containerType>
    class DiffusableVectorFortranDouble : public Steppable {
    public:
        typedef typename containerType::precision_t precision;  //you have to use typename becaue otherwise C++ will assume precision_t is an object or function


        DiffusableVectorFortranDouble()
                : Steppable(), concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {

        };

        virtual ~DiffusableVectorFortranDouble() {
            clearAllocatedFields();


        }


        virtual void start() {};

        virtual void step(const unsigned int currentStep) {};

        virtual void finish() {};

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

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only
            const std::vector <Point3D> &offsetVecRef = BoundaryStrategy::getInstance()->getOffsetVec();
            offsetVec.clear();
            for (int i = 0; i <= maxNeighborIndex; ++i) {
                offsetVec.push_back(offsetVecRef[i]);
            }

            clearAllocatedFields();
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                precision val = precision();

                concentrationFieldVector.push_back(new containerType(fieldDim, val));

            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }


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
                throw CC3DException(e.GetMsg());
            }
        }


        std::vector<containerType *> concentrationFieldVector;
        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;
        std::vector <Point3D> offsetVec;
        BoundaryStrategy *boundaryStrategy;

        Dim3D fieldDimLocal;
    };


};

#endif
