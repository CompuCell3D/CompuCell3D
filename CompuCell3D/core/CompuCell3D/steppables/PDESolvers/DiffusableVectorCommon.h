#ifndef COMPUCELL3DDIFFUSABLEVECTORCOMMON_H
#define COMPUCELL3DDIFFUSABLEVECTORCOMMON_H

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
    template<typename precision, template<class> class Array_Type>
    class DiffusableVectorCommon//:public Steppable
    {

    public:
        typedef Array_Type<precision> Array_t;

        DiffusableVectorCommon() :
        // Steppable(),
                concentrationFieldVector(0), maxNeighborIndex(0), boundaryStrategy(0) {
            using namespace std;
            boundaryStrategy = BoundaryStrategy::getInstance();
            CC3D_Log(LOG_DEBUG) << "Default constructor DiffusableVectorCommon";

        };

        virtual ~DiffusableVectorCommon() {
            clearAllocatedFields();


        }


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

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//for nearest neighbors only

            clearAllocatedFields();
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                precision val = precision();
                concentrationFieldVector.push_back(new Array_t(fieldDim, val));
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

        std::vector<Array_t *> concentrationFieldVector;

        std::vector <std::string> concentrationFieldNameVector;
        unsigned int maxNeighborIndex;

        BoundaryStrategy *boundaryStrategy;

        Dim3D fieldDimLocal;

    public:
        BoundaryStrategy const *getBoundaryStrategy() const { return boundaryStrategy; }

        unsigned int getMaxNeighborIndex() const { return maxNeighborIndex; }

        Array_t *getConcentrationField(int n) { return concentrationFieldVector[n]; }

        void setConcentrationFieldName(int n, std::string const &name) { concentrationFieldNameVector[n] = name; }

        std::string getConcentrationFieldName(int n) { return concentrationFieldNameVector[n]; }

        void initializeFieldUsingEquation(Field3DImpl<precision> *_field, std::string _expression) {
            Point3D pt;
            mu::Parser parser;
            double xVar, yVar, zVar; //variables used by parser
            try {
                parser.DefineVar("x", &xVar);
                parser.DefineVar("y", &yVar);
                parser.DefineVar("z", &zVar);

                parser.SetExpr(_expression);


                for (int x = 0; x < fieldDimLocal.x; ++x)
                    for (int y = 0; y < fieldDimLocal.y; ++y)
                        for (int z = 0; z < fieldDimLocal.z; ++z) {
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


    };

};

#endif
