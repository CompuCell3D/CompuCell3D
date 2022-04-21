#ifndef COMPUCELL3DREACTIONDIFFUSIONSOLVERFE_SAVHOG_H
#define COMPUCELL3DREACTIONDIFFUSIONSOLVERFE_SAVHOG_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableVector.h"

#include <string>
#include <vector>

#include <iostream>


#include "PDESolversDLLSpecifier.h"

namespace CompuCell3D {

/**
@author m
*/
//forward declarations
    class Potts3D;

    class Simulator;

    class Cell;

    class CellInventory;

    class Automaton;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class PDESOLVERS_EXPORT ReactionDiffusionSolverFE_SavHog : public DiffusableVector<float> {
    public:
        typedef float precision_t;
        typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
        typedef Array3DBordersField3DAdapter <precision_t> ConcentrationField_t;


    protected:

        Potts3D *potts;
        Simulator *simPtr;

        unsigned int currentStep;
        unsigned int maxDiffusionZ;
        float diffConst;
        float decayConst;
        float deltaX;///spacing
        float deltaT;///time interval
        float dt_dx2; ///ratio delta_t/delta_x^2
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;
        Point3D minDiffusionBoxCorner, maxDiffusionBoxCorner;
        bool imposeDiffusionBox;

        bool insideDiffusionBox(Point3D &pt);

        unsigned int dumpFrequency;

        CellInventory *cellInventoryPtr;

        void
        (ReactionDiffusionSolverFE_SavHog::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void diffuse();

        void scrarch2Concentration(ConcentrationField_t *concentrationField, ConcentrationField_t *scratchField);

        void outputField(std::ostream &_out, ConcentrationField_t *_concentrationField);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;
        ///Savill Hogeweg model parameters
        float C1, C2, C3;
        float c1, c2;
        float a;
        float eps1, eps2, eps3;
        float k;
        float b;

        float f(float c);

        float eps(float c);

        bool numberOfFieldsDeclared;
//    unsigned int concentrationFieldCounter;i
        std::vector <std::string> fieldNameVector;


        void initializeConcentration();

        void dumpResults(unsigned int _step);

    public:


        ReactionDiffusionSolverFE_SavHog();

        virtual ~ReactionDiffusionSolverFE_SavHog();


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int _currentStep);

        virtual void finish() {}
        // End Steppable interface


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };

    inline float ReactionDiffusionSolverFE_SavHog::f(float c) {
        if (c < c1)
            return C1 * c;
        else if (c < c2)
            return -C2 * c + a;
        else
            return C3 * (c - 1);
    }


    inline float ReactionDiffusionSolverFE_SavHog::eps(float c) {
        if (c < c1)
            return eps1;
        else if (c < c2)
            return eps2;
        else
            return eps3;

    }


    inline bool ReactionDiffusionSolverFE_SavHog::insideDiffusionBox(Point3D &pt) {
        return (
                pt.x >= minDiffusionBoxCorner.x && pt.x <= maxDiffusionBoxCorner.x &&
                pt.y >= minDiffusionBoxCorner.y && pt.y <= maxDiffusionBoxCorner.y &&
                pt.z >= minDiffusionBoxCorner.z && pt.z <= maxDiffusionBoxCorner.z
        );

    }


};

#endif
