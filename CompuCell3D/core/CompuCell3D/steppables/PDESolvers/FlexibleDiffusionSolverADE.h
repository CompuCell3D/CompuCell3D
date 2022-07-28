#ifndef COMPUCELL3DFlexibleDiffusionSolverADE_H
#define COMPUCELL3DFlexibleDiffusionSolverADE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableVector.h"

#include "DiffSecrData.h"

#include <CompuCell3D/Serializer.h>

#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>


//#include "FlexibleDiffusionSolverFE.h" // include the class FlexibleDiffusionSolverSerializer

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

    class DiffusionData;

    class SecretionDataADE;

//class FlexibleDiffusionSolverSerializer;
    class TestFlexibleDiffusionSolver; // Testing FlexibleDiffusionSolverADE

    class FlexibleDiffusionSolverADE;

    class FlexibleDiffusionSolverADESerializer;

    class PDESOLVERS_EXPORT SecretionDataADE : public SecretionData {
    public:
        typedef void (FlexibleDiffusionSolverADE::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT DiffusionSecretionADEFieldTupple {
    public:
        DiffusionData diffData;
        SecretionDataADE secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataADE *getSecretionData() { return &secrData; }
    };


    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

/* Implements ADE (Alternating Direction Explicit) finite difference scheme for the diffusion equation.
 *
 */
    class PDESOLVERS_EXPORT FlexibleDiffusionSolverADE : public DiffusableVector<float> {

        friend class FlexibleDiffusionSolverSerializer;

        friend class TestFlexibleDiffusionSolver;

    public :
        typedef void (FlexibleDiffusionSolverADE::*diffSecrFcnPtr_t)(void);

        typedef void (FlexibleDiffusionSolverADE::*secrSingleFieldFcnPtr_t)(unsigned int);

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

//    std::vector<DiffusionData> diffDataVec;
//    std::vector<SecretionDataFlex> secrDataVec;
        std::vector<bool> periodicBoundaryCheckVector;


        CellInventory *cellInventoryPtr;

        void (FlexibleDiffusionSolverADE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (FlexibleDiffusionSolverADE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void diffuse();

        void diffuseSingleField(unsigned int idx);

        void secrete();

        void secreteOnContact();

        void secreteSingleField(unsigned int idx);

        void secreteOnContactSingleField(unsigned int idx);

        void secreteConstantConcentrationSingleField(unsigned int idx);

        void scrarch2Concentration(ConcentrationField_t *concentrationField, ConcentrationField_t *scratchField);

        void outputField(std::ostream &_out, ConcentrationField_t *_concentrationField);

        void readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField);

        void boundaryConditionInit(ConcentrationField_t *concentrationField);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        float couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec, float _currentConcentration);

        void initializeConcentration();

        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;
        float A, B; // ADE parameters

        FlexibleDiffusionSolverADESerializer *serializerPtr;
        bool haveCouplingTerms;
        std::vector <DiffusionSecretionADEFieldTupple> diffSecrFieldTuppleVec;

    public:


        FlexibleDiffusionSolverADE();

        virtual ~FlexibleDiffusionSolverADE();


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int _currentStep);

        virtual void finish();
        // End Steppable interface


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };

    class PDESOLVERS_EXPORT FlexibleDiffusionSolverADESerializer : public Serializer {
    public:
        FlexibleDiffusionSolverADESerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~FlexibleDiffusionSolverADESerializer() {}

        FlexibleDiffusionSolverADE *solverPtr;

        virtual void serialize() {}

        virtual void readFromFile() {}

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };

};


#endif
