#ifndef COMPUCELL3DSTEADYSTATEDIFFUSIONSOLVER_H
#define COMPUCELL3DSTEADYSTATEDIFFUSIONSOLVER_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Array3D.h>
#include "DiffusableVectorFortran.h"

#include "DiffSecrData.h"
#include "BoundaryConditionSpecifier.h"

#include <CompuCell3D/Serializer.h>
#include <CompuCell3D/CC3DEvents.h>

#include <string>

#include <vector>
#include <set>
#include <map>
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

//class DiffusionData;
//class SecretionData;
    class SteadyStateDiffusionSolverSerializer;

// class BoxWatcher;
    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class SteadyStateDiffusionSolver;

//template<typename precision>
    class MyClass {
    public:
        DiffusableVectorFortran<Array3DLinearFortranField3DAdapter> a;
    };


    class OxygenSecretionParameters {

    public:
        OxygenSecretionParameters() : bf(0.0), pblood(0.0), beta(0.0), n(0.0), Khem(0.0), alpha(0.0), Hb(0.0),
                                      delta(0.0), dataInitialized(false) {}

        double bf;
        double pblood;
        double beta;
        double n;
        double Khem;
        double alpha;
        double Hb;
        double delta;
        bool dataInitialized;
    };
//MyClass<float> b;


    class PDESOLVERS_EXPORT SteadyStateSecretionData3D : public SecretionData {
    public:
        typedef void (SteadyStateDiffusionSolver::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT DiffusionSecretionFastFieldTupple3D {
    public:
        DiffusionSecretionFastFieldTupple3D() : useOxygenSecretion(false) {}

        DiffusionData diffData;
        SteadyStateSecretionData3D secrData;
        vector <OxygenSecretionParameters> oxygenSecretionData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SteadyStateSecretionData3D *getSecretionData() { return &secrData; }

        bool useOxygenSecretion;
    };


    class PDESOLVERS_EXPORT SteadyStateDiffusionSolver
            : public DiffusableVectorFortran<Array3DLinearFortranField3DAdapter> {

        friend class SteadyStateDiffusionSolverSerializer;

    public :
        typedef void (SteadyStateDiffusionSolver::*diffSecrFcnPtr_t)(void);

        typedef void (SteadyStateDiffusionSolver::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef Array3DLinearFortranField3DAdapter ConcentrationField_t;

        //typedef Array3DLinearFortranField3DAdapter<float>::precision_t precision_t;
        //typedef Array3DLinearFortranField3DAdapter<precision_t> ConcentrationField_t;

        double *scratch;
        vector<double> scratchVec;
        vector<double> bdaVec;
        vector<double> bdbVec;
        vector<double> bdcVec;
        vector<double> bddVec;
        vector<double> bdeVec;
        vector<double> bdfVec;



        // vector<vector<vector<float> > > scratchVec;



        // BoxWatcher *boxWatcherSteppable;

    protected:

        Potts3D *potts;
        Simulator *simPtr;

        unsigned int currentStep;
        unsigned int maxDiffusionZ;
        double diffConst;
        double decayConst;
        double deltaX;///spacing
        double deltaT;///time interval
        double dt_dx2; ///ratio delta_t/delta_x^2
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;

        std::vector<bool> manageSecretionInPythonVec; // this flag indicates that secretion will be done entirely in Python and user is fully responsible for proper setup of the solver

        std::vector<bool> periodicBoundaryCheckVector;

        std::vector <BoundaryConditionSpecifier> bcSpecVec;


        CellInventory *cellInventoryPtr;

        void (SteadyStateDiffusionSolver::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (SteadyStateDiffusionSolver::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void diffuse();

        void diffuseSingleField(unsigned int idx);

        void secrete();

        // void secreteOnContact();
        void secreteSingleField(unsigned int idx);

        void secreteOxygenSingleField(unsigned int idx);

        // void secreteOnContactSingleField(unsigned int idx);
        // void secreteConstantConcentrationSingleField(unsigned int idx);
        // void scrarch2Concentration(ConcentrationField_t *concentrationField, ConcentrationField_t *scratchField);
        void outputField(std::ostream &_out, ConcentrationField_t *_concentrationField);

        void readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField);

        void boundaryConditionInit(ConcentrationField_t *concentrationField);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        // float couplingTerm(Point3D & _pt,std::vector<CouplingData> & _couplDataVec,float _currentConcentration);
        void initializeConcentration();


        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;

        std::vector <DiffusionSecretionFastFieldTupple3D> diffSecrFieldTuppleVec;

        SteadyStateDiffusionSolverSerializer *serializerPtr;
        // bool haveCouplingTerms;


        // unsigned int index(unsigned int x,unsigned int y){
        // return workFieldDim.x*y+x;
        // }


    public:

        SteadyStateDiffusionSolver();

        virtual ~SteadyStateDiffusionSolver();


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);

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

    class PDESOLVERS_EXPORT SteadyStateDiffusionSolverSerializer : public Serializer {
    public:
        SteadyStateDiffusionSolverSerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~SteadyStateDiffusionSolverSerializer() {}

        SteadyStateDiffusionSolver *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };


};


#endif
