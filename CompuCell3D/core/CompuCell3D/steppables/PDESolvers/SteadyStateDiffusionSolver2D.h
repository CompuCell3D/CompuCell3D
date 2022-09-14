#ifndef COMPUCELL3DSTEADYSTATEDIFFUSIONSOLVER2D_H
#define COMPUCELL3DSTEADYSTATEDIFFUSIONSOLVER2D_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
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
    class SteadyStateDiffusionSolver2DSerializer;

    // class BoxWatcher;
    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class SteadyStateDiffusionSolver2D;

    class PDESOLVERS_EXPORT SteadyStateSecretionData : public SecretionData {
    public:
        typedef void (SteadyStateDiffusionSolver2D::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT SteadyStateDiffusionSecretionFieldTupple {
    public:
        DiffusionData diffData;
        SteadyStateSecretionData secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SteadyStateSecretionData *getSecretionData() { return &secrData; }
    };


    class PDESOLVERS_EXPORT SteadyStateDiffusionSolver2D
            : public DiffusableVectorFortran<Array2DLinearFortranField3DAdapter> {

        friend class SteadyStateDiffusionSolver2DSerializer;

    public :
        typedef void (SteadyStateDiffusionSolver2D::*diffSecrFcnPtr_t)(void);

        typedef void (SteadyStateDiffusionSolver2D::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef Array2DLinearFortranField3DAdapter ConcentrationField_t;


        double *scratch;
        vector<double> scratchVec;
        vector<double> bdaVec;
        vector<double> bdbVec;
        vector<double> bdcVec;
        vector<double> bddVec;

        // vector<vector<vector<float> > > scratchVec;



        // BoxWatcher *boxWatcherSteppable;

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

        std::vector<bool> manageSecretionInPythonVec; // this flag indicates that secretion will be done entirely in Python and user is fully responsible for proper setup of the solver

        std::vector<bool> periodicBoundaryCheckVector;

        std::vector <BoundaryConditionSpecifier> bcSpecVec;

        CellInventory *cellInventoryPtr;

        void (SteadyStateDiffusionSolver2D::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (SteadyStateDiffusionSolver2D::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void diffuse();

        void diffuseSingleField(unsigned int idx);

        void secrete();

        // void secreteOnContact();
        void secreteSingleField(unsigned int idx);

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

        std::vector <SteadyStateDiffusionSecretionFieldTupple> diffSecrFieldTuppleVec;

        SteadyStateDiffusionSolver2DSerializer *serializerPtr;
        // bool haveCouplingTerms;


        // unsigned int index(unsigned int x,unsigned int y){
        // return workFieldDim.x*y+x;
        // }


    public:

        SteadyStateDiffusionSolver2D();

        virtual ~SteadyStateDiffusionSolver2D();


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

    class PDESOLVERS_EXPORT SteadyStateDiffusionSolver2DSerializer : public Serializer {
    public:
        SteadyStateDiffusionSolver2DSerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~SteadyStateDiffusionSolver2DSerializer() {}

        SteadyStateDiffusionSolver2D *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };


    // class SecretionDataFast:public SecretionData{
    //    public:
    //       std::vector<SteadyStateDiffusionSolver2D::secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    // };



};


#endif
