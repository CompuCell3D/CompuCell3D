#ifndef COMPUCELL3DKERNELDIFFUSIONSOLVERFE_H
#define COMPUCELL3DKERNELDIFFUSIONSOLVERFE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
//#include "DiffusableVector.h"
#include "DiffusableVectorCommon.h"

#include "DiffSecrData.h"


#include <CompuCell3D/Serializer.h>

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

    class ParallelUtilsOpenMP;

    class KernelDiffusionSolverSerializer;

    class BoxWatcher;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;


    class KernelDiffusionSolver;

    class PDESOLVERS_EXPORT SecretionDataKernel : public SecretionData {
    public:
        typedef void (KernelDiffusionSolver::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT DiffusionSecretionKernelFieldTupple {
    public:
        DiffusionData diffData;
        SecretionDataKernel secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataKernel *getSecretionData() { return &secrData; }
    };


    class PDESOLVERS_EXPORT KernelDiffusionSolver
            : public DiffusableVectorCommon<float, Array3DContiguous>, public Steppable {

        friend class KernelDiffusionSolverSerializer;

    public :
        typedef void (KernelDiffusionSolver::*diffSecrFcnPtr_t)(void);

        typedef void (KernelDiffusionSolver::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef float precision_t;
        //typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
        //typedef Array3DBordersField3DAdapter<precision_t> ConcentrationField_t;
        typedef Array3DContiguous <precision_t> ConcentrationField_t;
        typedef vector <vector<vector < float>> >
        Array3D;

        float *scratch;
        vector <vector<vector < float>> >
        scratchVec;


        BoxWatcher *boxWatcherSteppable;

    protected:

        Potts3D *potts;
        Simulator *simPtr;
        ParallelUtilsOpenMP *pUtils;

        unsigned int currentStep;
        unsigned int maxDiffusionZ;
        vector<int> kernel;
        vector <vector<float>> NKer;
        int maxNeighborIndex;
        vector<int> tempmaxNeighborIndex;
        vector <Array3D> GreensFunc;
        vector <Point3D> neighbors;

        map<int, int> neighborDistance;
        map<int, int>::iterator neighborIter;

        float diffConst;
        float decayConst;
        float deltaX;///spacing
        float deltaT;///time interval
        float dt_dx2; ///ratio delta_t/delta_x^2
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;

        std::vector<bool> periodicBoundaryCheckVector;


        CellInventory *cellInventoryPtr;

        void (KernelDiffusionSolver::*diffusePtr)(void);///ptr to member method - diffusion solver
        void (KernelDiffusionSolver::*secretePtr)(void);///ptr to member method -  diffusion solver
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

        void initializeKernel(Simulator *simulator);

        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;

        KernelDiffusionSolverSerializer *serializerPtr;
        bool haveCouplingTerms;

        unsigned int index(unsigned int x, unsigned int y) {
            return workFieldDim.x * y + x;
        }

        std::vector <DiffusionSecretionKernelFieldTupple> diffSecrFieldTuppleVec;
        std::vector<unsigned int> coarseGrainFactorVec;

        void writePixelValue(Point3D pt, float value, unsigned int coarseGrainFactor,
                             ConcentrationField_t &_concentrationField);

        std::vector<unsigned int> coarseGrainMultiplicativeFactorVec;

    public:


        KernelDiffusionSolver();

        virtual ~KernelDiffusionSolver();


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

    class PDESOLVERS_EXPORT KernelDiffusionSolverSerializer : public Serializer {
    public:
        KernelDiffusionSolverSerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~KernelDiffusionSolverSerializer() {}

        KernelDiffusionSolver *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };


};


#endif
