#ifndef COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_H
#define COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include "DiffSecrData.h"

#include <CompuCell3D/Serializer.h>

#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>

#include "PDESolversDLLSpecifier.h"

struct SolverParams;

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

    class BoxWatcher;

//GPU PART
    class CellTypeMonitorPlugin;

    class BoundaryMonitorPlugin;

    class FlexibleDiffusionSolverFE_GPU_Device;
//GPU PART

    class DiffusionData;

//GPU PART
    template<typename GPU_Solver>
    class SecretionDataFlex_GPU;

    template<typename GPU_Solver>
    class FlexibleDiffusionSolverSerializer_GPU;
//GPU PART

//class TestFlexibleDiffusionSolver; // Testing FlexibleDiffusionSolverFE
    class ParallelUtilsOpenMP;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

//GPU PART
    template<typename GPU_Solver>
    class FlexibleDiffusionSolverFE_GPU;

    template<typename GPU_Solver>
    class PDESOLVERS_EXPORT SecretionDataFlex_GPU : public SecretionData {
    public:
        typedef void (FlexibleDiffusionSolverFE_GPU<GPU_Solver>::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    template<typename GPU_Solver>
    class PDESOLVERS_EXPORT DiffusionSecretionFlexFieldTupple_GPU {
    public:
        DiffusionData diffData;
        SecretionDataFlex_GPU<GPU_Solver> secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataFlex_GPU<GPU_Solver> *getSecretionData() { return &secrData; }
    };
//GPU PART

    template<typename GPU_Solver>
    class PDESOLVERS_EXPORT FlexibleDiffusionSolverFE_GPU
            : public DiffusableVectorCommon<float, Array3DCUDA>, public Steppable {
        template<typename GPU_Solver_foo>
        friend
        class FlexibleDiffusionSolverSerializer_GPU;

//suppress checking for MSVC 2008 and earlier. Should work in Linux but then c0x compilation flags must be set
#if defined _WIN32 && _MSC_VER >= 1600
        static_assert(std::is_base_of<FlexibleDiffusionSolverFE_GPU_Device, GPU_Solver>::value&&
            !std::is_same<FlexibleDiffusionSolverFE_GPU_Device, GPU_Solver>::value,
            "<GPU_Solver> template parameter must be derived from FlexibleDiffusionSolverFE_GPU_Device");
#endif

        // For Testing
        //friend class TestFlexibleDiffusionSolver; // In production version you need to enclose with #ifdef #endif

    public :
        typedef void (FlexibleDiffusionSolverFE_GPU::*diffSecrFcnPtr_t)(void);

        typedef void (FlexibleDiffusionSolverFE_GPU::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef float precision_t;
        //typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
        typedef Array3DCUDA <precision_t> ConcentrationField_t;

        BoxWatcher *boxWatcherSteppable;
        CellTypeMonitorPlugin *cellTypeMonitorPlugin;

        BoundaryMonitorPlugin *boundaryMonitorPlugin;

    protected:

        Potts3D *potts;
        Simulator *simPtr;
        ParallelUtilsOpenMP *pUtils;
        int gpuDeviceIndex;//requested GPU device index. "-1" stands for automatic selection

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

        std::vector <BoundaryConditionSpecifier> bcSpecVec;
        std::vector<bool> bcSpecFlagVec;

        CellInventory *cellInventoryPtr;

        void (FlexibleDiffusionSolverFE_GPU::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (FlexibleDiffusionSolverFE_GPU::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
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

        //void boundaryConditionInit(ConcentrationField_t *concentrationField);
        void boundaryConditionInit(int idx);

        bool isBoudaryRegion(int x, int y, int z, Dim3D dim);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        float couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec, float _currentConcentration);

        void initializeConcentration();

        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;

        FlexibleDiffusionSolverSerializer_GPU<GPU_Solver> *serializerPtr;
        bool haveCouplingTerms;
        std::vector <DiffusionSecretionFlexFieldTupple_GPU<GPU_Solver>> diffSecrFieldTuppleVec;
        //vector<string> concentrationFieldNameVectorTmp;

        //CUDA PART
        //CUDA MEMBERS
        int mem_size_field;
        int mem_size_celltype_field;


        //host
        Array3DCUDA<unsigned char> *h_celltype_field;
        Array3DCUDA<unsigned char> *h_boundary_field;
        //SolverParams *h_solverParamPtr;

        //device
        FlexibleDiffusionSolverFE_GPU_Device *gpuDevice;
/*	float * d_field;
	unsigned char * d_celltype_field;
	unsigned char * d_boundary_field;
	float * d_scratch;
	SolverParams *d_solverParam;*/
        //CUDA PART

    public:

        FlexibleDiffusionSolverFE_GPU();

        virtual ~FlexibleDiffusionSolverFE_GPU();


        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

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

    template<typename GPU_Solver>
    class PDESOLVERS_EXPORT FlexibleDiffusionSolverSerializer_GPU : public Serializer {
    public:
        FlexibleDiffusionSolverSerializer_GPU() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~FlexibleDiffusionSolverSerializer_GPU() {}

        FlexibleDiffusionSolverFE_GPU<GPU_Solver> *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };

} //namespace CompuCell3D

#include "FlexibleDiffusionSolverFE_GPU.hpp"


#endif