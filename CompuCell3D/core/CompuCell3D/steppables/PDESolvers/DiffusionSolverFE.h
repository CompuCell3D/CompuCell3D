#ifndef DIFFUSIONSOLVERFE_H
#define DIFFUSIONSOLVERFE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

#include "DiffusableVectorCommon.h"

#include "DiffSecrData.h"
#include "FluctuationCompensator.h"
#include "BoundaryConditionSpecifier.h"

#include <CompuCell3D/Serializer.h>
#include <CompuCell3D/CC3DEvents.h>


#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>

#include "PDESolversDLLSpecifier.h"
#include <Logger/CC3DLogger.h>

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

    class DiffusionData;

    class FluctuationCompensator;

    template<class Cruncher>
    class PDESOLVERS_EXPORT SecretionDataDiffusionFE;

    template<class Cruncher>
    class PDESOLVERS_EXPORT DiffusionSolverSerializer;

    class TestDiffusionSolver; // Testing DiffusionSolverFE
    class ParallelUtilsOpenMP;

    class CellTypeMonitorPlugin;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;


    template<class Cruncher>
    class PDESOLVERS_EXPORT DiffusionSolverFE;

    template<class Cruncher>
    class PDESOLVERS_EXPORT SecretionDataDiffusionFE : public SecretionData {
    public:
        typedef void (DiffusionSolverFE<Cruncher>::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    template<class Cruncher>
    class PDESOLVERS_EXPORT DiffusionSecretionDiffusionFEFieldTupple {
    public:
        DiffusionData diffData;
        SecretionDataDiffusionFE<Cruncher> secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataDiffusionFE<Cruncher> *getSecretionData() { return &secrData; }
    };

//CRT pattern is used to extract the common for CPU and GPU code part
//http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
    template<class Cruncher>
    class PDESOLVERS_EXPORT DiffusionSolverFE : public Steppable {

        template<class CruncherFoo>
        friend
        class PDESOLVERS_EXPORT DiffusionSolverSerializer;

        // For Testing
        friend class TestDiffusionSolver; // In production version you need to enclose with #ifdef #endif

    public :
        typedef void (DiffusionSolverFE::*diffSecrFcnPtr_t)(void);

        typedef void (DiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef float precision_t;
        float m_RDTime;//time spent in solving Reaction-Diffusion solver, ms


        BoxWatcher *boxWatcherSteppable;

        float diffusionLatticeScalingFactor; // for hex in 2Dlattice it is 2/3.0 , for 3D is 1/2.0, for cartesian lattice it is 1
        bool autoscaleDiffusion;
        bool scaleSecretion; // this flag is set to true. If user sets it to false via XML then DiffusionSolver will behave like FlexibleDiffusion solver - i.e. secretion will be done in one step followed by multiple diffusive steps

        FluctuationCompensator *fluctuationCompensator;

    protected:

        Potts3D *potts;
        Simulator *simPtr;
        ParallelUtilsOpenMP *pUtils;

        unsigned int currentStep;
        unsigned int maxDiffusionZ;
        float diffConst;

        float decayConst;
        float deltaX;///spacing
        float deltaT;///time interval
        float dt_dx2; ///ratio delta_t/delta_x^2
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;


        std::vector<bool> periodicBoundaryCheckVector;

        std::vector <BoundaryConditionSpecifier> bcSpecVec;
        std::vector<bool> bcSpecFlagVec;


        CellInventory *cellInventoryPtr;

        void (DiffusionSolverFE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (DiffusionSolverFE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver



        bool hasAdditionalTerms() const;

        void diffuse();

        void secrete();

        void secreteOnContact();

        virtual void secreteSingleField(unsigned int idx);

        virtual void secreteOnContactSingleField(unsigned int idx);

        virtual void secreteConstantConcentrationSingleField(unsigned int idx);

        template<typename ConcentrationField_t>
        void scrarch2Concentration(ConcentrationField_t *concentrationField, ConcentrationField_t *scratchField);

        template<typename ConcentrationField_t>
        void outputField(std::ostream &_out, ConcentrationField_t *_concentrationField);

        template<typename ConcentrationField_t>
        void readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField);


        virtual void
        boundaryConditionIndicatorInit(); // this function initializes indicator only not the actual boundary conditions used on non-cartesian lattices
        virtual void boundaryConditionInit(int idx);
        void init_cell_type_and_id_arrays();
        bool isBoudaryRegion(int x, int y, int z, Dim3D dim);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        float couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec, float _currentConcentration);

        void initializeConcentration();

        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;

        DiffusionSolverSerializer<Cruncher> *serializerPtr;
        bool haveCouplingTerms;
        std::vector <DiffusionSecretionDiffusionFEFieldTupple<Cruncher>> diffSecrFieldTuppleVec;


        //used to deal with large diffusion constants
        int scalingExtraMCS;
        std::vector<int> scalingExtraMCSVec; //TODO: check if used
        std::vector<float> maxDiffConstVec;
        std::vector<float> maxDecayConstVec;
        float maxStableDiffConstant;
        float maxStableDecayConstant;

        std::vector<float> diffConstVec;
        std::vector<float> decayConstVec;

        CellTypeMonitorPlugin *cellTypeMonitorPlugin;
        Array3DCUDA<unsigned char> *h_celltype_field;
        Array3DCUDA<float> *h_cellid_field;

        Array3DCUDA<signed char> *bc_indicator_field;


        std::vector <std::vector<Point3D>> hexOffsetArray;
        std::vector <Point3D> offsetVecCartesian;
        LatticeType latticeType;

        const std::vector <Point3D> &getOffsetVec(Point3D &pt) const {
            if (latticeType == HEXAGONAL_LATTICE) {
                return hexOffsetArray[(pt.z % 3) * 2 + pt.y % 2];
            } else {
                return offsetVecCartesian;
            }
        }

        bool checkIfOffsetInArray(Point3D _pt, std::vector <Point3D> &_array);

        void prepareForwardDerivativeOffsets();


        //functions to realize in derived classes
        virtual void initImpl() = 0;//first step of initialization
        virtual void extraInitImpl() = 0;//second step of initialization, when more parameters are known
        virtual void initCellTypesAndBoundariesImpl() = 0;

        virtual void solverSpecific(CC3DXMLElement *_xmlData) = 0;//reading solver-specific information from XML file
        virtual std::string
        toStringImpl() = 0; //to string has to be customized too it has to return correct name for the solver depending if it is GPU or CPU solver

        //for debugging
        template<typename ConcentrationField_t>
        void CheckConcentrationField(ConcentrationField_t &concentrationField) const;


    public:

        DiffusionSolverFE();

        virtual ~DiffusionSolverFE();


        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int _currentStep);

        virtual void finish();

        // End Steppable interface

        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        int getFieldsCount() const { return diffSecrFieldTuppleVec.size(); }

    protected:
        virtual void Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant,
                           std::vector<float> const &maxDecayConstVec);

        virtual void prepCellTypeField(int idx);

        virtual Dim3D getInternalDim();

        //if an array used for storing has an extra boundary layer around it
        virtual bool hasExtraLayer() const;

        virtual void diffuseSingleField(unsigned int idx);

        virtual void stepImpl(const unsigned int _currentStep);

        unsigned int fieldsCount() const { return diffSecrFieldTuppleVec.size(); }



    };

    template<class Cruncher>
    class PDESOLVERS_EXPORT DiffusionSolverSerializer : public Serializer {
    public:
        DiffusionSolverSerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~DiffusionSolverSerializer() {}

        DiffusionSolverFE<Cruncher> *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };

//for debugging
    template<class Cruncher>
    template<class ConcentrationField_t>
    void DiffusionSolverFE<Cruncher>::CheckConcentrationField(ConcentrationField_t &concentrationField) const {

        double sum = 0.f;
        float minVal = numeric_limits<float>::max();
        float maxVal = -numeric_limits<float>::max();
        for (int z = 1; z <= fieldDim.z; ++z) {
            for (int y = 1; y <= fieldDim.y; ++y) {
                for (int x = 1; x <= fieldDim.x; ++x) {
                    //float val=h_field[z*(fieldDim.x+2)*(fieldDim.y+2)+y*(fieldDim.x+2)+x];
                    float val = concentrationField.getDirect(x, y, z);
                    if (!isfinite(val)) {
                        CC3D_Log(LOG_DEBUG) << "NaN at position: " << x << "x" << y << "x" << z;
                        continue;
                    }

                    sum += val;
                    minVal = std::min(val, minVal);
                    maxVal = std::max(val, maxVal);
                }
            }
        }

        CC3D_Log(LOG_DEBUG) << "min: " << minVal << "; max: " << maxVal << " " << sum;
    };


};

#endif