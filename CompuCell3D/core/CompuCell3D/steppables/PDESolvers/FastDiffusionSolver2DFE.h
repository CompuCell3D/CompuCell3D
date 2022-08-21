#ifndef COMPUCELL3DFASTDIFFUSIONSOLVER2DFE_H
#define COMPUCELL3DFASTDIFFUSIONSOLVER2DFE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableVector2D.h"

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

/*class DiffusionData;
class SecretionDataFast;*/
    class FastDiffusionSolver2DSerializer;

    class BoxWatcher;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class FastDiffusionSolver2DFE;

    class ParallelUtilsOpenMP;

    class PDESOLVERS_EXPORT SecretionDataFast : public SecretionData {
    public:
        typedef void (FastDiffusionSolver2DFE::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT DiffusionSecretionFastFieldTupple {
    public:
        DiffusionData diffData;
        SecretionDataFast secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataFast *getSecretionData() { return &secrData; }
    };


    class PDESOLVERS_EXPORT FastDiffusionSolver2DFE : public DiffusableVector2D<float> {

        friend class FastDiffusionSolver2DSerializer;

    public :
        typedef void (FastDiffusionSolver2DFE::*diffSecrFcnPtr_t)(void);

        typedef void (FastDiffusionSolver2DFE::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef float precision_t;
        //typedef Array2DBorders<precision_t>::ContainerType Array2D_t;
        typedef Array2DContiguous <precision_t> ConcentrationField_t;

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
        float diffConst;
        float decayConst;
        float deltaX;///spacing
        float deltaT;///time interval
        float dt_dx2; ///ratio delta_t/delta_x^2
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;

/*   std::vector<DiffusionData> diffDataVec;
   std::vector<SecretionDataFast> secrDataVec;*/
        std::vector<bool> periodicBoundaryCheckVector;

        std::vector <BoundaryConditionSpecifier> bcSpecVec;
        std::vector<bool> bcSpecFlagVec;


        CellInventory *cellInventoryPtr;

        void (FastDiffusionSolver2DFE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (FastDiffusionSolver2DFE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
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

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        float couplingTerm(Point3D &_pt, std::vector <CouplingData> &_couplDataVec, float _currentConcentration);

        void initializeConcentration();


        bool serializeFlag;
        bool readFromFileFlag;
        unsigned int serializeFrequency;

        std::vector <DiffusionSecretionFastFieldTupple> diffSecrFieldTuppleVec;

        FastDiffusionSolver2DSerializer *serializerPtr;
        bool haveCouplingTerms;


        unsigned int index(unsigned int x, unsigned int y) {
            return workFieldDim.x * y + x;
        }


    public:

        FastDiffusionSolver2DFE();

        virtual ~FastDiffusionSolver2DFE();


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

    class PDESOLVERS_EXPORT FastDiffusionSolver2DSerializer : public Serializer {
    public:
        FastDiffusionSolver2DSerializer() : Serializer() {
            solverPtr = 0;
            serializedFileExtension = "dat";
            currentStep = 0;
        }

        ~FastDiffusionSolver2DSerializer() {}

        FastDiffusionSolver2DFE *solverPtr;

        virtual void serialize();

        virtual void readFromFile();

        void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

    protected:
        unsigned int currentStep;

    };


// class SecretionDataFast:public SecretionData{
//    public:
//       std::vector<FastDiffusionSolver2DFE::secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
// };



};


#endif
