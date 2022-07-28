#ifndef COMPUCELL3DSCALABLEFLEXIBLEDIFFUSIONSOLVERFE_H
#define COMPUCELL3DSCALABLEFLEXIBLEDIFFUSIONSOLVERFE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

#include "DiffusableVectorContiguous.h"

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

    class BoxWatcher;

    class DiffusionData;

    class SecretionDataScalableFlex;

    class ScalableFlexibleDiffusionSolverSerializer;

    class TestScalableFlexibleDiffusionSolver; // Testing ScalableFlexibleDiffusionSolverFE
    class ParallelUtilsOpenMP;

    class CellTypeMonitorPlugin;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;


    class ScalableFlexibleDiffusionSolverFE;

    class PDESOLVERS_EXPORT SecretionDataScalableFlex

    :public SecretionData {
    public:

    typedef void (ScalableFlexibleDiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);

    std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
};

class PDESOLVERS_EXPORT DiffusionSecretionScalableFlexFieldTupple{
        public:
        DiffusionData diffData;
        SecretionDataScalableFlex secrData;
        DiffusionData * getDiffusionData(){ return &diffData; }
        SecretionDataScalableFlex * getSecretionData(){ return &secrData; }
};


class PDESOLVERS_EXPORT ScalableFlexibleDiffusionSolverFE

:public DiffusableVectorContiguous<float>
{

friend class ScalableFlexibleDiffusionSolverSerializer;

// For Testing
friend class TestScalableFlexibleDiffusionSolver; // In production version you need to enclose with #ifdef #endif

public :

typedef void (ScalableFlexibleDiffusionSolverFE::*diffSecrFcnPtr_t)(void);

typedef void (ScalableFlexibleDiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);

typedef float precision_t;
//typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
typedef Array3DContiguous <precision_t> ConcentrationField_t;

BoxWatcher *boxWatcherSteppable;

float diffusionLatticeScalingFactor; // for hex in 2Dlattice it is 2/3.0 , for 3D is 1/2.0, for cartesian lattice it is 1
bool autoscaleDiffusion;

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

//    std::vector<DiffusionData> diffDataVec;
//    std::vector<SecretionDataScalableFlex> secrDataVec;
std::vector<bool> periodicBoundaryCheckVector;

std::vector <BoundaryConditionSpecifier> bcSpecVec;
std::vector<bool> bcSpecFlagVec;


CellInventory *cellInventoryPtr;

void (ScalableFlexibleDiffusionSolverFE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
void (ScalableFlexibleDiffusionSolverFE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
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

ScalableFlexibleDiffusionSolverSerializer *serializerPtr;
bool haveCouplingTerms;
std::vector <DiffusionSecretionScalableFlexFieldTupple> diffSecrFieldTuppleVec;
//vector<string> concentrationFieldNameVectorTmp;

vector<int> scalingExtraMCSVec;
std::vector<float> diffConstVec;
std::vector<float> decayConstVec;

CellTypeMonitorPlugin *cellTypeMonitorPlugin;
Array3DCUDA<unsigned char> *h_celltype_field;

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


public:

ScalableFlexibleDiffusionSolverFE();

virtual ~

ScalableFlexibleDiffusionSolverFE();


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

class PDESOLVERS_EXPORT ScalableFlexibleDiffusionSolverSerializer

: public Serializer{
public:

ScalableFlexibleDiffusionSolverSerializer() : Serializer() {
    solverPtr = 0;
    serializedFileExtension = "dat";
    currentStep = 0;
}

~

ScalableFlexibleDiffusionSolverSerializer() {}

ScalableFlexibleDiffusionSolverFE *solverPtr;

virtual void serialize();

virtual void readFromFile();

void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

protected:
unsigned int currentStep;

};





};


#endif
