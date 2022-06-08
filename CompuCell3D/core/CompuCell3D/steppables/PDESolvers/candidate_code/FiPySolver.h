#ifndef COMPUCELL3DFIPYSOLVER_H
#define COMPUCELL3DFIPYSOLVER_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "FiPyContiguous.h"

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

    class SecretionDataFiPy;

    class FiPySolverSerializer;

    class TestFiPySolver; // Testing FiPySolver
    class ParallelUtilsOpenMP;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;


    class FiPySolver;

    class PDESOLVERS_EXPORT SecretionDataFiPy

    :public SecretionData {
    public:

    typedef void (FiPySolver::*secrSingleFieldFcnPtr_t)(unsigned int);

    std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
};

class PDESOLVERS_EXPORT DiffusionSecretionFiPyFieldTupple{
        public:
        DiffusionData diffData;
        SecretionDataFiPy secrData;
        DiffusionData * getDiffusionData(){ return &diffData; }
        SecretionDataFiPy * getSecretionData(){ return &secrData; }
};


class PDESOLVERS_EXPORT FiPySolver

:public FiPyContiguous<float>
{

friend class FiPySolverSerializer;

// For Testing
friend class TestFiPySolver; // In production version you need to enclose with #ifdef #endif

public :

typedef void (FiPySolver::*diffSecrFcnPtr_t)(void);

typedef void (FiPySolver::*secrSingleFieldFcnPtr_t)(unsigned int);

typedef float precision_t;
typedef Array3DFiPy <precision_t> ConcentrationField_t;

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

//    std::vector<DiffusionData> diffDataVec;
//    std::vector<SecretionDataFiPy> secrDataVec;
std::vector<bool> periodicBoundaryCheckVector;

std::vector <BoundaryConditionSpecifier> bcSpecVec;
std::vector<bool> bcSpecFlagVec;


CellInventory *cellInventoryPtr;

void (FiPySolver::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
void (FiPySolver::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
void secrete();

void FindDoNotDiffusePixels(unsigned int idx);

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

FiPySolverSerializer *serializerPtr;
bool haveCouplingTerms;
std::vector <DiffusionSecretionFiPyFieldTupple> diffSecrFieldTuppleVec;
//vector<string> concentrationFieldNameVectorTmp;

public:

FiPySolver();

virtual ~

FiPySolver();


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

class PDESOLVERS_EXPORT FiPySolverSerializer

: public Serializer{
public:

FiPySolverSerializer() : Serializer() {
    solverPtr = 0;
    serializedFileExtension = "dat";
    currentStep = 0;
}

~

FiPySolverSerializer() {}

FiPySolver *solverPtr;

virtual void serialize();

virtual void readFromFile();

void setCurrentStep(unsigned int _currentStep) { currentStep = _currentStep; }

protected:
unsigned int currentStep;

};





};


#endif
