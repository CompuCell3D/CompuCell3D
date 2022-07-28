#ifndef COMPUCELL3DADVECTIONDIFFUSIONSOLVERFE_H
#define COMPUCELL3DADVECTIONDIFFUSIONSOLVERFE_H


#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableGraph.h"
#include "DiffSecrData.h"

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

    class DiffusionData;

    class SecretionDataFlexAD;

    template<typename Y>
    class Field3D;

    template<typename Y>
    class Field3DImpl;

    template<typename Y>
    class WatchableField3D;

    class NeighborTracker;

    class AdvectionDiffusionSolverFE;

    class PDESOLVERS_EXPORT SecretionDataFlexAD : public SecretionData {
    public:
        typedef void (AdvectionDiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);

        std::vector <secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    };

    class PDESOLVERS_EXPORT DiffusionSecretionADFieldTupple {
    public:
        DiffusionData diffData;
        SecretionDataFlexAD secrData;

        DiffusionData *getDiffusionData() { return &diffData; }

        SecretionDataFlexAD *getSecretionData() { return &secrData; }
    };


    class PDESOLVERS_EXPORT AdvectionDiffusionSolverFE : public DiffusableGraph<float> {
    public :
        typedef void (AdvectionDiffusionSolverFE::*diffSecrFcnPtr_t)(void);

        typedef void (AdvectionDiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);

        typedef float precision_t;
        typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
        typedef Array3DBordersField3DAdapter <precision_t> ConcentrationField_t;

    protected:


        Potts3D *potts;
        Simulator *simPtr;

        unsigned int currentStep;
        unsigned int maxDiffusionZ;
        float averageRadius;

        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;

        std::vector <DiffusionData> diffDataVec;
        std::vector <SecretionDataFlexAD> secrDataVec;


        CellInventory *cellInventoryPtr;

        void (AdvectionDiffusionSolverFE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void (AdvectionDiffusionSolverFE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver
        void diffuse();

        void diffuseSingleField(unsigned int idx);

        void secrete();

        //void secreteOnContact();
        void secreteSingleField(unsigned int idx);

        void secreteOnContactSingleField(unsigned int idx);

        void
        scrarch2Concentration(std::map<CellG *, float> *scratchField, std::map<CellG *, float> *concentrationField);

        void cellMap2Field(std::map<CellG *, float> *concentrationMapField, ConcentrationField_t *concentrationField);

        void field2CellMap(ConcentrationField_t *concentrationField, std::map<CellG *, float> *concentrationMapField);

        //     void outputField( std::ostream & _out,Field3DImpl<float> *_concentrationField);
        void readConcentrationField(std::string fileName, ConcentrationField_t *concentrationField);

        unsigned int numberOfFields;
        Dim3D fieldDim;
        Dim3D workFieldDim;

        std::vector <DiffusionSecretionADFieldTupple> diffSecrFieldTuppleVec;

        void initializeConcentration();

        double computeAverageCellRadius();


        void updateCellInventories();

        void updateLocalCellInventory(unsigned int idx);

        ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr;

    public:
        AdvectionDiffusionSolverFE();

        virtual ~AdvectionDiffusionSolverFE();


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

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

    // class SecretionDataFlexAD:public SecretionData{
    //    public:
    //       std::vector<AdvectionDiffusionSolverFE::secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
    // };


};


#endif
