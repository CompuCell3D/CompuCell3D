#ifndef ADHESIONFLEXPLUGIN_H
#define ADHESIONFLEXPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "AdhesionFlexData.h"
#include "AdhesionFlexDLLSpecifier.h"


class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;

    class Automaton;

    class AdhesionFlexData;

    class BoundaryStrategy;

    class ParallelUtilsOpenMP;


    class ADHESIONFLEX_EXPORT  AdhesionFlexPlugin : public Plugin, public EnergyFunction {
    public:
        typedef double (AdhesionFlexPlugin::*adhesionFlexEnergyPtr_t)(const CellG *cell1, const CellG *cell2);


    private:
        ExtraMembersGroupAccessor <AdhesionFlexData> adhesionFlexDataAccessor;
        CC3DXMLElement *xmlData;
        Potts3D *potts;
        Simulator *sim;
        ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
        //Energy function data

        typedef std::map<int, double> bindingParameters_t;

        typedef std::vector <std::vector<double>> bindingParameterArray_t;

        std::map<std::string, unsigned int> mapCadNameToIndex;
        unsigned int numberOfCadherins;

        std::string contactFunctionType;
        std::string autoName;
        double depth;


        ExtraMembersGroupAccessor <AdhesionFlexData> *adhesionFlexDataAccessorPtr;

        Automaton *automaton;
        bool weightDistance;

        adhesionFlexEnergyPtr_t adhesionFlexEnergyPtr;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;

        bindingParameters_t bindingParameters;
        bindingParameterArray_t bindingParameterArray;
        int numberOfAdhesionMolecules;
        bool adhesionDensityInitialized;
        std::vector <std::string> adhesionMoleculeNameVec;
        std::map<std::string, int> moleculeNameIndexMap;
        std::map<int, std::vector<float> > typeToAdhesionMoleculeDensityMap;
        std::vector<float> adhesionMoleculeDensityVecMedium;

        //vectorized variables for convenient parallel access
        std::string formulaString; //expression for cad-molecule adhesion function
        std::vector<double> molecule1Vec; //used to keep arguments for molecule-molecule adhesion function
        std::vector<double> molecule2Vec;//used to keep arguments for cad-molecule adhesion function
        std::vector <mu::Parser> pVec;

        //default non existing density
        static const int errorDensity = -1000000;

        void initializeAdhesionMoleculeDensityVector();

    public:

        AdhesionFlexPlugin();

        virtual ~AdhesionFlexPlugin();

        ExtraMembersGroupAccessor <AdhesionFlexData> *
        getAdhesionFlexDataAccessorPtr() { return &adhesionFlexDataAccessor; }


        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);


        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);


        // Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        /**
        * @return The contact energy between cell1 and cell2.
        */
        double adhesionFlexEnergyCustom(const CellG *cell1, const CellG *cell2);

        void
        setBindingParameter(const std::string moleculeName1, const std::string moleculeName2, const double parameter,
                            bool parsing_flag = false);

        void setBindingParameterDirect(const std::string moleculeName1, const std::string moleculeName2,
                                       const double parameter);

        void setBindingParameterByIndexDirect(int _idx1, int _idx2, const double parameter);

        std::vector <std::vector<double>> getBindingParameterArray();

        std::vector <std::string> getAdhesionMoleculeNameVec();

        //functions used to manipulate densities of adhesion molecules
        void setAdhesionMoleculeDensity(CellG *_cell, std::string _moleculeName, float _density);

        void setAdhesionMoleculeDensityByIndex(CellG *_cell, int _idx, float _density);

        void setAdhesionMoleculeDensityVector(CellG *_cell, std::vector<float> _denVec);

        void assignNewAdhesionMoleculeDensityVector(CellG *_cell, std::vector<float> _denVec);

        //Medium functions
        void setMediumAdhesionMoleculeDensity(std::string _moleculeName, float _density);

        void setMediumAdhesionMoleculeDensityByIndex(int _idx, float _density);

        void setMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec);

        void assignNewMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec);


        float getAdhesionMoleculeDensity(CellG *_cell, std::string _moleculeName);

        float getAdhesionMoleculeDensityByIndex(CellG *_cell, int _idx);

        std::vector<float> getAdhesionMoleculeDensityVector(CellG *_cell);

        //Medium functions
        float getMediumAdhesionMoleculeDensity(std::string _moleculeName);

        float getMediumAdhesionMoleculeDensityByIndex(int _idx);

        std::vector<float> getMediumAdhesionMoleculeDensityVector();

        void overrideInitialization();


    protected:
        /**
        * @return The index used for ordering contact energies in the map.
        */
        int getIndex(const int type1, const int type2) const;


    };
};
#endif
