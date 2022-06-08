
#ifndef CURVATUREPLUGIN_H
#define CURVATUREPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "CurvatureTracker.h"

#include "CurvatureDLLSpecifier.h"

class CC3DXMLElement;

//Note: Target distance is set to be 0.9 * current distance between cells. target distance from xml is ignored

namespace CompuCell3D {
    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class ParallelUtilsOpenMP;

    class CURVATURE_EXPORT CurvaturePlugin : public Plugin, public EnergyFunction, public CellGChangeWatcher {

        ParallelUtilsOpenMP *pUtils;


        ExtraMembersGroupAccessor <CurvatureTracker> curvatureTrackerAccessor;

        Potts3D *potts;


        std::string autoName;
        double depth;

        Automaton *automaton;
        bool weightDistance;
        unsigned int maxNeighborIndex;
        unsigned int maxNeighborIndexJunctionMove;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;

        std::set <std::string> plasticityTypesNames;
        std::set<unsigned char> plasticityTypes;
        std::set<unsigned char> internalPlasticityTypes;

        Dim3D fieldDim;
        double lambda;

        double activationEnergy;
        double targetDistance;
        double maxDistance;

        double potentialFunction(double _lambda, double _offset, double _targetDistance, double _distance);


        //vectorized variables for convenient parallel access
        std::vector<short> newJunctionInitiatedFlagWithinClusterVec;
        std::vector<CellG *> newNeighborVec;

        unsigned int maxNumberOfJunctions;


        enum FunctionType {
            GLOBAL = 0, BYCELLTYPE = 1, BYCELLID = 2
        };

        FunctionType functionType;

        typedef double (CurvaturePlugin::*diffEnergyFcnPtr_t)(float _deltaL, float _lBefore,
                                                              const CurvatureTrackerData *_curvatureTrackerData,
                                                              const CellG *_cell);

        diffEnergyFcnPtr_t diffEnergyFcnPtr;


        double diffEnergyLocal(float _deltaL, float _lBefore, const CurvatureTrackerData *_curvatureTrackerData,
                               const CellG *_cell);

        double diffEnergyGlobal(float _deltaL, float _lBefore, const CurvatureTrackerData *_curvatureTrackerData,
                                const CellG *_cell);

        double diffEnergyByType(float _deltaL, float _lBefore, const CurvatureTrackerData *_curvatureTrackerData,
                                const CellG *_cell);


        double tryAddingNewJunction(const Point3D &pt, const CellG *newCell);

        double tryAddingNewJunctionWithinCluster(const Point3D &pt, const CellG *newCell);

        double
        calculateInverseCurvatureSquare(const Vector3 &_leftVec, const Vector3 &_middleVec, const Vector3 &_rightVec);

        typedef std::map<int, CurvatureTrackerData> curvatureParams_t;


        // plastParams_t plastParams;
        curvatureParams_t internalCurvatureParams;

        // plastParams_t typeSpecificPlastParams;
        curvatureParams_t internalTypeSpecificCurvatureParams;


        typedef std::unordered_map<unsigned char, std::unordered_map<unsigned char, CurvatureTrackerData> > CurvatureTrackerDataArray_t;
        typedef std::unordered_map<unsigned char, CurvatureTrackerData> CurvatureTrackerDataMap_t;

        // CurvatureTrackerDataArray_t plastParamsArray;
        CurvatureTrackerDataArray_t internalCurvatureParamsArray;

        CurvatureTrackerDataMap_t internalTypeSpecificCurvatureParamsMap;

    public:
        CurvaturePlugin();

        virtual ~CurvaturePlugin();


        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);

        //EnergyFunction Interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        // Field3DChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);


        //used to manually control parameters plasticity term for pair of cells involved
        void setPlasticityParameters(CellG *_cell1, CellG *_cell2, double _lambda, double targetDistance = 0.0);

        double getPlasticityParametersLambdaDistance(CellG *_cell1, CellG *_cell2);

        double getPlasticityParametersTargetDistance(CellG *_cell1, CellG *_cell2);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    protected:
        int getIndex(const int type1, const int type2) const;

    };
};
#endif
