
#ifndef SURFACEPLUGIN_H
#define SURFACEPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "SurfaceDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class CellG;

    class BoundaryStrategy;


    class SURFACE_EXPORT SurfaceEnergyParam {
    public:
        SurfaceEnergyParam() : targetSurface(0.0), lambdaSurface(0.0) {}

        double targetSurface;
        double lambdaSurface;
        std::string typeName;
    };

    class SURFACE_EXPORT SurfacePlugin : public Plugin, public EnergyFunction {

        Potts3D *potts;

        CC3DXMLElement *xmlData{};
        ParallelUtilsOpenMP *pUtils;
        ExpressionEvaluatorDepot eed;
        bool energyExpressionDefined;


        std::string pluginName;

        BoundaryStrategy *boundaryStrategy;
        unsigned int maxNeighborIndex{};
        LatticeMultiplicativeFactors lmf;
        WatchableField3D<CellG *> *cellFieldG{};


        enum FunctionType {
            GLOBAL = 0, BYCELLTYPE = 1, BYCELLID = 2
        };
        FunctionType functionType;

        double targetSurface{};
        double lambdaSurface{};

        double scaleSurface;


        std::unordered_map<unsigned char, SurfaceEnergyParam> surfaceEnergyParamMap;

        typedef double (SurfacePlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        SurfacePlugin::changeEnergy_t changeEnergyFcnPtr{};

        double changeEnergyGlobal(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellType(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellId(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        std::pair<double, double> getNewOldSurfaceDiffs(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double diffEnergy(double lambda, double targetSurface, double surface, double diff);


    public:
        SurfacePlugin() : potts(0), energyExpressionDefined(false), pUtils(0), pluginName("Surface"), scaleSurface(1.0),
                          boundaryStrategy(0) {};

        virtual ~SurfacePlugin();



        // SimObject interface

        virtual void extraInit(Simulator *simulator);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual void handleEvent(CC3DEvent &_event);

        //EnergyFunction interface

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


        virtual std::string steerableName();

        virtual std::string toString();
    };
};
#endif
