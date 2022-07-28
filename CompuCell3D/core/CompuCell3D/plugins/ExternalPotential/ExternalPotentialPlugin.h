#ifndef COMPUCELL3DEXTERNALPOTENTIALPLUGIN_H
#define COMPUCELL3DEXTERNALPOTENTIALPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "ExternalPotentialDLLSpecifier.h"

namespace CompuCell3D {

    /**
    @author m
    */

    class Simlator;

    class Potts3D;

    class AdjacentNeighbor;

    template<class T>
    class Field3D;

    class Potts3D;

    class BoundaryStrategy;

    class EXTERNALPOTENTIAL_EXPORT ExternalPotentialParam {
    public:
        ExternalPotentialParam() {
            lambdaVec.x = 0.0;
            lambdaVec.y = 0.0;
            lambdaVec.z = 0.0;
        }

        Coordinates3D<float> lambdaVec;
        std::string typeName;
    };

    class EXTERNALPOTENTIAL_EXPORT ExternalPotentialPlugin : public Plugin, public EnergyFunction {

    private:

        Potts3D *potts;
        AdjacentNeighbor adjNeighbor;
        CC3DXMLElement *xmlData;
        Point3D boundaryConditionIndicator;
        BoundaryStrategy *boundaryStrategy;

        //EnergyFunction data
        Coordinates3D<float> lambdaVec;

        AdjacentNeighbor *adjNeighbor_ptr;
        WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        enum FunctionType {
            GLOBAL = 0, BYCELLTYPE = 1, BYCELLID = 2
        };
        FunctionType functionType;
        std::unordered_map<unsigned char, ExternalPotentialParam> externalPotentialParamMap;

        typedef double (ExternalPotentialPlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,
                                                                  const CellG *oldCell);

        ExternalPotentialPlugin::changeEnergy_t changeEnergyFcnPtr;

        double changeEnergyGlobal(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellType(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellId(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        //COM based functions
        double changeEnergyGlobalCOMBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellTypeCOMBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellIdCOMBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        std::set<unsigned char> participatingTypes;

    public:
        ExternalPotentialPlugin();

        ~ExternalPotentialPlugin();

        //plugin interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *_simulator);

        //energyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        //steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        //energyFunction methods

        void initData();


    };

};

#endif
