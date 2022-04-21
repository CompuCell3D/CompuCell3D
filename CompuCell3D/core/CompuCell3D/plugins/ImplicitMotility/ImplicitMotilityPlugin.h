/*
@author jfg
*/
#ifndef IMPLICITMOTILITYPLUGIN_H
#define IMPLICITMOTILITYPLUGIN_H


#include <CompuCell3D/CC3D.h>
#include "ImplicitMotilityDLLSpecifier.h"


class CC3DXMLElement;


namespace CompuCell3D {

    class Simulator;


    class Potts3D;

    class Automaton;

    class AdjacentNeighbor;

    class BoundaryStrategy;

    class ParallelUtilsOpenMP;


    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class IMPLICITMOTILITY_EXPORT ImplicitMotilityParam {
    public:
        ImplicitMotilityParam() {

            lambdaMotility = 0.0;
        }

        double lambdaMotility;
        std::string typeName;
    };


    class IMPLICITMOTILITY_EXPORT  ImplicitMotilityPlugin : public Plugin, public EnergyFunction {


    private:


        CC3DXMLElement *xmlData;


        Potts3D *potts;

        AdjacentNeighbor adjNeighbor;


        Simulator *sim;

        Point3D boundaryConditionIndicator;

        ParallelUtilsOpenMP *pUtils;


        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;


        Automaton *automaton;


        BoundaryStrategy *boundaryStrategy;

        WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        AdjacentNeighbor *adjNeighbor_ptr;

        std::unordered_map<unsigned char, ImplicitMotilityParam> motilityParamMap;
        Coordinates3D<double> biasVecTmp;


        enum FunctionType {
            BYCELLTYPE = 0, BYCELLID = 1
        };
        FunctionType functionType;


    public:


        ImplicitMotilityPlugin();

        virtual ~ImplicitMotilityPlugin();


        //Energy function interface
        typedef double (ImplicitMotilityPlugin::*changeEnergy_t)(const Point3D &pt,
                                                                 const CellG *newCell, const CellG *oldCell);

        ImplicitMotilityPlugin::changeEnergy_t changeEnergyFcnPtr;

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellType(const Point3D &pt,
                                      const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellId(const Point3D &pt,
                                    const CellG *newCell, const CellG *oldCell);


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);


        virtual void extraInit(Simulator *simulator);

        //Steerable interface


        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };

};

#endif

        

