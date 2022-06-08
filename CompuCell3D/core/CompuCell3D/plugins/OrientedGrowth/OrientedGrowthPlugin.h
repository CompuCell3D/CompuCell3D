
#ifndef ORIENTEDGROWTHPLUGIN_H
#define ORIENTEDGROWTHPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "OrientedGrowthData.h"

#include "OrientedGrowthDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;

    class Automaton;

    //class AdhesionFlexData;
    class BoundaryStrategy;

    class ParallelUtilsOpenMP;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class ORIENTEDGROWTH_EXPORT  OrientedGrowthPlugin : public Plugin, public EnergyFunction, public Stepper {

    private:
        ExtraMembersGroupAccessor <OrientedGrowthData> orientedGrowthDataAccessor;
        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;

    public:

        OrientedGrowthPlugin();

        virtual ~OrientedGrowthPlugin();

        ExtraMembersGroupAccessor <OrientedGrowthData> *
        getOrientedGrowthDataAccessorPtr() { return &orientedGrowthDataAccessor; }


        //Energy function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        //my variables
        double xml_energy_penalty;
        double xml_energy_falloff;

        //access and set my variables
        virtual void setConstraintWidth(CellG *Cell, float _constraint);

        virtual void setElongationAxis(CellG *Cell, float _elongX, float _elongY);

        virtual void setElongationEnabled(CellG *Cell, bool _enabled);

        virtual float getElongationAxis_X(CellG *Cell);

        virtual float getElongationAxis_Y(CellG *Cell);

        virtual bool getElongationEnabled(CellG *Cell);

        // Stepper interface
        virtual void step();

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif
        
