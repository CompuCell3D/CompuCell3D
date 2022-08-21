
#ifndef ORIENTEDGROWTH2PLUGIN_H
#define ORIENTEDGROWTH2PLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "OrientedGrowth2Data.h"

#include "OrientedGrowth2DLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;
    class Automaton;
    //class AdhesionFlexData;
    class BoundaryStrategy;
    class ParallelUtilsOpenMP;
    
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class ORIENTEDGROWTH2_EXPORT OrientedGrowth2Plugin : public Plugin ,public EnergyFunction  ,public Stepper{
        
    private:    
        ExtraMembersGroupAccessor<OrientedGrowth2Data> orientedGrowth2DataAccessor;
        CC3DXMLElement *xmlData;        
        
        Potts3D *potts;
        
        Simulator *sim;
        
        ParallelUtilsOpenMP *pUtils;            
        
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
        
    public:

        OrientedGrowth2Plugin();
        virtual ~OrientedGrowth2Plugin();
        
        ExtraMembersGroupAccessor<OrientedGrowth2Data> * getOrientedGrowth2DataAccessorPtr(){return & orientedGrowth2DataAccessor;}

        
        //Energy function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
        
        //my variables
        double xml_energy_penalty;
        double xml_energy_falloff;
        
        //access and set my variables
        virtual void setConstraintWidth(CellG *Cell, float _constraint);
        virtual void setConstraintLength(CellG *Cell, float _constraint);
        virtual void setConstraintVolume(CellG *Cell, int _constraint);
        virtual void setApicalRadius(CellG *Cell, float _constraint);
        virtual void setBasalRadius(CellG *Cell, float _constraint);
        virtual void setElongationAxis(CellG *Cell, float _elongX, float _elongY, float _elongZ);
        virtual void setElongationCOM(CellG *Cell, float _elongXCOM, float _elongYCOM, float _elongZCOM);
        virtual void setElongationEnabled(CellG *Cell, bool _enabled);
        virtual void setConstrictionEnabled(CellG *Cell, bool _enabled);
        virtual void updateElongationAxis(CellG *ogCell);
    
        virtual float getConstraintWidth(CellG *Cell);
        virtual float getConstraintLength(CellG *Cell);
        virtual int getConstraintVolume(CellG *Cell);
        virtual float getApicalRadius(CellG *Cell);
        virtual float getBasalRadius(CellG *Cell);
        virtual float getElongationAxis_X(CellG *Cell);
        virtual float getElongationAxis_Y(CellG *Cell);
        virtual float getElongationAxis_Z(CellG *Cell);
        virtual float getElongationCOM_X(CellG *Cell);
        virtual float getElongationCOM_Y(CellG *Cell);
        virtual float getElongationCOM_Z(CellG *Cell);
        virtual bool getElongationEnabled(CellG *Cell);
        virtual bool getConstrictionEnabled(CellG *Cell);
        
        // Stepper interface
        virtual void step();        
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
        virtual void extraInit(Simulator *simulator);

        //Steerrable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName();
        virtual std::string toString();

    };
};
#endif
        
