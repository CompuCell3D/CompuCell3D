
#ifndef NEIGHBOURSURFACECONSTRAINTPLUGIN_H
#define NEIGHBOURSURFACECONSTRAINTPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "NeighbourSurfaceConstraintDLLSpecifier.h"

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

    class NEIGHBOURSURFACECONSTRAINT_EXPORT  NeighbourSurfaceConstraintPlugin : public Plugin ,public EnergyFunction  {
        
    private:    
                        
        CC3DXMLElement *xmlData;        
        
        Potts3D *potts;
        
        Simulator *sim;
        
        ParallelUtilsOpenMP *pUtils;            
        
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
        
    public:

        NeighbourSurfaceConstraintPlugin();
        virtual ~NeighbourSurfaceConstraintPlugin();
        
                        

        
        //Energy function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);        
        
                
        
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

        virtual void extraInit(Simulator *simulator);

        //Steerrable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName();
        virtual std::string toString();

    };
};
#endif
        
