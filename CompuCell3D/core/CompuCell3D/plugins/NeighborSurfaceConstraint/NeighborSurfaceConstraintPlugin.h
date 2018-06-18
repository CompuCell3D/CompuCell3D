
#ifndef NEIGHBORSURFACECONSTRAINTPLUGIN_H
#define NEIGHBORSURFACECONSTRAINTPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "NeighborSurfaceConstraintDLLSpecifier.h"

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

    class NEIGHBORSURFACECONSTRAINT_EXPORT  NeighborSurfaceConstraintPlugin : public Plugin ,public EnergyFunction  {
        
    private:    

        CC3DXMLElement *xmlData;        
        
        Potts3D *potts;
        
        Simulator *sim;
        
        ParallelUtilsOpenMP *pUtils;            
        
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;

        WatchableField3D<CellG *> *cellFieldG;
        //
        bool energyExpressionDefined;

        unsigned int maxNeighborIndex;

		LatticeMultiplicativeFactors lmf;

		//Energy function data:
		double neighborTargetSurface;

		double neighborLambdaSurface;

		/* The above are place holders, they will be changed once it becomes a pair energy
		 * something like this (found in the contact local flex):
		 * typedef std::map<int, double> contactEnergies_t;
		 * typedef std::vector<std::vector<double> > contactEnergyArray_t;
		 */
		double scaleSurface;

		// for the user defined energy
		std::vector<NeighborSurfaceEnergyParam> neighborSurfaceEnergyParamVector;

		typedef double (NeighborSurfaceConstraintPlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

		NeighborSurfaceConstraintPlugin::changeEnergy_t changeEnergyFcnPtr;

		double changeEnergyGlobal(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double changeEnergyByCellType(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double changeEnergyByCellId(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

		std::pair<double,double> getNewOldSurfaceDiffs(const Point3D &pt, const CellG *newCell,const CellG *oldCell);
		double energyChange(double lambda, double targetSurface,double surface,  double diff);

    public:

        NeighborSurfaceConstraintPlugin();
        virtual ~NeighborSurfaceConstraintPlugin();
        
                        

        
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
        
