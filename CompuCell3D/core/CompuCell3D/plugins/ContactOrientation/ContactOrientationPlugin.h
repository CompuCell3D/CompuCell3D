
#ifndef CONTACTORIENTATIONPLUGIN_H
#define CONTACTORIENTATIONPLUGIN_H
#include <CompuCell3D/CC3D.h>
#include "ContactOrientationData.h"

// // // #include <CompuCell3D/Plugin.h>

// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>


// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>

// // // #include <muParser/muParser.h>

// // // // basic STL includes
// // // #include <vector>
// // // #include <list>
// // // #include <map>
// // // #include <set>
// // // #include <string>

#include "ContactOrientationDLLSpecifier.h"
// // // #include <muParser/ExpressionEvaluator/ExpressionEvaluator.h>

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;
    class Automaton;    
    class BoundaryStrategy;
    class ParallelUtilsOpenMP;
    
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class CONTACTORIENTATION_EXPORT  ContactOrientationPlugin : public Plugin ,public EnergyFunction  {
        
    private:    
        BasicClassAccessor<ContactOrientationData> contactOrientationDataAccessor;                
        CC3DXMLElement *xmlData;        
        
        Potts3D *potts;
        
        Simulator *sim;
        
        ParallelUtilsOpenMP *pUtils;            
        
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        
        
		ExpressionEvaluatorDepot eed;
		bool angularTermDefined;
        
        
        

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
    
        //contact energy part
        typedef std::map<int, double> contactEnergies_t;
        typedef std::vector<std::vector<double> > contactEnergyArray_t;

        contactEnergies_t contactEnergies;

        contactEnergyArray_t contactEnergyArray;

        double depth;        
        unsigned int maxNeighborIndex;
		Dim3D fieldDim;
        
        
        
        
    public:

        ContactOrientationPlugin();
        virtual ~ContactOrientationPlugin();
        
        BasicClassAccessor<ContactOrientationData> * getContactOrientationDataAccessorPtr(){return & contactOrientationDataAccessor;}                

        
        //Energy function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);        
        
                
        
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
        virtual void handleEvent(CC3DEvent & _event);

        virtual void extraInit(Simulator *simulator);

        //Accessors - Python Interface
        virtual void setOriantationVector(CellG * _cell,double _x, double _y, double _z);
        virtual Vector3 getOriantationVector(const CellG * _cell);
        virtual void setAlpha(CellG * _cell,double _alpha);
        virtual double getAlpha(const CellG * _cell);
		
        //Steerrable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName();
        virtual std::string toString();

        double contactEnergy(const CellG *cell1, const CellG *cell2);
        /**
        * Sets the contact energy for two cell types.  A -1 type is interpreted
        * as the medium.
        */
        void setContactEnergy(const std::string typeName1,const std::string typeName2, const double energy);        
            
    protected:
        /**
        * @return The index used for ordering contact energies in the map.
        */
        int getIndex(const int type1, const int type2) const; 
		double singleTermFormula(double _alpha,double _theta);
        double angularTermFunction(double _alpha,double _theta);
        
        typedef double ( ContactOrientationPlugin::*angularTerm_t)(double _alpha,double _theta);
        ContactOrientationPlugin::angularTerm_t angularTermFcnPtr;
    
    };
};
#endif
        
