#ifndef ELASTICITYTRACKERPLUGIN_H
#define ELASTICITYTRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include "ElasticityTracker.h"

#include <CompuCell3D/Field3D/AdjacentNeighbor.h>


#include "ElasticityDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class CellInventory;
  class BoundaryStrategy;

class ELASTICITY_EXPORT ElasticityTrackerPlugin : public Plugin, public CellGChangeWatcher {

	  ParallelUtilsOpenMP *pUtils;
      ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
      
      WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      BasicClassAccessor<ElasticityTracker> elasticityTrackerAccessor;
      Simulator *simulator;
      CellInventory * cellInventoryPtr;
      bool initialized;

      unsigned int maxNeighborIndex;
      BoundaryStrategy *boundaryStrategy;
		CC3DXMLElement *xmlData;
		bool manualInit;
   public:
      ElasticityTrackerPlugin();
      virtual ~ElasticityTrackerPlugin();
      
      // SimObject interface
      virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
      virtual void extraInit(Simulator *simulator);

      // BCGChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
      
      BasicClassAccessor<ElasticityTracker> * getElasticityTrackerAccessorPtr(){return & elasticityTrackerAccessor;}
		//had to include this function to get set inereation working properly with Python , and Player that has restart capabilities 
      ElasticityTrackerData * getElasticityTrackerData(ElasticityTrackerData * _psd){return _psd;}
		ElasticityTrackerData * findTrackerData(CellG * _cell1 ,CellG * _cell2);
		void assignElasticityPair(CellG * _cell1 ,CellG * _cell2);
		void addNewElasticLink(CellG * _cell1 ,CellG * _cell2,float _lambdaElasticityLink=0.0,float _targetLinkLength=0.0);
		void removeElasticityPair(CellG * _cell1 ,CellG * _cell2);

		
		

      void initializeElasticityNeighborList();

   protected:

      
      std::set<std::string> elasticityTypesNames;
      std::set<unsigned char> elasticityTypes;

  };
};
#endif
