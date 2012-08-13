#ifndef MOLECULARCONTACTTRACKERPLUGIN_H
#define MOLECULARCONTACTTRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "MolecularContactTracker.h"
#include <CompuCell3D/Field3D/AdjacentNeighbor.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>


//#include <CompuCell3D/dllDeclarationSpecifier.h>
#include "MolecularContactDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class CellInventory;
  class BoundaryStrategy;

class MOLECULARCONTACT_EXPORT MolecularContactTrackerPlugin : public Plugin, public CellGChangeWatcher {

      WatchableField3D<CellG *> *cellFieldG;
      Dim3D fieldDim;
      BasicClassAccessor<MolecularContactTracker> molecularcontactTrackerAccessor;
      BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
      
      Simulator *simulator;
      CellInventory * cellInventoryPtr;
      bool initialized;

      unsigned int maxNeighborIndex;
      BoundaryStrategy *boundaryStrategy;
		CC3DXMLElement *xmlData;
   public:
      MolecularContactTrackerPlugin();
      virtual ~MolecularContactTrackerPlugin();
      
      // SimObject interface
      virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
      virtual void extraInit(Simulator *simulator);

      // BCGChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
      
      BasicClassAccessor<MolecularContactTracker> * getMolecularContactTrackerAccessorPtr(){return & molecularcontactTrackerAccessor;}
		//had to include this function to get set inereation working properly with Python , and Player that has restart capabilities 
      MolecularContactTrackerData * getMolecularContactTrackerData(MolecularContactTrackerData * _psd){return _psd;}
      void initializeMolecularContactNeighborList();
      void addMolecularContactNeighborList();

   protected:

      
      std::set<std::string> molecularcontactTypesNames;
      std::set<unsigned char> molecularcontactTypes;

  };
};
#endif
