#ifndef CELLTRAIL_H
#define CELLTRAIL_H

 
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CA/SimulationObject.h>
#include <CA/CACellStackFieldChangeWatcher.h>



#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "CellTrailDLLSpecifier.h"

namespace CompuCell3D {
  class CACell;
  class CAManager;
  class BoundaryStrategy;  

  template<typename T>
  class Field3D;  


  class CELLTRAIL_EXPORT CellTrail : public SimulationObject, public CACellStackFieldChangeWatcher {
    
    
  private:
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   BoundaryStrategy *boundaryStrategy;
   CAManager *caManager;
   Field3D<CACell *> * cellField;

   std::map<unsigned char,std::pair<unsigned char, int> > movingTypeId2TrailTypeIdMap;
   typedef std::map<unsigned char,std::pair<unsigned char, int> >::iterator mitr_t;





  public:
    CellTrail();
    virtual ~CellTrail();
    
	virtual void init(CAManager *_caManager);		        
	virtual void extraInit();
    virtual std::string toString();		

	void _addMovingCellTrail(std::string _movingCellType, std::string _trailCellType,int _trailCellSize=1);

    // CACellStackFieldChangeWatcherinterface
    virtual void field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack);  
	

	// virtual std::string steerableName();
  };
};
#endif
