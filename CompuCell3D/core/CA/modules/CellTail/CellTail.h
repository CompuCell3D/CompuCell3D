#ifndef CELLTAIL_H
#define CELLTAIL_H

 
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CA/CACellStackFieldChangeWatcher.h>



#define roundf(a) ((fmod(a,1)<0.5)?floor(a):ceil(a))


#include "CellTailDLLSpecifier.h"

namespace CompuCell3D {
  class CACell;
  class CAManager;
  class BoundaryStrategy;  

  template<typename T>
  class Field3D;  


  class CELLTAIL_EXPORT CellTail : public CACellStackFieldChangeWatcher {
    
    
  private:
   Point3D boundaryConditionIndicator;
   Dim3D fieldDim;
   BoundaryStrategy *boundaryStrategy;
   CAManager *caManager;
   Field3D<CACell *> * cellField;





  public:
    CellTail();
    virtual ~CellTail();
    
	void init(CAManager *_caManager);

    // CACellStackFieldChangeWatcherinterface
    virtual void field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack);  
	
    virtual std::string toString();
	// virtual std::string steerableName();
  };
};
#endif
