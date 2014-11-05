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

   std::map<unsigned char,std::pair<unsigned char, int> > movingTypeId2TailTypeIdMap;
   typedef std::map<unsigned char,std::pair<unsigned char, int> >::iterator mitr_t;





  public:
    CellTail();
    virtual ~CellTail();
    
	void init(CAManager *_caManager);
	void setMovingCellTrail(std::string _movingCellType, std::string _tailCellType,int _tailCellSize=1);

    // CACellStackFieldChangeWatcherinterface
    virtual void field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack);  
	
    virtual std::string toString();
	// virtual std::string steerableName();
  };
};
#endif
