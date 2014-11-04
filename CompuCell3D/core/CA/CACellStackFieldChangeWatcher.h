#ifndef CACELLFIELDSTACKCHANGEWATCHER_H
#define CACELLFIELDSTACKCHANGEWATCHER_H

// #include <CompuCell3D/Field3D/Field3DChangeWatcher.h>
// #include "CACell.h"

namespace CompuCell3D {
  class CACell;
  class CACellStack;
  
  class CACellStackFieldChangeWatcher {
    public:
    virtual void field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){} 
  };

};
#endif
