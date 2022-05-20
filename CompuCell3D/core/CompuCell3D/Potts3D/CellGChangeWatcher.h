
#ifndef CELLGCHANGEWATCHER_H
#define CELLGCHANGEWATCHER_H

#include "../Field3D/Field3DChangeWatcher.h"
#include "Cell.h"

namespace CompuCell3D {

    class CellGChangeWatcher : public Field3DChangeWatcher<CellG *> {
    };

};
#endif
