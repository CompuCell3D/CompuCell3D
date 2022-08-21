#ifndef CELLCHANGEWATCHER_H
#define CELLCHANGEWATCHER_H

#include "../Field3D/Field3DChangeWatcher.h"
#include "Cell.h"

namespace CompuCell3D {

    class CellChangeWatcher : public Field3DChangeWatcher<Cell *> {
    };
};
#endif
