#ifndef TYPECHANGEWATCHER
#define TYPECHANGEWATCHER

#include <CompuCell3D/Potts3D/Cell.h>

namespace CompuCell3D {

    class TypeChangeWatcher {

    public:
        TypeChangeWatcher() {}

        virtual ~TypeChangeWatcher() {}

        virtual void typeChange(CellG *_cell, CellG::CellType_t _newType) = 0;


    };
};
#endif

