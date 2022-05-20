#ifndef TYPETRANSITION
#define TYPETRANSITION

#include <CompuCell3D/Potts3D/Cell.h>
#include<vector>

namespace CompuCell3D {
    class TypeChangeWatcher;


    class TypeTransition {
    private:
        std::vector<TypeChangeWatcher *> typeChangeWatcherVec;

    public:
        TypeTransition() {}

        virtual ~TypeTransition() {}

        void registerTypeChangeWatcher(TypeChangeWatcher *_watcher);

        void setType(CellG *_cell, CellG::CellType_t _newType);


    };
};

#endif


