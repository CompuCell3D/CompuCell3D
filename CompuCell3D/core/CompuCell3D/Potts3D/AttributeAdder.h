#ifndef ATTRIBUTEADDER_H
#define ATTRIBUTEADDER_H

#include "CellInventoryWatcher.h"

namespace CompuCell3D {

    class CellG;
    class AttributeAdderWatcher;

    class AttributeAdder {

        AttributeAdderWatcher *cInvWatcher;

    public:
        AttributeAdder() : cInvWatcher{0} {}

        virtual ~AttributeAdder();

        AttributeAdderWatcher *getInventoryWatcher();

        virtual void addAttribute(CellG *) {};

        virtual void destroyAttribute(CellG *) {};

    };

    class AttributeAdderWatcher : public CellInventoryWatcher {

        AttributeAdder *attrAdder;

    public:

        AttributeAdderWatcher(AttributeAdder *_attrAdder) : attrAdder{_attrAdder} {};

        void onCellAdd(CellG *cell) {
            if(attrAdder) 
                attrAdder->addAttribute(cell);
        }

        void onCellRemove(CellG *cell) {
            if(attrAdder) 
                attrAdder->destroyAttribute(cell);
        }

    };

};

#endif
