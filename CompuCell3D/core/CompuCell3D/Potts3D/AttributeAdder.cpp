#include "AttributeAdder.h"

using namespace CompuCell3D;

AttributeAdder::~AttributeAdder() {
    if(cInvWatcher) {
        delete cInvWatcher;
        cInvWatcher = 0;
    }
}

AttributeAdderWatcher *AttributeAdder::getInventoryWatcher() {
    if(!cInvWatcher) 
        cInvWatcher = new AttributeAdderWatcher(this);
    return cInvWatcher;
}
