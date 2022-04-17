#ifndef EVENTS_H
#define EVENTS_H

#include <CompuCell3D/Field3D/Dim3D.h>

namespace CompuCell3D {

    enum CC3DEvent_t {
        BASE, LATTICE_RESIZE, CHANGE_NUMBER_OF_WORK_NODES
    };


    class CC3DEvent {
    public:
        CC3DEvent() {
            id = BASE;
        }

        CC3DEvent_t id;

    };


    class CC3DEventLatticeResize : public CC3DEvent {
    public:
        CC3DEventLatticeResize() {
            id = LATTICE_RESIZE;
        }

        Dim3D newDim;
        Dim3D oldDim;
        Dim3D shiftVec;

    };

    class CC3DEventChangeNumberOfWorkNodes : public CC3DEvent {
    public:
        CC3DEventChangeNumberOfWorkNodes() {
            id = CHANGE_NUMBER_OF_WORK_NODES;
            oldNumberOfNodes = 1;
            newNumberOfNodes = 1;
        }

        int oldNumberOfNodes;
        int newNumberOfNodes;


    };


};
#endif