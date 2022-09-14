#ifndef FIXEDSTEPPER_H
#define FIXEDSTEPPER_H

namespace CompuCell3D {

    class FixedStepper {
    public:
        virtual void step() = 0;
    };
};
#endif
