#ifndef STEPPER_H
#define STEPPER_H

namespace CompuCell3D {

    class Stepper {
    public:
        virtual void step() = 0;
    };
};
#endif
