#ifndef STEPPABLE_H
#define STEPPABLE_H

#include "SimObject.h"

namespace CompuCell3D {
    class Simulator;

    class Steppable : public SimObject {
    public:
        int frequency;

        Steppable() : frequency(1) {}

        virtual ~Steppable() {}

        virtual void start() {};

        virtual void step(const unsigned int currentStep) {};

        virtual void finish() {};

        virtual std::string toString() { return "Steppable"; }

    };
};
#endif
