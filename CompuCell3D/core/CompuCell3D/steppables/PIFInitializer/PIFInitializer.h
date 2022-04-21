#ifndef PIFINITIALIZER_H
#define PIFINITIALIZER_H

#include <CompuCell3D/CC3D.h>
#include <string>


#include "PIFInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class Simulator;

    class PIFINITIALIZER_EXPORT PIFInitializer : public Steppable {
        Potts3D *potts;
        Simulator *sim;

        std::string pifname;

    public:


        PIFInitializer();

        PIFInitializer(std::string);

        void setPotts(Potts3D *potts) { this->potts = potts; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}
        // End Steppable interface


    };
};
#endif
