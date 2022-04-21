

#ifndef OBJINITIALIZER_H
#define OBJINITIALIZER_H

#include <CompuCell3D/CC3D.h>

#include "ObjInitializerDLLSpecifier.h"

namespace CompuCell3D {

    class Potts3D;

    class OBJINITIALIZER_EXPORT ObjInitializer :

            public Steppable {

        // don't change the name of the "potts" variable in this steppable,
        //   otherwise it can't be found by other (??? which ???) CC3D code:
        Potts3D *potts;

        std::string gObjFileName;

    public:

        ObjInitializer();

        ObjInitializer(std::string);

        void setPotts(Potts3D *potts) { this->potts = potts; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}
        // End Steppable interface

    }; // end of public Steppable

}; // end of namespace CompuCell3D

#endif // #ifndef OBJINITIALIZER_H
