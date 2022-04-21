

#ifndef PIFDUMPER_H
#define PIFDUMPER_H


#include <CompuCell3D/CC3D.h>


#include "PIFDumperDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class CellTypePlugin;

    class PIFDUMPER_EXPORT PIFDumper : public Steppable {
        Potts3D *potts;

        std::string pifname;
        int numDigits;
        std::string pifFileExtension;
        CellTypePlugin *typePlug;

    public:

        PIFDumper();

        PIFDumper(std::string);

        void setPotts(Potts3D *potts) { this->potts = potts; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep);

        virtual void finish() {}
        // End Steppable interface


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();
    };
};
#endif
