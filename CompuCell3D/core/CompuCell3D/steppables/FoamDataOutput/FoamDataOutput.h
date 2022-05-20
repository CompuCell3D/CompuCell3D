#ifndef FOAMDATAOUTPUT_H
#define FOAMDATAOUTPUT_H

#include <CompuCell3D/CC3D.h>

#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>
#include "FoamDataOutputDLLSpecifier.h"
#include <string>

namespace CompuCell3D {
    class Potts3D;

    class CellInventory;

    class FOAMDATAOUTPUT_EXPORT FoamDataOutput : public Steppable {
        Potts3D *potts;
        CellInventory *cellInventoryPtr;
        Dim3D dim;
        ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr;
        std::string fileName;
        bool surFlag;
        bool volFlag;
        bool numNeighborsFlag;
        bool cellIDFlag;

    public:
        FoamDataOutput();

        virtual ~FoamDataOutput() {};

        void setPotts(Potts3D *potts) { this->potts = potts; }

        // SimObject interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual std::string toString();

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep);

        virtual void finish() {}
        // End Steppable interface
    };
};
#endif
