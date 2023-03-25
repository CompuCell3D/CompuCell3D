

//Author: Margriet Palm CWI, Netherlands

#ifndef RANDOMBLOBINITIALIZER_H
#define RANDOMBLOBINITIALIZER_H

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/steppables/Mitosis/MitosisSteppable.h>

#include "FieldBuilder.h"

#include "RandomFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class RANDOMINITIALIZERS_EXPORT RandomBlobInitializer : public Steppable {
        void setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData);

        void divide();

        MitosisSteppable *mit;
        Potts3D *potts;
        Simulator *simulator;
        RandomNumberGenerator *rand;
        ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr;
        WatchableField3D<CellG *> *cellField;
        FieldBuilder *builder;
        CellInventory *cellInventoryPtr;
        Dim3D dim, boxMin, boxMax, blobsize, blobpos;
        bool showStats;
        int ndiv, growsteps, borderTypeID;


    public:
        RandomBlobInitializer();

        virtual ~RandomBlobInitializer();

        void setPotts(Potts3D *potts) { this->potts = potts; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}
        // End Steppable interface

        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
