

//Author: Margriet Palm CWI, Netherlands

#ifndef RANDOMFIELDINITIALIZER_H
#define RANDOMFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>
#include "FieldBuilder.h"
#include "RandomFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class RANDOMINITIALIZERS_EXPORT RandomFieldInitializer : public Steppable {
        void setParameters(Simulator *_simulator, CC3DXMLElement *_xmlData);

        Potts3D *potts;
        Simulator *simulator;
        RandomNumberGenerator *rand;
        WatchableField3D<CellG *> *cellField;
        FieldBuilder *builder;
        Dim3D dim, boxMin, boxMax;
        bool showStats;
        int ncells, growsteps, borderTypeID;

    public:
        RandomFieldInitializer();

        virtual ~RandomFieldInitializer();

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
