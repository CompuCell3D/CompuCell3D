

#ifndef DICTYCHEMOTAXISSTEPPABLE_H
#define DICTYCHEMOTAXISSTEPPABLE_H

#include <CompuCell3D/CC3D.h>

#include "DictyDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class CellInventory;


    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;


    template<class T>
    class Field3DImpl;


    class CellG;

    class SimpleClock;

    class DICTY_EXPORT DictyChemotaxisSteppable : public Steppable {
        Potts3D *potts;
        Field3D<float> *field;

        WatchableField3D<CellG *> *cellFieldG;

        Dim3D fieldDim;
        std::string chemicalFieldSource;
        std::string chemicalFieldName;

        CellInventory *cellInventoryPtr;
        ExtraMembersGroupAccessor <SimpleClock> *simpleClockAccessorPtr;

        unsigned int clockReloadValue;
        unsigned int chemotactUntil;
        float chetmotaxisActivationThreshold;
        unsigned int ignoreFirstSteps;
        int chemotactingCellsCounter;

    public:


        DictyChemotaxisSteppable();

        virtual ~DictyChemotaxisSteppable() {};

        // SimObject interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *_simulator);

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
