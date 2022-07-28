

#ifndef DICTYFIELDINITIALIZER_H
#define DICTYFIELDINITIALIZER_H

#include <CompuCell3D/CC3D.h>


#include "DictyDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class CellInventory;

    class CellG;

    class Automaton;

    template<typename T>
    class Field3D;

    template<typename T>
    class WatchableField3D;

    class DICTY_EXPORT DictyFieldInitializer : public Steppable {
        Simulator *sim;
        Potts3D *potts;
        Automaton *automaton;
        CellInventory *cellInventoryPtr;

        int gap;
        int width;
        Dim3D dim;
        WatchableField3D<CellG *> *cellField;

        Point3D zonePoint;
        unsigned int zoneWidth;

        unsigned int amoebaeFieldBorder;
        bool gotAmoebaeFieldBorder;
        CellG *groundCell;
        CellG *wallCell;
        float presporeRatio;

        bool belongToZone(Point3D com);

    public:

        DictyFieldInitializer();

        virtual ~DictyFieldInitializer() {};


        void setPotts(Potts3D *potts) { this->potts = potts; }

        void initializeCellTypes();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        //steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        // Begin Steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep) {}

        virtual void finish() {}
        // End Steppable interface

    };
};
#endif
