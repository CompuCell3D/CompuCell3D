#ifndef BOXWATCHERSTEPPABLE_H
#define BOXWATCHERSTEPPABLE_H

#include <CompuCell3D/CC3D.h>

#include "BoxWatcherDLLSpecifier.h"

namespace CompuCell3D {


    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class Potts3D;

    class BOXWATCHER_EXPORT BoxWatcher : public Steppable {

        WatchableField3D<CellG *> *cellFieldG;
        Simulator *sim;
        Potts3D *potts;
        Dim3D fieldDim;

        Point3D minCoordinates;
        Point3D maxCoordinates;
        std::vector<unsigned char> frozenTypeVector;

        void adjustBox();

        void adjustCoordinates(Point3D pt);

        bool checkIfFrozen(unsigned char _type);

        unsigned int xMargin;
        unsigned int yMargin;
        unsigned int zMargin;

    public:
        BoxWatcher();

        virtual ~BoxWatcher();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);


        virtual void start();

        virtual void step(const unsigned int currentStep);

        virtual void finish() {}

        Point3D getMinCoordinates();

        Point3D getMaxCoordinates();

        Point3D *getMinCoordinatesPtr() { return &minCoordinates; }

        Point3D *getMaxCoordinatesPtr() { return &maxCoordinates; }

        Point3D getMargins();


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
