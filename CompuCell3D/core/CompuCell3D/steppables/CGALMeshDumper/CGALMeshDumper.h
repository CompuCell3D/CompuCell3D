
#ifndef CGALMESHDUMPERSTEPPABLE_H
#define CGALMESHDUMPERSTEPPABLE_H

#include <CompuCell3D/Steppable.h>

#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>

#include "CGALMeshDumperDLLSpecifier.h"

//STL containers
#include <vector>
#include <list>
#include <set>
#include <map>

namespace CompuCell3D {

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class CellInventory;

    class CellG;

    class CGALMESHDUMPER_EXPORT CGALMeshDumper : public Steppable {


        WatchableField3D<CellG *> *cellFieldG;
        Simulator *sim;
        Potts3D *potts;
        CC3DXMLElement *xmlData;
        Automaton *automaton;
        BoundaryStrategy *boundaryStrategy;
        CellInventory *cellInventoryPtr;

        Dim3D fieldDim;


    public:
        CGALMeshDumper();

        virtual ~CGALMeshDumper();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);


        //steppable interface
        virtual void start();

        virtual void step(const unsigned int currentStep);

        virtual void finish() {}


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif        
