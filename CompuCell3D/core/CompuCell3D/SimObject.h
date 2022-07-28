#ifndef SIMOBJECT_H
#define SIMOBJECT_H


#include <string>
#include <CompuCell3D/SteerableObject.h>

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class ParseData;

    class CC3DEvent;

    class SimObject : public virtual SteerableObject {
    protected:
        Simulator *simulator;
        ParseData *pd;
    public:
        SimObject() : simulator(0), pd(0) {}

        virtual ~SimObject() {}


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0) { this->simulator = simulator; }

        virtual void extraInit(Simulator *simulator) { this->simulator = simulator; }

        virtual std::string toString() { return "SimObject"; }

        virtual ParseData *getParseData() { return pd; }

        virtual void handleEvent(CC3DEvent &_event) {}

    };
}
#endif
