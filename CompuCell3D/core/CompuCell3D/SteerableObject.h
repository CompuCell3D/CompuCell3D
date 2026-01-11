

#ifndef STEERABLEOBJECT_H
#define STEERABLEOBJECT_H

#include <string>

class CC3DXMLElement;
namespace CompuCell3D {

    class ParseData;

    class SteerableObject {
    public:
        SteerableObject() {}

        virtual ~SteerableObject() {}

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false) {
            (void)_xmlData;
            (void)_fullInitFlag;
        }

        virtual std::string steerableName() { return "SteerableObject"; }
    };
}
#endif
