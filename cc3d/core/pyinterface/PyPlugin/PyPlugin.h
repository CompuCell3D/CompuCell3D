#ifndef PYPLUGIN_H
#define PYPLUGIN_H

#include <CompuCell3D/Plugin.h>

namespace CompuCell3D{
   class PyPlugin: public Plugin{
    virtual void init(Simulator *simulator) {}
    virtual void extraInit(Simulator *simulator){}
    virtual std::string toString(){return "PyPlugin";}
    virtual void readXML(XMLPullParser &xmlIn){}
    virtual void writeXML(XMLSerializer &xmlOut){}
   };

};

#endif