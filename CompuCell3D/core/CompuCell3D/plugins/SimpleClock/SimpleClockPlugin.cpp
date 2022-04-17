

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
using namespace CompuCell3D;



using namespace std;


#include "SimpleClockPlugin.h"

SimpleClockPlugin::SimpleClockPlugin() : potts(0) {}

SimpleClockPlugin::~SimpleClockPlugin() {}

void SimpleClockPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
   
   potts = simulator->getPotts(); 
   potts->getCellFactoryGroupPtr()->registerClass(&simpleClockAccessor);
}

