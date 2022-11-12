


#define CompuCellLibShared_EXPORTS
// if you dont define this DLL import/export macro
// from CompuCellLib you will get error "definition of dllimport static data member not allowed"
//this is because you define static members in the Simulator class and without
// this macro they will be redefined here as import symbols which is not allowed

#include "Simulator.h"

using namespace CompuCell3D;

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

#include <stdlib.h>

#include <Logger/CC3DLogger.h>
////the reason to declare BoundaryStrategy* BoundaryStrategy::singleton; here is because 
////Simulator.h includes Potts.h which includes WatchableField3D.h which includes Field3Dimpl.h which includes BoundaryStrategy.h
////BoundaryStrategy* BoundaryStrategy::singleton;
//
//
PluginManager<Plugin> Simulator::pluginManager;
PluginManager<Steppable> Simulator::steppableManager;
PluginManager<PluginBase> Simulator::pluginBaseManager;

void Syntax(const string name) {
    CC3D_Log(LOG_DEBUG) << "Syntax: " << name << " <config>";
    exit(1);
}

int main(int argc, char *argv[]) {

    return 1;
}
