

#include "DictyChemotaxisSteppable.h"
#include "DictyFieldInitializer.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto dictyChemotaxisSteppableProxy = registerPlugin<Steppable, DictyChemotaxisSteppable>(
        "DictyChemotaxisSteppable",
        "Enables chemotaxis in cells by by simple tagging",
        &Simulator::steppableManager
);

auto dictyInitializerSteppableProxy = registerPlugin<Steppable, DictyFieldInitializer>(
        "DictyInitializer",
        "Initializes cell field for dictyostelim simulation",
        &Simulator::steppableManager
);
