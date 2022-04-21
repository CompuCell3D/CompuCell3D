

#include "AdhesionFlexPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto adhesionFlexProxy = registerPlugin<Plugin, AdhesionFlexPlugin>("AdhesionFlex",
                                                                    "Contact energy function .Energy is calculated as a matrix product of cadherins conncentration with custom functional forms or entirely using custom functions",
                                                                    &Simulator::pluginManager
);
