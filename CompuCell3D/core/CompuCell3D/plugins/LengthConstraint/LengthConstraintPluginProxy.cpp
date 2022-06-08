

#include "LengthConstraintPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto lengthConstraintProxy = registerPlugin<Plugin, LengthConstraintPlugin>(
        "LengthConstraint",
        "Tracks cell lengths and adds length constraints. Each cell type may have different lambda and target lengths",
        &Simulator::pluginManager
);

auto lengthConstraintLocalFlexProxy = registerPlugin<Plugin, LengthConstraintPlugin>(
        "LengthConstraintLocalFlex",
        "Tracks cell lengths and adds length constraints. Each individual cell  may have different lambda and target lengths",
        &Simulator::pluginManager
);
