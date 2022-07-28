

#include "MitosisSteppable.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto mitosisSteppableProxy = registerPlugin<Steppable, MitosisSteppable>(
        "Mitosis",
        "Splits cells on demand with user selectable criteria. Called From Python",
        &Simulator::steppableManager
);
