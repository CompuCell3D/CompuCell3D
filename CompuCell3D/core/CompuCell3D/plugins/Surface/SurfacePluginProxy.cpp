#include "SurfacePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto surfaceProxy = registerPlugin<Plugin, SurfacePlugin>(
        "Surface",
        "Tracks cell surfaces and adds surface energy function.",
        &Simulator::pluginManager
);

auto surfaceFlexProxy = registerPlugin<Plugin, SurfacePlugin>(
        "SurfaceFlex",
        "Tracks cell surfaces and adds surface energy function.",
        &Simulator::pluginManager
);

auto surfaceLocalFlexProxy = registerPlugin<Plugin, SurfacePlugin>(
        "SurfaceLocalFlex",
        "Tracks cell surfaces and adds surface energy function.",
        &Simulator::pluginManager
);

auto surfaceEnergyProxy = registerPlugin<Plugin, SurfacePlugin>(
        "SurfaceEnergy",
        "Tracks cell surfaces and adds surface energy function.",
        &Simulator::pluginManager
);
