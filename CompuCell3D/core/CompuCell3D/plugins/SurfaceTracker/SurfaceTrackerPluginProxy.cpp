

#include "SurfaceTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto surfaceTrackerProxy = registerPlugin<Plugin, SurfaceTrackerPlugin>(
        "SurfaceTracker",
        "Tracks cell surfaces",
        &Simulator::pluginManager
);
