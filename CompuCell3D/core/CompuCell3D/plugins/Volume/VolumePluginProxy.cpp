

#include "VolumePlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto volumeProxy = registerPlugin<Plugin, VolumePlugin>(
        "Volume",
        "Tracks cell volumes and adds volume energy function.",
        &Simulator::pluginManager
);

auto volumeFlexProxy = registerPlugin<Plugin, VolumePlugin>(
        "VolumeFlex",
        "Tracks cell volumes and adds volume energy function.",
        &Simulator::pluginManager
);

auto volumeLocalFlexProxy = registerPlugin<Plugin, VolumePlugin>(
        "VolumeLocalFlex",
        "Tracks cell volumes and adds volume energy function.",
        &Simulator::pluginManager
);

auto volumeEnergyProxy = registerPlugin<Plugin, VolumePlugin>(
        "VolumeEnergy",
        "Tracks cell volumes and adds volume energy function.",
        &Simulator::pluginManager
);
