from cc3d.core.PySteppables import SteppableBasePy


class RuntimeDiscoverySteppable(SteppableBasePy):
    def start(self):
        shared = self.shared_steppable_vars
        shared['python_plugin_cell_type_loaded'] = bool(
            self.simulator.pluginManager.isLoaded('CellType')
        )
        shared['python_plugin_missing_loaded'] = bool(
            self.simulator.pluginManager.isLoaded('DefinitelyMissingPlugin')
        )
        shared['python_plugin_cell_type_registered'] = bool(
            self.simulator.pluginManager.isRegistered('CellType')
        )
        shared['python_steppable_blob_loaded'] = bool(
            self.simulator.steppableManager.isLoaded('BlobInitializer')
        )
        shared['python_steppable_missing_loaded'] = bool(
            self.simulator.steppableManager.isLoaded('DefinitelyMissingSteppable')
        )
        shared['python_steppable_blob_registered'] = bool(
            self.simulator.steppableManager.isRegistered('BlobInitializer')
        )

    def step(self, mcs):
        shared = self.shared_steppable_vars
        if 'discovery_cpp_failures' in shared:
            print(
                'Runtime discovery ABI mcs=', mcs,
                'failures=', shared.get('discovery_cpp_failures'),
                'plugins=', shared.get('discovery_cpp_plugin_count'),
                'steppables=', shared.get('discovery_cpp_steppable_count')
            )
