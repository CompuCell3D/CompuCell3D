from os.path import dirname, join
from cc3d.core.sim_service import service_cc3d

output_frequency = None

# Single run version
# simulation_fname = join(dirname(__file__), 'steppableBasedMitosis', 'steppableBasedMitosis.cc3d')
# root_output_folder = join(dirname(__file__), 'steppableBasedMitosis', 'Output')

# Embedded CC3D version
root_dir = r'C:\Users\T.J\Desktop\Current\CC3D\_CC3DService\CC3DConcurrentSimDemo\InfectionSite'
simulation_fname = join(root_dir, r'InfectionSite.cc3d')
root_output_folder = join(root_dir, 'Output')

if __name__ == '__main__':
    """
    Doing this through the Service module, with additional features:  
    from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
    cc3d_sim = CC3DSimService(cc3d_sim_fname=simulation_fname,
                              output_frequency=output_frequency,
                              output_dir=root_output_folder)
    """

    cc3d_sim = service_cc3d(cc3d_sim_fname=simulation_fname,
                            output_frequency=output_frequency,
                            output_dir=root_output_folder)
    cc3d_sim.run()
    # This is defined in the CC3D simulation files in the function
    # "add" and renamed for the service wrap as "user_named_add"
    val = cc3d_sim.user_named_add(1, 2)
    print(f"Got internal function evaluation via a service function: {val}")

    cc3d_sim.init()
    cc3d_sim.start()
    [cc3d_sim.step() for _ in range(10)]
    s = cc3d_sim.current_step
    o = cc3d_sim.sim_output
    p = cc3d_sim.profiler_report
    cc3d_sim.finish()
