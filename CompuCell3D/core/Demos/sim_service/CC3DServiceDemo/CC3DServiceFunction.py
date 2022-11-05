"""
This demo teaches some applications of the simservice CC3D implementation, including
    - How to interactively run a CC3D simulation service in Python using a service proxy
    - How to run CC3D simulations within a CC3D simulation as a service
    - How to define and use a service function
    - How to forward the service functions of services running within services 

A simulation service behaves similarly to an interactive CC3D simulation, but with some subtle differences. When
instantiating a simulation service, a reference to the actual simulation service is not returned. Rather, a proxy
to the simulation service is returned that has methods to interact with the underlying simulation service, which
is runnning elsewhere on a server (whether locally, or remotely). Some of the methods available are demonstrated
below, including interactively running a simulation and designing/employing customization of a service interface
through service functions.

A service function defines an internal service method that can be employed through the service interface,
which can be accessed as an attribute on the service proxy. They can be designed into a simulation without
breaking the simulation when running normally (e.g., when running directly in Player), as the simservice
implementation will detect if a simulation is being run as a service. In cases where a simulation is employed as
a service, service function instantiations occur. In cases where a simulation is not being employed as a serivce,
service function instantiations are simply ignored.

A service proxy can be instantiated using its service factory defined in simservice. For CC3D, the
factory is service_cc3d, which takes the same arguments as those that we would pass to CC3DCaller and
CC3DSimService to instantiate a simulation.

When running services within services, the service functions of a child service can be mapped onto the proxy of
the parent service by simply declaring the service functions of the child service as service functions of the parent
service in the typical way.
"""
from os.path import dirname, join
from cc3d.core.simservice import service_cc3d

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

output_frequency = None
total_steps = 10000  # Total number of simulation steps
report_frequency = 100  # Frequency of reporting simulation details

# Embedded CC3D: the CC3D project InfectionSite embeds a second CC3D project EffectorProductionSite. It can be executed
# in Player, or through the simservice implementation as demonstrated here
root_dir = join(dirname(dirname(__file__)), 'CC3DConcurrentSimDemo', 'InfectionSite')
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

    cc3d_sim.init()
    cc3d_sim.start()
    for step in range(total_steps):
        if step % report_frequency == 0:
            print(f"Simulation step {cc3d_sim.current_step}")
            # Report available information defined by the services via their available service functions
            # These are defined by the CC3D simulation specification, and are not standard features of the CC3D
            # simulation service in general
            # Note that two of these service functions (total_signal_sent and total_cells_departed) are defined by
            # the EffectorProductionSite simulation specification, which is acting as a service within in the
            # InfectionSite simulation service. In hierarchies like these, the parent service (here InfectionSite) can
            # make available the service functions of child serivces (here EffectorProductionSite) by simply declaring
            # the service functions of child services as service functions of the parent service in the typical way
            # See the steppable scripts for the InfectionSite and EffectorProductionSite simulations for examples of
            # how this is done, especially by noting where the function "service_function" is employed
            print(f"Total signal sent by infection site: {cc3d_sim.total_signal_sent()}")
            print(f"Total signal received by recruitment site: {cc3d_sim.total_signal_received()}")
            print(f"Total cells sent by recruitment site: {cc3d_sim.total_cells_departed()}")
            print(f"Total cells received by infection stie: {cc3d_sim.total_cells_received()}")

        cc3d_sim.step()

    cc3d_sim.finish()
    print(cc3d_sim.profiler_report)
