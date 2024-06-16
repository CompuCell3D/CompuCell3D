"""
This is demo script shows how one ca use CC3D to run a sequence of CC3d simulations where each simulation returns
some value(s) - think of values as simulation metrics. This particular example shows a simple loop that executes
few simulations in a row and collects simulation return values. Depending on your needs you can be more creative
and replace a simple loop with e.g. optimization algorithm of your choice. We intentionally kep this example simple

Notice: CC3DCaller executes simulation in the non-gui mode only.

Notice regarding screenshots -  when you run simulations you may take screenshots as the simulation runs. Keep in mind
that taking screenshots takes time so your simulation can be slowed down because of this.
Screenshots on linux notice: if you are on linux systems in order for screenshots to work you need to compile VTK
in the offscreen mode and then recompile CC3D using this offscreen vtk library. Provisional  instructions can be found
here:
https://github.com/CompuCell3D/cc3d_build_scripts/tree/master/linux/vtk-offscreen

but be aware that with newer VTK libraries that are necessary to run versions 4.x.x of CC3D,  the details of
compilations may vary.

Running this example:
====================
BEfore running the example make sure to set environment variables or prepare your own Python environment
that has cc3d install in it. you can find details of hot to do it in the Python Scripting Manual

https://pythonscriptingmanual.readthedocs.io/en/latest/

To run you execute the following command:

python cc3d_call_single_cpu.py

"""

from os.path import dirname, join, expanduser, abspath
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller


def main():
    number_of_runs = 4

    # You may put a direct path to a simulation of your choice here and comment out simulation_fname line below
    # simulation_fname = <direct path your simulation>
    simulation_fname = join(dirname(dirname(abspath(__file__))), 'cellsort_2D', 'cellsort_2D.cc3d')
    root_output_folder = join(expanduser('~'), 'CC3DCallerOutput')

    # this creates a list of simulation file names where simulation_fname is repeated number_of_runs times
    # you can create a list of different simulations if you want.
    sim_fnames = [simulation_fname] * number_of_runs

    ret_values = []
    for i, sim_fname in enumerate(sim_fnames):
        cc3d_caller = CC3DCaller(
            cc3d_sim_fname=sim_fname,
            screenshot_output_frequency=10,
            output_dir=join(root_output_folder, f'cellsort_{i}'),
            result_identifier_tag=i
        )

        ret_value = cc3d_caller.run()
        ret_values.append(ret_value)

    print('return values', ret_values)


if __name__ == '__main__':
    main()
