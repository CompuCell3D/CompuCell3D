from os.path import dirname, join
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller
from cc3d.CompuCellSetup.CC3DRenderer import CC3DRenderer, standard_lds_file, standard_screenshot_file

sim_name = 'cellsort_2D.cc3d'
output_frequency = 40
render_dir = 'Figs'


def main():
    # Get the absolute path to the cc3d simulation file and output directory where results will be stored
    cc3d_sim_fname = join(dirname(__file__), sim_name)  # where to find the cc3d simulation file
    output_dir = join(dirname(__file__), 'Results')  # where results will be stored
    # Execute CC3D with a specified output frequency
    cc3d_caller = CC3DCaller(
        cc3d_sim_fname=cc3d_sim_fname,
        output_frequency=output_frequency,
        output_dir=output_dir
    )
    ret_val = cc3d_caller.run()  # Run it!
    # Get the absolute path to the simulation directory, lattice data summary file and screenshot json
    sim_dir = join(dirname(__file__), 'Simulation')  # where to find the CC3D model specification
    lds_file = standard_lds_file(output_dir)  # lattice data summary file
    screenshot_spec = standard_screenshot_file(sim_dir)  # screenshot json
    # Render results and store the renders in a specific location
    render_dir_abs = join(dirname(__file__), render_dir)  # Storing here
    cc3d_renderer = CC3DRenderer(
        lds_file=lds_file,
        screenshot_spec=screenshot_spec,
        output_dir=render_dir_abs
    )
    cc3d_renderer.initialize()  # Initialize CC3D rendering backend; we can do manipulations before and after this call
    cc3d_renderer.export_all()  # Render and export all available results


if __name__ == '__main__':
    main()
