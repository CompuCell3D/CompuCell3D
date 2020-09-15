from os.path import dirname, join
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller
from cc3d.CompuCellSetup.CC3DRenderer import CC3DRenderer, standard_lds_file, standard_screenshot_file

sim_name = 'cellsort_2D.cc3d'
output_frequency = 40
render_dir = 'Figs'


def main():
    cc3d_sim_fname = join(dirname(__file__), sim_name)
    output_dir = join(dirname(__file__), 'Results')
    
    cc3d_caller = CC3DCaller(
        cc3d_sim_fname=cc3d_sim_fname,
        output_frequency=output_frequency,
        output_dir=output_dir
    )
    ret_val = cc3d_caller.run()

    sim_dir = join(dirname(__file__), 'Simulation')
    lds_file = standard_lds_file(output_dir)
    screenshot_spec = standard_screenshot_file(sim_dir)
    render_dir_abs = join(dirname(__file__), render_dir)
    cc3d_renderer = CC3DRenderer(
        lds_file=lds_file,
        screenshot_spec=screenshot_spec,
        output_dir=render_dir_abs
    )
    cc3d_renderer.initialize()
    cc3d_renderer.export_all()


if __name__ == '__main__':
    main()
