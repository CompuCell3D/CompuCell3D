from cc3d.CC3DCaller import CC3DCaller

sim_fnames = [r'd:\CC3DProjects\demo_py3\cellsort_2D\cellsort_2D.cc3d'] * 4

ret_values = []
for i, sim_fname in enumerate(sim_fnames):

    cc3d_caller = CC3DCaller()

    cc3d_caller.cc3d_sim_fname = sim_fname
    cc3d_caller.screenshot_output_frequency = 10
    cc3d_caller.output_dir = r'c:\Users\m\CC3DWorkspace\cellsort_'+f'{i}'

    ret_value = cc3d_caller.run()
    ret_values.append(ret_value)

print ('return values', ret_values)