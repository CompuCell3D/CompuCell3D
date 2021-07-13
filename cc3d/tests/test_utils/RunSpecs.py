
class RunSpecs():
    def __init__(self):

        # path to ramulator run script
        self.run_command = ''
        self.cc3d_project = ''
        self.test_output_root = ''
        self.num_steps = 100
        # the directory - created for each individual run where the data from emulated run gets written
        self.test_output_dir = ''
        self.player_interactive_flag = True
