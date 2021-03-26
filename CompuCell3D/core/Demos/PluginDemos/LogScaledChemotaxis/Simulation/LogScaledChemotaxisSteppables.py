from cc3d.core.PySteppables import *


class LogScaledChemotaxisSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)
        self.plot_win = None

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        self.plot_win = self.add_new_plot_window(title='Number of cells',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Total', 
                                                 x_scale_type='linear', 
                                                 y_scale_type='linear',
                                                 grid=True,
                                                 config_options={"legend": True})
        self.plot_win.add_plot("LogScaled", style='Lines', color='red', size=5)
        self.plot_win.add_plot("NotScaled", style='Lines', color='blue', size=5)
        self.plot_win.add_data_point("LogScaled", 0, len(self.cell_list_by_type(self.cell_type.LogScaled)))
        self.plot_win.add_data_point("NotScaled", 0, len(self.cell_list_by_type(self.cell_type.NotScaled)))

        # If not specifying chemotaxis for LogScaled in CC3DML, then uncomment this loop!
        # for cell in self.cell_list_by_type(self.cell_type.LogScaled):
        #     cd: CompuCell.ChemotaxisData = self.chemotaxisPlugin.addChemotaxisData(cell, "Chemoattractant")
        #     cd.setLambda(1E3)
        #     cd.setLogScaledCoef(1.0)

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        
        :param mcs: current Monte Carlo step
        """
        self.plot_win.add_data_point("LogScaled", mcs, len(self.cell_list_by_type(self.cell_type.LogScaled)))
        self.plot_win.add_data_point("NotScaled", mcs, len(self.cell_list_by_type(self.cell_type.NotScaled)))

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return
