
from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell


class FluctuationCompensatorTestSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

        self.total_man_ini = None
        self.total_bias = 0.0

        self.plot_win1 = None
        self.plot_win2 = None

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

        # Generate plots for graphical reporting
        self.plot_win1 = self.add_new_plot_window(
            title='Total medium concentration',
            x_axis_title='Monte Carlo Step',
            y_axis_title='Total medium concentration',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True,
            config_options={'legend': True}
        )
        self.plot_win1.add_plot("ConcPTPMed", style='Dots', color='red', size=5)
        self.plot_win1.add_plot("ConcManMed", style='Dots', color='green', size=5)
        self.plot_win1.add_plot("ConcMan", style='Dots', color='blue', size=5)

        # Create plot window for Cell 1 volume and surface
        self.plot_win2 = self.add_new_plot_window(
            title='Total medium concentration errors',
            x_axis_title='Monte Carlo Step',
            y_axis_title='Errors',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=True,
            config_options={'legend': True}
        )
        self.plot_win2.add_plot("ErrorConcPTPMed", style='Dots', color='red', size=5)
        self.plot_win2.add_plot("ErrorConcManMod", style='Dots', color='green', size=5)

        # Set all values to zero in cells
        f = CompuCell.getConcentrationField(self.simulator, "F1")
        for x, y, z in self.every_pixel():
            if self.cell_field[x, y, z]:
                f[x, y, z] = 0.0

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        if self.pixel_tracker_plugin is None:
            return

        f = CompuCell.getConcentrationField(self.simulator, "F1")

        # Test modification of field values outside of core routines
        mcs_change_value = 1e3
        if mcs == mcs_change_value:
            id_of_cell_to_change = 1
            val_after_change = 1.0
            for ptd in self.get_cell_pixel_list(self.fetch_cell_by_id(id_of_cell_to_change)):
                self.total_bias += val_after_change - f[ptd.pixel.x, ptd.pixel.y, 0]
                f[ptd.pixel.x, ptd.pixel.y, 0] = val_after_change

            # Without this call, modifications (other than by core routines) to a field with a solver using
            # FluctuationCompensator will likely cause numerical errors
            CompuCell.updateFluctuationCompensators()

        # Do testing
        mcs_min_test = 0
        if mcs >= mcs_min_test:

            # For all periodic boundary conditions and any initial distribution, changes in these values
            # over simulation time should only be due to the accumulation of rounding errors in the
            # FluctuationCompensator algorithm
            total_ptp_medium = 0.0
            total_man_medium = 0.0
            total_man = 0.0
            for ptd in self.pixel_tracker_plugin.getMediumPixelSet():
                total_ptp_medium += f[ptd.pixel.x, ptd.pixel.y, ptd.pixel.z]

            for x, y, z in self.every_pixel():
                this_f = f[x, y, z]
                total_man += this_f
                if not self.cell_field[x, y, z]:
                    total_man_medium += this_f

            self.plot_win1.add_data_point('ConcPTPMed', mcs, total_ptp_medium)
            self.plot_win1.add_data_point('ConcManMed', mcs, total_man_medium)
            self.plot_win1.add_data_point('ConcMan', mcs, total_man)

            # This compares the total concentration counted by looping over the lattice (the old way) to the total
            # concentration counted from medium pixel sites returned by pixel tracker (a new feature).
            # Pixel tracker should be able to keep track of medium sites for single- and multi-threaded simulations
            # so that looping over the entire lattice is no longer necessary.
            # If pixel tracker isn't properly tracking medium sites, then errors will be non-zero here
            err = 0.0
            if total_man_medium != 0:
                err = total_ptp_medium / total_man_medium - 1.0

            self.plot_win2.add_data_point('ErrorConcPTPMed', mcs, err)

            if self.total_man_ini is None:
                self.total_man_ini = total_man

            # This compares the current total concentration to the initial total concentration. FluctuationCompensator
            # should be able to keep these values the same (neglecting machine errors) for domains with all periodic
            # boundary conditions, even when field values are modified outside of core routines (Metropolis and solver).
            # For FluctuationCompensator to account for modifications to any field outside of core routines, call
            # updateFluctuationCompensators after all modifications for a step have been made.
            # If a field is modified outside of core routines and updateFluctuationCompensators is not called,
            # then errors generated by FluctuationCompensator can be detected here
            # To test this functionality, make changes to field values, add all differences in field values due to
            # the changes to self.total_bias and see what happens when updateFluctuationCompensators is/isn't called
            err = 0.0
            if self.total_man_ini != 0:
                err = (total_man - self.total_bias) / self.total_man_ini - 1.0

            self.plot_win2.add_data_point('ErrorConcManMod', mcs, err)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

