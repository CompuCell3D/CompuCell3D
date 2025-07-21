# ... (previous code remains the same)

class PottsWidget:
    def __init__(self, saved_values):
        self.widgets = {}
        self.defaults = PottsCore().spec_dict
        self.create_widgets(saved_values or self.defaults)

    def create_widgets(self, saved_values):
        # Dimension inputs with max=101
        self.widgets["dim_x"] = widgets.IntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, max=101, description='X Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_y"] = widgets.IntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, max=101, description='Y Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_z"] = widgets.IntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, max=101, description='Z Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Error display
        self.widgets["dim_x_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_y_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_z_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=0, description='MC Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["steps_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        
        self.widgets["anneal"] = widgets.IntText(
            value=saved_values.get("anneal", self.defaults["anneal"]),
            min=0, description='Anneal Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["anneal_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        
        self.widgets["fluctuation_amplitude"] = widgets.FloatText(
            value=saved_values.get("fluctuation_amplitude", self.defaults["fluctuation_amplitude"]),
            min=0.0, description='Fluctuation Amplitude:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["fluctuation_amplitude_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        
        # Fluctuation amplitude function
        self.widgets["fluctuation_amplitude_function"] = widgets.Dropdown(
            options=['Min', 'Max', 'Average'],
            value=saved_values.get("fluctuation_amplitude_function", self.defaults["fluctuation_amplitude_function"]),
            description='Fluctuation Function:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", self.defaults["neighbor_order"]),
            min=1, description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        
        self.widgets["lattice_type"] = widgets.Dropdown(
            options=['Cartesian', 'Hexagonal'],
            value=saved_values.get("lattice_type", self.defaults["lattice_type"]),
            description='Lattice Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Advanced settings
        self.widgets["offset"] = widgets.FloatText(
            value=saved_values.get("offset", self.defaults.get("offset", 0.0)),
            description='Offset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["random_seed"] = widgets.IntText(
            value=saved_values.get("random_seed", self.defaults.get("random_seed", 0)),
            min=0, description='Random Seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["energy_calculator"] = widgets.Dropdown(
            options=['Fast', 'Regular', 'Precise'],
            value=saved_values.get("energy_calculator", self.defaults.get("energy_calculator", 'Regular')),
            description='Energy Calculator:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        boundary_options = ['NoFlux', 'Periodic']
        self.widgets["boundary_x"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_x", self.defaults["boundary_x"]),
            description='X Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["boundary_y"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_y", self.defaults["boundary_y"]),
            description='Y Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["boundary_z"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_z", self.defaults["boundary_z"]),
            description='Z Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Potts",
            button_style='warning'
        )

    def get_config(self):
        return {
            "dim_x": self.widgets["dim_x"].value,
            "dim_y": self.widgets["dim_y"].value,
            "dim_z": self.widgets["dim_z"].value,
            "steps": self.widgets["steps"].value,
            "anneal": self.widgets["anneal"].value,
            "fluctuation_amplitude": self.widgets["fluctuation_amplitude"].value,
            "fluctuation_amplitude_function": self.widgets["fluctuation_amplitude_function"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order"].value,
            "lattice_type": self.widgets["lattice_type"].value,
            "offset": self.widgets["offset"].value,
            "random_seed": self.widgets["random_seed"].value,
            "energy_calculator": self.widgets["energy_calculator"].value
        }

    def reset(self):
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                widget.value = self.defaults[key]

    def create_ui(self):
        # Create centered rows
        dimensions_row = HBox([
            VBox([self.widgets["dim_x"], self.widgets["dim_x_error"]]),
            VBox([self.widgets["dim_y"], self.widgets["dim_y_error"]]),
            VBox([self.widgets["dim_z"], self.widgets["dim_z_error"]])
        ], layout=Layout(justify_content='center', align_items='center'))
        dimensions_row.add_class('vbox-row-spacing')

        core_params_row = HBox([
            VBox([self.widgets["steps"], self.widgets["steps_error"]]),
            VBox([self.widgets["anneal"], self.widgets["anneal_error"]]),
            VBox([self.widgets["fluctuation_amplitude"], self.widgets["fluctuation_amplitude_error"]])
        ], layout=Layout(justify_content='center', align_items='center'))
        core_params_row.add_class('vbox-row-spacing')
        
        fluct_function_row = HBox([
            self.widgets["fluctuation_amplitude_function"]
        ], layout=Layout(justify_content='center', align_items='center'))
        fluct_function_row.add_class('vbox-row-spacing')

        boundaries_row = HBox([
            self.widgets["boundary_x"],
            self.widgets["boundary_y"],
            self.widgets["boundary_z"]
        ], layout=Layout(justify_content='center', align_items='center'))
        boundaries_row.add_class('vbox-row-spacing')

        advanced_row1 = HBox([
            VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]]),
            self.widgets["lattice_type"]
        ], layout=Layout(justify_content='center', align_items='center'))
        advanced_row1.add_class('vbox-row-spacing')

        advanced_row2 = HBox([
            self.widgets["offset"],
            self.widgets["random_seed"],
            self.widgets["energy_calculator"]
        ], layout=Layout(justify_content='center', align_items='center'))
        advanced_row2.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Potts Core Parameters</b>", layout=Layout(display='flex', justify_content='center')),
            dimensions_row,
            core_params_row,
            fluct_function_row,
            HTML("<b>Boundary Conditions:</b>", layout=Layout(display='flex', justify_content='center')),
            boundaries_row,
            HTML("<b>Advanced Settings:</b>", layout=Layout(display='flex', justify_content='center')),
            advanced_row1,
            advanced_row2,
            HBox([self.widgets["reset_button"]], layout=Layout(justify_content='center'))
        ])

# ... (rest of the code remains the same)