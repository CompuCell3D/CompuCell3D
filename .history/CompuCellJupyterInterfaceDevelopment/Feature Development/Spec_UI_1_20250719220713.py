import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML
)
from IPython.display import display
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin,
    BoundaryPixelTrackerPlugin,
    CellTypePlugin,
    CenterOfMassPlugin,
    ChemotaxisPlugin,
    ConnectivityGlobalPlugin,
    ConnectivityPlugin,
    ContactPlugin,
    CurvaturePlugin,
    ExternalPotentialPlugin,
    FocalPointPlasticityPlugin,
    LengthConstraintPlugin,
    MomentOfInertiaPlugin,
    NeighborTrackerPlugin,
    PixelTrackerPlugin,
    SecretionPlugin,
    SurfacePlugin,
    VolumePlugin,
)

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Default values
DEFAULTS = {
    "Metadata": {
        "num_processors": 4,
        "debug_output_frequency": 10
    },
    "PottsCore": {
        "dim_x": 100,
        "dim_y": 100,
        "dim_z": 1,
        "steps": 100000,
        "anneal": 0,
        "fluctuation_amplitude": 10.0,
        "fluctuation_amplitude_function": "Min",
        "boundary_x": "NoFlux",
        "boundary_y": "NoFlux",
        "boundary_z": "NoFlux",
        "neighbor_order": 1,
        "random_seed": None,
        "lattice_type": "Cartesian",
        "offset": 0
    },
    "CellType": [
        {"Cell type": "Medium", "freeze": False}
    ],
    "Constraints": {
        "Volume": [
            {"CellType": "Medium", "enabled": False, "target_volume": 25.0, "lambda_volume": 2.0}
        ],
        "Surface": []
    }
}


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

class CellTypeWidget:
    def __init__(self, saved_entries):
        self.celltype_entries = saved_entries
        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()
    
    def create_widgets(self):
        # Display area for current cell types
        self.widgets["display_box"] = VBox(
            layout=Layout(border='1px solid gray', padding='10px', width='300px')
        )
        
        # Input widgets
        self.widgets["preset_dropdown"] = Dropdown(
            options=["Medium", "Condensing", "NonCondensing", "Customize"],
            value="Medium",
            description="Cell Type:",
            style={'description_width': 'initial'},
            layout=Layout(width='250px')
        )
        self.widgets["freeze_checkbox"] = Checkbox(
            value=True,
            description="Freeze",
            indent=False,
            layout=Layout(margin='0 0 0 20px')
        )
        self.widgets["custom_name_input"] = Text(
            description="Name:",
            placeholder="e.g. T1",
            style={'description_width': 'initial'},
            layout=Layout(width='250px')
        )
        self.widgets["custom_name_input"].layout.display = 'none'
        self.widgets["add_button"] = Button(
            description="Add",
            button_style="success",
            layout=Layout(width='80px', margin='10px 0 0 0')
        )
        
        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Tab",
            button_style='warning',
            layout=Layout(width='100px')
        )
    
    def setup_event_handlers(self):
        self.widgets["preset_dropdown"].observe(self.toggle_custom_input, names='value')
        self.widgets["add_button"].on_click(self.on_add_clicked)
        self.widgets["reset_button"].on_click(self.reset)
    
    def toggle_custom_input(self, change):
        self.widgets["custom_name_input"].layout.display = 'block' if change['new'] == "Customize" else 'none'
    
    def on_add_clicked(self, _):
        selected = self.widgets["preset_dropdown"].value
        name = (self.widgets["custom_name_input"].value.strip() 
                if selected == "Customize" else selected)
        
        if not name:
            self.widgets["custom_name_input"].placeholder = "Please enter a name!"
            return
        
        self.celltype_entries.append({
            "Cell type": name, 
            "freeze": self.widgets["freeze_checkbox"].value
        })
        
        self.update_celltype_display()
        self.widgets["custom_name_input"].value = ""
    
    def update_celltype_display(self):
        items = []
        for i, entry in enumerate(self.celltype_entries):
            label_str = entry["Cell type"]
            if entry.get("freeze", False):
                label_str += " (frozen)"
            label = Label(label_str, layout=Layout(flex='1 1 auto'))
            remove_btn = Button(description="Remove", button_style='danger', layout=Layout(width='80px'))
            
            def make_remove_handler(index):
                def handler(_):
                    del self.celltype_entries[index]
                    self.update_celltype_display()
                return handler
            
            remove_btn.on_click(make_remove_handler(i))
            items.append(HBox([label, remove_btn], layout=Layout(justify_content='space-between')))
        
        self.widgets["display_box"].children = items
    
    def get_config(self):
        return self.celltype_entries.copy()
    
    def reset(self, _=None):
        self.celltype_entries = DEFAULTS["CellType"].copy()
        self.update_celltype_display()
    
    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]
    
    def create_ui(self):
        cell_type_row = HBox([
            self.widgets["preset_dropdown"],
            self.widgets["freeze_checkbox"]
        ])
        
        return VBox([
            HBox([
                VBox([
                    Label("Current Cell Types:", style={'font_weight': 'bold'}),
                    self.widgets["display_box"],
                    self.widgets["reset_button"]
                ], layout=Layout(width='320px', padding='0 20px 0 0')),
                VBox([
                    Label("Add Cell Type:", style={'font_weight': 'bold'}),
                    cell_type_row,
                    self.widgets["custom_name_input"],
                    self.widgets["add_button"]
                ])
            ])
        ], layout=Layout(padding='10px'))

class ConstraintsWidget:
    def __init__(self, saved_constraints, celltype_widget):
        self.constraints = saved_constraints
        self.celltype_widget = celltype_widget
        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_constraints_display()
    
    def create_widgets(self):
        # Cell type selection
        self.widgets["celltype_dropdown"] = widgets.Dropdown(
            options=self.celltype_widget.get_cell_type_names(),
            description='Cell Type:',
            style={'description_width': 'initial'}
        )
        
        # Volume constraints
        self.widgets["vol_enabled"] = widgets.Checkbox(
            value=False,
            description="Enable Volume Constraints",
            indent=False,
            style={'description_width': 'initial'}
        )
        self.widgets["target_volume"] = widgets.FloatText(
            value=25.0,
            min=1.0,
            description='Target Volume:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_volume"] = widgets.FloatText(
            value=2.0,
            min=0.0,
            description='Lambda Volume:',
            style={'description_width': 'initial'}
        )
        
        # Surface constraints
        self.widgets["surf_enabled"] = widgets.Checkbox(
            value=False,
            description="Enable Surface Constraints",
            indent=False,
            style={'description_width': 'initial'}
        )
        self.widgets["target_surface"] = widgets.FloatText(
            value=100.0,
            min=1.0,
            description='Target Surface:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_surface"] = widgets.FloatText(
            value=0.5,
            min=0.0,
            description='Lambda Surface:',
            style={'description_width': 'initial'}
        )
        
        # Action buttons
        self.widgets["add_button"] = Button(
            description="Add Constraints",
            button_style="success",
            layout=Layout(width='150px')
        )
        self.widgets["reset_button"] = Button(
            description="Reset Tab",
            button_style='warning',
            layout=Layout(width='100px')
        )
        
        # Display area
        self.widgets["display_box"] = VBox(
            layout=Layout(border='1px solid gray', padding='10px', width='500px'))
    
    def setup_event_handlers(self):
        self.widgets["add_button"].on_click(self.on_add_constraints)
        self.widgets["reset_button"].on_click(self.reset)
    
    def on_add_constraints(self, _):
        cell_type = self.widgets["celltype_dropdown"].value
        
        # Volume constraints
        vol_exists = False
        for entry in self.constraints["Volume"]:
            if entry["CellType"] == cell_type:
                entry.update({
                    "enabled": self.widgets["vol_enabled"].value,
                    "target_volume": self.widgets["target_volume"].value,
                    "lambda_volume": self.widgets["lambda_volume"].value
                })
                vol_exists = True
                break
        
        if not vol_exists:
            self.constraints["Volume"].append({
                "CellType": cell_type,
                "enabled": self.widgets["vol_enabled"].value,
                "target_volume": self.widgets["target_volume"].value,
                "lambda_volume": self.widgets["lambda_volume"].value
            })
        
        # Surface constraints
        surf_exists = False
        for entry in self.constraints["Surface"]:
            if entry["CellType"] == cell_type:
                entry.update({
                    "enabled": self.widgets["surf_enabled"].value,
                    "target_surface": self.widgets["target_surface"].value,
                    "lambda_surface": self.widgets["lambda_surface"].value
                })
                surf_exists = True
                break
        
        if not surf_exists and self.widgets["surf_enabled"].value:
            self.constraints["Surface"].append({
                "CellType": cell_type,
                "enabled": self.widgets["surf_enabled"].value,
                "target_surface": self.widgets["target_surface"].value,
                "lambda_surface": self.widgets["lambda_surface"].value
            })
        
        self.update_constraints_display()
    
    def update_constraints_display(self):
        items = []
        
        # Volume constraints
        vol_header = Label("Volume Constraints:", style={'font_weight': 'bold', 'font_style': 'italic'})
        items.append(vol_header)
        
        for entry in self.constraints["Volume"]:
            ct = entry["CellType"]
            enabled = "✓" if entry["enabled"] else "✗"
            text = f"{ct}: Enabled={enabled}, Target={entry['target_volume']}, λ={entry['lambda_volume']:.2f}"
            label = Label(text, layout=Layout(flex='1 1 auto'))
            
            remove_btn = Button(description="Remove", button_style='danger', layout=Layout(width='80px'))
            
            def make_remove_handler(ct):
                def handler(_):
                    self.constraints["Volume"] = [e for e in self.constraints["Volume"] if e["CellType"] != ct]
                    self.constraints["Surface"] = [e for e in self.constraints["Surface"] if e["CellType"] != ct]
                    self.update_constraints_display()
                return handler
            
            remove_btn.on_click(make_remove_handler(ct))
            items.append(HBox([label, remove_btn], layout=Layout(justify_content='space-between')))
        
        # Surface constraints
        surf_header = Label("Surface Constraints:", style={'font_weight': 'bold', 'font_style': 'italic'})
        items.append(surf_header)
        
        for entry in self.constraints["Surface"]:
            ct = entry["CellType"]
            enabled = "✓" if entry["enabled"] else "✗"
            text = f"{ct}: Enabled={enabled}, Target={entry['target_surface']}, λ={entry['lambda_surface']:.2f}"
            label = Label(text, layout=Layout(flex='1 1 auto'))
            
            remove_btn = Button(description="Remove", button_style='danger', layout=Layout(width='80px'))
            
            def make_remove_handler(ct):
                def handler(_):
                    self.constraints["Surface"] = [e for e in self.constraints["Surface"] if e["CellType"] != ct]
                    self.update_constraints_display()
                return handler
            
            remove_btn.on_click(make_remove_handler(ct))
            items.append(HBox([label, remove_btn], layout=Layout(justify_content='space-between')))
        
        if not self.constraints["Volume"] and not self.constraints["Surface"]:
            items.append(Label("No constraints defined"))
        
        self.widgets["display_box"].children = items
    
    def get_config(self):
        return self.constraints
    
    def reset(self, _=None):
        self.constraints = DEFAULTS["Constraints"].copy()
        self.update_constraints_display()
    
    def update_cell_type_options(self):
        self.widgets["celltype_dropdown"].options = self.celltype_widget.get_cell_type_names()
    
    def create_ui(self):
        return VBox([
            HBox([
                self.widgets["celltype_dropdown"],
                self.widgets["add_button"]
            ]),
            HTML("<hr><b>Volume Constraints:</b>"),
            self.widgets["vol_enabled"],
            HBox([self.widgets["target_volume"], self.widgets["lambda_volume"]]),
            HTML("<hr><b>Surface Constraints:</b>"),
            self.widgets["surf_enabled"],
            HBox([self.widgets["target_surface"], self.widgets["lambda_surface"]]),
            HTML("<hr><b>Current Constraints:</b>"),
            self.widgets["display_box"],
            self.widgets["reset_button"]
        ], layout=Layout(padding='10px'))


class SpecificationSetupUI:
    def __init__(self):
        self.widgets = {}
        self.saved_values = self.load_saved_values()
        self.celltype_entries = self.saved_values.get("CellType", DEFAULTS["CellType"].copy())

        self.metadata = Metadata(
            num_processors=self.saved_values["Metadata"]["num_processors"],
            debug_output_frequency=self.saved_values["Metadata"]["debug_output_frequency"]
        )

        # this part works only when all parameters are passed properly
        '''self.potts_core = PottsCore(
            **self.saved_values["PottsCore"]
        )'''

        pottscore_keys = [
            "dim_x", "dim_y", "dim_z", "steps", "anneal", "fluctuation_amplitude",
            "fluctuation_amplitude_function", "boundary_x", "boundary_y", "boundary_z",
            "neighbor_order", "random_seed", "lattice_type", "offset"
        ]
        pottscore_dict = {k: v for k, v in self.saved_values["PottsCore"].items() if k in pottscore_keys}
        self.potts_core = PottsCore(**pottscore_dict)

        cell_type_data = self.saved_values.get("CellType", DEFAULTS["CellType"].copy())
        self.cell_type_plugin = CellTypePlugin()
        for entry in cell_type_data:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                frozen=entry.get("freeze", False)
            )

        # Create metadata widgets immediately
        self.create_metadata_widgets()

        # Initialize widgets
        self.potts_widget = PottsWidget(self.potts_core.spec_dict)
        self.celltype_widget = CellTypeWidget(cell_type_data)
        self.constraints_widget = ConstraintsWidget(
            self.saved_values.get("Constraints", json.loads(json.dumps(DEFAULTS["Constraints"]))),
            self.celltype_widget
        )

        # Create the UI
        self.create_ui()
        self.setup_event_handlers()

    def create_metadata_widgets(self):
        """Metadata widgets"""
        self.widgets["num_proc"] = widgets.IntText(
            value=self.metadata.num_processors, min=1,
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_freq"] = widgets.IntText(
            value=self.metadata.debug_output_frequency, min=0,
            description='Debug Output Frequency:',
            style={'description_width': 'initial'}
        )
        self.widgets["reset_metadata_button"] = Button(
            description="Reset Tab",
            button_style='warning',
            layout=Layout(width='100px'))
        self.widgets["reset_button"] = Button(
            description="Reset All to Defaults",
            button_style='danger')

    def load_saved_values(self):
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, 'r') as f:
                    return json.load(f)
            return json.loads(json.dumps(DEFAULTS))
        except (json.JSONDecodeError, IOError):
            print("JSON file is corrupted or inaccessible. Resetting to defaults.")
            return json.loads(json.dumps(DEFAULTS))

    def current_config(self):
        return {
            "Metadata": {
                "num_processors": self.metadata.num_processors,
                "debug_output_frequency": self.metadata.debug_output_frequency
            },
            "PottsCore": self.potts_core.spec_dict,
            "CellType": [
                {"Cell type": ct[0], "freeze": ct[2]}
                for ct in self.cell_type_plugin.spec_dict["cell_types"]
            ],
            "Constraints": self.constraints_widget.get_config()
        }

    def save_to_json(self, _=None):
        with open(SAVE_FILE, 'w') as f:
            json.dump(self.current_config(), f, indent=4)

    def setup_event_handlers(self):
        # Metadata handlers
        self.widgets["num_proc"].observe(
            lambda change: setattr(self.metadata, 'num_processors', change.new), 
            names='value'
        )
        self.widgets["debug_freq"].observe(
            lambda change: setattr(self.metadata, 'debug_output_frequency', change.new),
            names='value'
        )

        # Connect reset handlers for each tab
        self.potts_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_potts_tab()
        )
        self.celltype_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_celltype_tab()
        )
        self.constraints_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_constraints_tab()
        )
        self.widgets["reset_metadata_button"].on_click(
            lambda _: self.reset_metadata_tab()
        )

        # Global reset button
        self.widgets["reset_button"].on_click(
            lambda _: self.reset_all()
        )

        # PottsCore widget handlers
        for name, widget in self.potts_widget.widgets.items():
            if hasattr(widget, 'observe'):
                widget.observe(
                    lambda change, prop=name: self.update_potts_core(prop, change.new),
                    names='value'
                )

        # CellType handlers
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.update_cell_types()
        )

        # Constraints handlers
        self.constraints_widget.widgets["add_button"].on_click(
            lambda _: self.save_to_json()
        )

        # Update cell type options in constraints when cell types change
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.constraints_widget.update_cell_type_options()
        )

        # Save triggers
        self.widgets["num_proc"].observe(
            lambda _: self.save_to_json(), names='value'
        )
        self.widgets["debug_freq"].observe(
            lambda _: self.save_to_json(), names='value'
        )

    def update_potts_core(self, property_name, value):
        """
        Update PottsCore object when UI widgets change
        Handles special cases like lattice type conversion and random seed
        """
        property_map = {
            "x_slider": "dim_x",
            "y_slider": "dim_y",
            "z_slider": "dim_z",
            "steps_input": "steps",
            "anneal_input": "anneal",
            "fluctuation_slider": "fluctuation_amplitude",
            "flunct_fn_dropdown": "fluctuation_amplitude_function",
            "boundary_x": "boundary_x",
            "boundary_y": "boundary_y",
            "boundary_z": "boundary_z",
            "neighbor_order_input": "neighbor_order",
            "seed_input": "random_seed",
            "lattice_dropdown": "lattice_type",
            "offset_input": "offset"
        }
        core_property = property_map.get(property_name)
        if not core_property:
            return
        if property_name == "lattice_dropdown":
            value = 'Cartesian' if value == 'Square' else 'Hexagonal'
        elif property_name == "seed_input":
            value = int(value) if value.strip() else None
        try:
            setattr(self.potts_core, core_property, value)
            self.save_to_json()
        except Exception as e:
            print(f"Error updating PottsCore: {e}")

    def create_ui(self):
        tabs = Tab()
        tabs.children = [
            self.create_metadata_tab(),
            self.potts_widget.create_ui(),
            self.celltype_widget.create_ui(),
            self.constraints_widget.create_ui()
        ]
        tabs.set_title(0, 'Basic Setup')
        tabs.set_title(1, 'Potts Core')
        tabs.set_title(2, 'Cell Types')
        tabs.set_title(3, 'Constraints')
        display(VBox([
            tabs,
            self.widgets["reset_button"]
        ]))

    def create_metadata_tab(self):
        return VBox([
            self.widgets["num_proc"],
            self.widgets["debug_freq"],
            self.widgets["reset_metadata_button"]
        ], layout=Layout(padding='10px'))

    def reset_potts_tab(self):
        self.potts_widget.reset()
        for prop, value in DEFAULTS["PottsCore"].items():
            setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        self.celltype_widget.reset()
        self.cell_type_plugin.spec_dict["cell_types"] = [("Medium", 0, False)]
        for entry in self.celltype_widget.celltype_entries:
            if entry["Cell type"] != "Medium":
                self.cell_type_plugin.cell_type_append(
                    entry["Cell type"],
                    frozen=entry["freeze"]
                )
        self.constraints_widget.update_cell_type_options()
        self.save_to_json()

    def reset_constraints_tab(self):
        self.constraints_widget.reset()
        self.save_to_json()

    def reset_metadata_tab(self):
        self.widgets["num_proc"].value = DEFAULTS["Metadata"]["num_processors"]
        self.widgets["debug_freq"].value = DEFAULTS["Metadata"]["debug_output_frequency"]
        self.metadata.num_processors = DEFAULTS["Metadata"]["num_processors"]
        self.metadata.debug_output_frequency = DEFAULTS["Metadata"]["debug_output_frequency"]
        self.save_to_json()

    def reset_all(self):
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_constraints_tab()
        self.reset_metadata_tab()

    def update_cell_types(self):
        # Update cell_type_plugin from celltype_widget
        self.cell_type_plugin.spec_dict["cell_types"] = [("Medium", 0, False)]
        for entry in self.celltype_widget.celltype_entries:
            if entry["Cell type"] != "Medium":
                self.cell_type_plugin.cell_type_append(
                    entry["Cell type"],
                    frozen=entry["freeze"]
                )
        self.constraints_widget.update_cell_type_options()
        self.save_to_json()

# Main execution
if __name__ == "__main__":
    ui = SpecificationSetupUI()