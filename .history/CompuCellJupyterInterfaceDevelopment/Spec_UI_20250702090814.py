import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML
)
from IPython.display import display
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, CellTypePlugin


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
        self.create_widgets(saved_values)
        
    def create_widgets(self, saved_values):
        # Dimension sliders
        self.widgets["x_slider"] = widgets.IntSlider(
            value=saved_values["dim_x"],
            min=0, max=100, description='X:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        self.widgets["y_slider"] = widgets.IntSlider(
            value=saved_values["dim_y"],
            min=0, max=100, description='Y:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        self.widgets["z_slider"] = widgets.IntSlider(
            value=saved_values["dim_z"],
            min=0, max=100, description='Z:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        # Core parameters
        self.widgets["steps_input"] = widgets.IntText(
            value=saved_values["steps"],
            description='MC Steps:',
            style={'description_width': 'initial'}
        )
        self.widgets["anneal_input"] = widgets.FloatText(
            value=saved_values["anneal"],
            description='Anneal:',
            style={'description_width': 'initial'}
        )
        self.widgets["fluctuation_slider"] = widgets.FloatSlider(
            value=saved_values["fluctuation_amplitude"],
            min=0.0, max=50.0, step=0.1,
            description='Fluctuation:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        self.widgets["flunct_fn_dropdown"] = widgets.Dropdown(
            options=['Min', 'Max', 'ArithmeticAverage'],
            value=saved_values["fluctuation_amplitude_function"],
            description='Fluctuation Function:',
            style={'description_width': 'initial'}
        )
        
        # Boundaries
        self.widgets["boundary_x"] = widgets.Dropdown(
            options=['NoFlux', 'Periodic'],
            value=saved_values["boundary_x"],
            description='Boundary X:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_y"] = widgets.Dropdown(
            options=['NoFlux', 'Periodic'],
            value=saved_values["boundary_y"],
            description='Boundary Y:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_z"] = widgets.Dropdown(
            options=['NoFlux', 'Periodic'],
            value=saved_values["boundary_z"],
            description='Boundary Z:',
            style={'description_width': 'initial'}
        )
        
        # Advanced settings
        self.widgets["neighbor_order_input"] = widgets.BoundedIntText(
            value=saved_values["neighbor_order"],
            min=1, max=20, description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["seed_input"] = widgets.Text(
            value='' if saved_values["random_seed"] is None 
                else str(saved_values["random_seed"]),
            description='Random Seed:',
            placeholder='e.g. 123456',
            style={'description_width': 'initial'}
        )
        self.widgets["lattice_dropdown"] = widgets.Dropdown(
            options=['Square', 'Hexagonal'],
            value='Square' if saved_values["lattice_type"] == 'Cartesian' 
                else 'Hexagonal',
            description='Lattice Type:',
            style={'description_width': 'initial'}
        )
        self.widgets["offset_input"] = widgets.IntText(
            value=saved_values["offset"],
            description='Offset:',
            style={'description_width': 'initial'}
        )
        
        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Tab",
            button_style='warning',
            layout=Layout(width='100px')
        )
    
    def get_config(self):
        return {
            "dim_x": self.widgets["x_slider"].value,
            "dim_y": self.widgets["y_slider"].value,
            "dim_z": self.widgets["z_slider"].value,
            "steps": self.widgets["steps_input"].value,
            "anneal": self.widgets["anneal_input"].value,
            "fluctuation_amplitude": self.widgets["fluctuation_slider"].value,
            "fluctuation_amplitude_function": self.widgets["flunct_fn_dropdown"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order_input"].value,
            "random_seed": int(self.widgets["seed_input"].value) if self.widgets["seed_input"].value.strip() else None,
            "lattice_type": 'Cartesian' if self.widgets["lattice_dropdown"].value == 'Square' else 'Hexagonal',
            "offset": self.widgets["offset_input"].value
        }
    
    def reset(self):
        defaults = DEFAULTS["PottsCore"]
        self.widgets["x_slider"].value = defaults["dim_x"]
        self.widgets["y_slider"].value = defaults["dim_y"]
        self.widgets["z_slider"].value = defaults["dim_z"]
        self.widgets["steps_input"].value = defaults["steps"]
        self.widgets["anneal_input"].value = defaults["anneal"]
        self.widgets["fluctuation_slider"].value = defaults["fluctuation_amplitude"]
        self.widgets["flunct_fn_dropdown"].value = defaults["fluctuation_amplitude_function"]
        self.widgets["boundary_x"].value = defaults["boundary_x"]
        self.widgets["boundary_y"].value = defaults["boundary_y"]
        self.widgets["boundary_z"].value = defaults["boundary_z"]
        self.widgets["neighbor_order_input"].value = defaults["neighbor_order"]
        self.widgets["seed_input"].value = ''
        self.widgets["lattice_dropdown"].value = 'Square' if defaults["lattice_type"] == 'Cartesian' else 'Hexagonal'
        self.widgets["offset_input"].value = defaults["offset"]
    
    def create_ui(self):
        return VBox([
            HTML("<b>Dimensions:</b>"),
            HBox([self.widgets["x_slider"], self.widgets["y_slider"], self.widgets["z_slider"]]),
            HTML("<b>Core Parameters:</b>"),
            self.widgets["steps_input"],
            self.widgets["anneal_input"],
            self.widgets["fluctuation_slider"],
            self.widgets["flunct_fn_dropdown"],
            HTML("<b>Boundaries:</b>"),
            HBox([self.widgets["boundary_x"], self.widgets["boundary_y"], self.widgets["boundary_z"]]),
            HTML("<b>Advanced:</b>"),
            self.widgets["neighbor_order_input"],
            self.widgets["lattice_dropdown"],
            self.widgets["offset_input"],
            self.widgets["seed_input"],
            self.widgets["reset_button"]
        ], layout=Layout(padding='10px'))

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
        self.widgets = {}  # Initialize FIRST
        self.saved_values = self.load_saved_values()
        #self.celltype_entries = self.saved_values.get("CellType", DEFAULTS["CellType"].copy())
        #self.constraints = self.saved_values.get("Constraints", DEFAULTS["Constraints"].copy())
        
        self.metadata = Metadata(
            num_processors=self.saved_values["Metadata"]["num_processors"],
            debug_output_frequency=self.saved_values["Metadata"]["debug_output_frequency"]
        )

        self.potts_core = PottsCore(
            **self.saved_values["PottsCore"]
        )

        cell_type_data = self.saved_values.get("CellType", DEFAULTS["CellType"].copy())
        self.cell_type_plugin = CellTypePlugin()
        for entry in cell_type_data:
            self.cell_type_plugin.cell_type_append(
                entry["Cell Type"],
                frozen=entry.get("freeze", False)
            )
        
        # Create metadata widgets immediately
        # self.create_metadata_widgets()
        
        # Initialize widgets
        self.potts_widget = PottsWidget(self.potts_core.spec_dict)
        self.celltype_widget = CellTypeWidget(
            {"Cell type:": ct[0], "freeze": ct[2]}
            for ct in self.cell_type_plugin.spec_dict["cell_types"]
        )
        """self.constraints_widget = ConstraintsWidget(
            
        )"""
        
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
            value=self.metadata.debug_output_frequency, min=1,
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
            return DEFAULTS.copy()
        except (json.JSONDecodeError, IOError):
            print("JSON file is corrupted or inaccessible. Resetting to defaults.")
            return DEFAULTS.copy()
    
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
            "Constraints": self.constraints_widget.get_config() #to be connect to plugin
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
    # Map UI property names to PottsCore attributes
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
    
    # Get the actual property name in PottsCore
    core_property = property_map.get(property_name)
    if not core_property:
        return
    
    # Special handling for certain properties
    if property_name == "lattice_dropdown":
        # Convert UI dropdown value to core lattice type
        value = 'Cartesian' if value == 'Square' else 'Hexagonal'
    
    elif property_name == "seed_input":
        # Handle random seed (empty string becomes None)
        value = int(value) if value.strip() else None
    
    # Update the PottsCore object
    try:
        setattr(self.potts_core, core_property, value)
        self.save_to_json()
    except Exception as e:
        print(f"Error updating PottsCore: {e}")
    
    def create_ui(self):
        # Create tabs
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
        
        # Display everything
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
        """Reset Potts tab to defaults"""
        self.potts_widget.reset()
        # Update core object with reset values
        for prop, value in DEFAULTS["PottsCore"].items():
            setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        """Reset CellType tab to defaults"""
        self.celltype_widget.reset()
        # Rebuild core cell types
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
        """Reset Constraints tab to defaults"""
        self.constraints_widget.reset()
        self.save_to_json()

    def reset_metadata_tab(self):
        """Reset Metadata tab to defaults"""
        self.widgets["num_proc"].value = DEFAULTS["Metadata"]["num_processors"]
        self.widgets["debug_freq"].value = DEFAULTS["Metadata"]["debug_output_frequency"]
        self.metadata.num_processors = DEFAULTS["Metadata"]["num_processors"]
        self.metadata.debug_output_frequency = DEFAULTS["Metadata"]["debug_output_frequency"]
        self.save_to_json()

    def reset_all(self):
        """Reset all tabs to defaults"""
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_constraints_tab()
        self.reset_metadata_tab()

# Main execution
if __name__ == "__main__":
    ui = SpecificationSetupUI()