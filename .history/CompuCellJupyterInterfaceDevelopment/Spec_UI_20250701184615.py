import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML
)
from IPython.display import display

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
    # Keep this class unchanged from previous implementation
    # ... [full PottsWidget class implementation] ...

class CellTypeWidget:
    # Keep this class unchanged from previous implementation
    # ... [full CellTypeWidget class implementation] ...

class ConstraintsWidget:
    # Keep this class unchanged from previous implementation
    # ... [full ConstraintsWidget class implementation] ...

class SimulationSetupUI:
    def __init__(self):
        self.widgets = {}  # Initialize FIRST
        self.saved_values = self.load_saved_values()
        self.celltype_entries = self.saved_values.get("CellType", DEFAULTS["CellType"].copy())
        self.constraints = self.saved_values.get("Constraints", DEFAULTS["Constraints"].copy())
        
        # Create metadata widgets immediately
        self.create_metadata_widgets()
        
        # Initialize widgets
        self.potts_widget = PottsWidget(self.saved_values["PottsCore"])
        self.celltype_widget = CellTypeWidget(self.celltype_entries)
        self.constraints_widget = ConstraintsWidget(self.constraints, self.celltype_widget)
        
        # Create the UI
        self.create_ui()
        self.setup_event_handlers()
    
    def create_metadata_widgets(self):
        """Create metadata widgets first"""
        self.widgets["num_proc"] = widgets.IntText(
            value=self.saved_values["Metadata"]["num_processors"], min=1,
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_freq"] = widgets.IntText(
            value=self.saved_values["Metadata"]["debug_output_frequency"], min=1,
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
            print("⚠️ JSON file is corrupted or inaccessible. Resetting to defaults.")
            return DEFAULTS.copy()
    
    def current_config(self):
        return {
            "Metadata": {
                "num_processors": self.widgets["num_proc"].value,
                "debug_output_frequency": self.widgets["debug_freq"].value
            },
            "PottsCore": self.potts_widget.get_config(),
            "CellType": self.celltype_widget.get_config(),
            "Constraints": self.constraints_widget.get_config()
        }
    
    def save_to_json(self, _=None):
        with open(SAVE_FILE, 'w') as f:
            json.dump(self.current_config(), f, indent=4)
    
    def setup_event_handlers(self):
        # Connect reset handlers
        self.potts_widget.widgets["reset_button"].on_click(lambda _: self.potts_widget.reset())
        self.celltype_widget.widgets["reset_button"].on_click(lambda _: self.celltype_widget.reset())
        self.constraints_widget.widgets["reset_button"].on_click(lambda _: self.constraints_widget.reset())
        self.widgets["reset_metadata_button"].on_click(self.reset_metadata)
        
        # Connect cell type changes to constraints widget
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.constraints_widget.update_cell_type_options()
        )
        
        # Save triggers
        self.widgets["num_proc"].observe(self.save_to_json, names='value')
        self.widgets["debug_freq"].observe(self.save_to_json, names='value')
        
        # Save triggers for Potts widgets
        for w in self.potts_widget.widgets.values():
            if hasattr(w, 'observe'):
                w.observe(self.save_to_json, names='value')
        
        # Save triggers for CellType
        self.celltype_widget.widgets["add_button"].on_click(self.save_to_json)
        
        # Save triggers for Constraints
        self.constraints_widget.widgets["add_button"].on_click(self.save_to_json)
        
        # Global reset
        self.widgets["reset_button"].on_click(self.reset_all)
    
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
    
    def reset_metadata(self, _):
        self.widgets["num_proc"].value = DEFAULTS["Metadata"]["num_processors"]
        self.widgets["debug_freq"].value = DEFAULTS["Metadata"]["debug_output_frequency"]
        self.save_to_json()
    
    def reset_all(self, _):
        self.potts_widget.reset()
        self.celltype_widget.reset()
        self.constraints_widget.reset()
        self.reset_metadata(None)
        self.save_to_json()

# Main execution
if __name__ == "__main__":
    ui = SimulationSetupUI()