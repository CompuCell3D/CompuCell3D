import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output
)
from IPython.display import display
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin, BoundaryPixelTrackerPlugin, CellTypePlugin, 
    ChemotaxisPlugin, ContactPlugin, CurvaturePlugin, 
    ExternalPotentialPlugin, FocalPointPlasticityPlugin,
    LengthConstraintPlugin, PixelTrackerPlugin, SecretionPlugin, 
    SurfacePlugin, VolumePlugin
)

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Get default values from class constructors using .spec_dict
def get_defaults():
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        "CellType": [{"Cell type": "Medium", "id": 0, "freeze": False}], # I think this error is because actually the default for celltype is Medium, 0, False and that applies to cell type, cell type id, and freeze. 
        "Plugins": {
            "AdhesionFlexPlugin": AdhesionFlexPlugin().spec_dict,
            "BoundaryPixelTrackerPlugin": BoundaryPixelTrackerPlugin().spec_dict,
            "ChemotaxisPlugin": ChemotaxisPlugin().spec_dict,
            "ContactPlugin": ContactPlugin().spec_dict,
            "CurvaturePlugin": CurvaturePlugin().spec_dict,
            "ExternalPotentialPlugin": ExternalPotentialPlugin().spec_dict,
            "FocalPointPlasticityPlugin": FocalPointPlasticityPlugin().spec_dict,
            "LengthConstraintPlugin": LengthConstraintPlugin().spec_dict,
            "PixelTrackerPlugin": PixelTrackerPlugin().spec_dict,
            "SecretionPlugin": SecretionPlugin().spec_dict,
            "VolumePlugin": VolumePlugin().spec_dict,
            "SurfacePlugin": SurfacePlugin().spec_dict
        }
    }

DEFAULTS = get_defaults()

class PluginWidget:
    def __init__(self, plugin_name, plugin_class, saved_values, cell_types):
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.create_widgets(saved_values or self.default_instance.spec_dict)
        
    def create_widgets(self, saved_values):
        # Main checkbox with plugin name
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )
        
        # Container for plugin-specific widgets
        self.widgets["config_container"] = VBox([], layout=Layout(
            padding='10px 0 0 20px',
            display='none' if not saved_values else 'block'
        ))
        
        # Create plugin-specific widgets
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets(saved_values)
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets(saved_values)
        elif self.plugin_name == "AdhesionFlexPlugin":
            self.create_adhesion_widgets(saved_values)
        elif self.plugin_name == "ContactPlugin":
            self.create_contact_widgets(saved_values)
        elif self.plugin_name == "ChemotaxisPlugin":
            self.create_chemotaxis_widgets(saved_values)
        elif self.plugin_name == "BoundaryPixelTrackerPlugin":
            self.create_boundary_tracker_widgets(saved_values)
        # Add other plugins as needed...
        
        # Set up active toggle
        self.widgets["active"].observe(self.toggle_config_visibility, names='value')
    
    def toggle_config_visibility(self, change):
        self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'
    
    # VolumePlugin widgets
    def create_volume_widgets(self, saved_values):
        """Widgets for VolumePlugin"""
        params = saved_values.get("params", [])
        
        if not params:
            default_params = VolumePlugin().spec_dict.get("params", [])
            params = default_params or [{
                "CellType": "Medium",
                "target_volume": 25.0,
                "lambda_volume": 2.0
            }]
        
        rows = []
        for param in params:
            cell_type_value = param["CellType"]
            if cell_type_value not in self.cell_types:
                cell_type_value = self.cell_types[0] if self.cell_types else "Medium"
            
            row = {}
            row["cell_type"] = widgets.Dropdown(
                options=self.cell_types,
                value=cell_type_value,
                description='Cell Type:',
                style={'description_width': 'initial'}
            )
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                min=1.0,
                description='Target Volume:',
                style={'description_width': 'initial'}
            )
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                min=0.0,
                description='Lambda Volume:',
                style={'description_width': 'initial'}
            )
            rows.append(row)
        
        self.widgets["rows"] = rows
        
        # Add button for new row
        self.widgets["add_btn"] = widgets.Button(
            description="Add Constraint",
            button_style='success'
        )
        
        # Define add_volume_row as a method
        def add_volume_row(_):
            default_params = VolumePlugin().spec_dict.get("params", [])
            default_values = default_params[0] if default_params else {
                "CellType": "Medium",
                "target_volume": 25.0,
                "lambda_volume": 2.0
            }
            
            new_row = {
                "cell_type": widgets.Dropdown(
                    options=self.cell_types,
                    value=default_values["CellType"],
                    description='Cell Type:',
                    style={'description_width': 'initial'}
                ),
                "target_volume": widgets.FloatText(
                    value=default_values["target_volume"],
                    min=1.0,
                    description='Target Volume:',
                    style={'description_width': 'initial'}
                ),
                "lambda_volume": widgets.FloatText(
                    value=default_values["lambda_volume"],
                    min=0.0,
                    description='Lambda Volume:',
                    style={'description_width': 'initial'}
                )
            }
            
            self.widgets["rows"].append(new_row)
            self.update_volume_ui()
        
        # Connect the button handler
        self.widgets["add_btn"].on_click(add_volume_row)
        
        # Build UI
        self.update_volume_ui()
    
    def update_volume_ui(self):
        """Update the UI after row changes"""
        row_widgets = []
        for i, row in enumerate(self.widgets["rows"]):
            remove_btn = widgets.Button(
                description="Remove", 
                button_style='danger',
                layout=Layout(width='100px')
            )
            
            def make_remove_handler(index):
                def handler(_):
                    del self.widgets["rows"][index]
                    self.update_volume_ui()
                return handler
            
            remove_btn.on_click(make_remove_handler(i))
            
            row_widgets.append(HBox([
                row["cell_type"], 
                row["target_volume"], 
                row["lambda_volume"],
                remove_btn
            ]))
        
        self.widgets["config_container"].children = [
            *row_widgets,
            self.widgets["add_btn"]
        ]
    
    # SurfacePlugin widgets
    def create_surface_widgets(self, saved_values):
        """Widgets for SurfacePlugin"""
        params = saved_values.get("params", [])
        
        if not params:
            default_params = SurfacePlugin().spec_dict.get("params", [])
            params = default_params or [{
                "CellType": "Medium",
                "target_surface": 100.0,
                "lambda_surface": 0.5
            }]
        
        rows = []
        for param in params:
            cell_type_value = param["CellType"]
            if cell_type_value not in self.cell_types:
                cell_type_value = self.cell_types[0] if self.cell_types else "Medium"
            
            row = {}
            row["cell_type"] = widgets.Dropdown(
                options=self.cell_types,
                value=cell_type_value,
                description='Cell Type:',
                style={'description_width': 'initial'}
            )
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                min=1.0,
                description='Target Surface:',
                style={'description_width': 'initial'}
            )
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                min=0.0,
                description='Lambda Surface:',
                style={'description_width': 'initial'}
            )
            rows.append(row)
        
        self.widgets["rows"] = rows
        
        # Add button for new row
        self.widgets["add_btn"] = widgets.Button(
            description="Add Constraint",
            button_style='success'
        )
        
        # Define add_surface_row as a method
        def add_surface_row(_):
            default_params = SurfacePlugin().spec_dict.get("params", [])
            default_values = default_params[0] if default_params else {
                "CellType": "Medium",
                "target_surface": 100.0,
                "lambda_surface": 0.5
            }
            
            new_row = {
                "cell_type": widgets.Dropdown(
                    options=self.cell_types,
                    value=default_values["CellType"],
                    description='Cell Type:',
                    style={'description_width': 'initial'}
                ),
                "target_surface": widgets.FloatText(
                    value=default_values["target_surface"],
                    min=1.0,
                    description='Target Surface:',
                    style={'description_width': 'initial'}
                ),
                "lambda_surface": widgets.FloatText(
                    value=default_values["lambda_surface"],
                    min=0.0,
                    description='Lambda Surface:',
                    style={'description_width': 'initial'}
                )
            }
            
            self.widgets["rows"].append(new_row)
            self.update_surface_ui()
        
        # Connect the button handler
        self.widgets["add_btn"].on_click(add_surface_row)
        
        # Build UI
        self.update_surface_ui()
    
    def update_surface_ui(self):
        """Update the UI after row changes"""
        row_widgets = []
        for i, row in enumerate(self.widgets["rows"]):
            remove_btn = widgets.Button(
                description="Remove", 
                button_style='danger',
                layout=Layout(width='100px')
            )
            
            def make_remove_handler(index):
                def handler(_):
                    del self.widgets["rows"][index]
                    self.update_surface_ui()
                return handler
            
            remove_btn.on_click(make_remove_handler(i))
            
            row_widgets.append(HBox([
                row["cell_type"], 
                row["target_surface"], 
                row["lambda_surface"],
                remove_btn
            ]))
        
        self.widgets["config_container"].children = [
            *row_widgets,
            self.widgets["add_btn"]
        ]
    
    # AdhesionFlexPlugin widgets
    def create_adhesion_widgets(self, saved_values):
        """Widgets for AdhesionFlexPlugin"""
        # Get default values from class
        defaults = AdhesionFlexPlugin().spec_dict
        
        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            min=1, max=10, 
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["max_distance"] = widgets.BoundedIntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            min=1, max=100, 
            description='Max Distance:',
            style={'description_width': 'initial'}
        )
        
        # Build UI
        self.widgets["config_container"].children = [
            HBox([
                self.widgets["neighbor_order"],
                self.widgets["max_distance"]
            ])
        ]
    
    # ContactPlugin widgets
    def create_contact_widgets(self, saved_values):
        """Widgets for ContactPlugin"""
        # Get default values from class
        defaults = ContactPlugin().spec_dict
        
        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 1)),
            min=1, max=10, 
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        
        # Build UI
        self.widgets["config_container"].children = [
            self.widgets["neighbor_order"]
        ]
    
    # ChemotaxisPlugin widgets
    def create_chemotaxis_widgets(self, saved_values):
        """Widgets for ChemotaxisPlugin"""
        # Get default values from class
        defaults = ChemotaxisPlugin().spec_dict
        
        self.widgets["field"] = widgets.Text(
            value=saved_values.get("field", defaults.get("field", "chemoattractant")),
            description='Field Name:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_val"] = widgets.FloatText(
            value=saved_values.get("lambda", defaults.get("lambda", 100.0)),
            min=0.0,
            description='Lambda Value:',
            style={'description_width': 'initial'}
        )
        
        # Build UI
        self.widgets["config_container"].children = [
            HBox([
                self.widgets["field"],
                self.widgets["lambda_val"]
            ])
        ]
    
    # BoundaryPixelTrackerPlugin widgets
    def create_boundary_tracker_widgets(self, saved_values):
        """Widgets for BoundaryPixelTrackerPlugin"""
        # Get default values from class
        defaults = BoundaryPixelTrackerPlugin().spec_dict
        
        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            min=1, max=10, 
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        
        # Build UI
        self.widgets["config_container"].children = [
            self.widgets["neighbor_order"]
        ]
    
    def create_ui(self):
        return VBox([
            self.widgets["active"],
            self.widgets["config_container"]
        ])
    
    def get_config(self):
        if not self.widgets["active"].value:
            return None
            
        config = {}
        
        # Handle plugins with rows (Volume, Surface)
        if "rows" in self.widgets:
            params = []
            for row in self.widgets["rows"]:
                if self.plugin_name == "VolumePlugin":
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_volume": row["target_volume"].value,
                        "lambda_volume": row["lambda_volume"].value
                    })
                elif self.plugin_name == "SurfacePlugin":
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_surface": row["target_surface"].value,
                        "lambda_surface": row["lambda_surface"].value
                    })
            config["params"] = params
        
        # Add other plugin-specific values
        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"]:
                config[key] = widget.value
        
        # Validate configuration using plugin class
        try:
            plugin_instance = self.plugin_class()
            for k, v in config.items():
                setattr(plugin_instance, k, v)
            plugin_instance.validate()
            return config
        except Exception as e:
            print(f"Validation error for {self.plugin_name}: {str(e)}")
            return None
        
    def reset(self):
        self.widgets["active"].value = False
        self.widgets["config_container"].layout.display = 'none'
        default = self.default_instance.spec_dict
        
        # Reset to plugin defaults
        if "rows" in self.widgets:
            self.widgets["rows"] = []
            if "params" in default:
                for param in default["params"]:
                    if self.plugin_name == "VolumePlugin":
                        self.add_volume_row(param)
                    elif self.plugin_name == "SurfacePlugin":
                        self.add_surface_row(param)
        
        # Reset other widgets
        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"] and key in default:
                widget.value = default[key]

class PluginsTab:
    def __init__(self, saved_plugins, cell_types):
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
        self.create_widgets(saved_plugins or DEFAULTS["Plugins"])
        
    def create_widgets(self, saved_plugins):
        # Create tabs for plugin categories
        self.widgets["tabs"] = widgets.Tab()
        
        # Create categories
        behavior_plugins = []
        constraint_plugins = []
        tracker_plugins = []
        other_plugins = []
        
        # Plugin classes mapping
        plugin_classes = {
            "AdhesionFlexPlugin": AdhesionFlexPlugin,
            "BoundaryPixelTrackerPlugin": BoundaryPixelTrackerPlugin,
            "ChemotaxisPlugin": ChemotaxisPlugin,
            "ContactPlugin": ContactPlugin,
            "CurvaturePlugin": CurvaturePlugin,
            "ExternalPotentialPlugin": ExternalPotentialPlugin,
            "FocalPointPlasticityPlugin": FocalPointPlasticityPlugin,
            "LengthConstraintPlugin": LengthConstraintPlugin,
            "PixelTrackerPlugin": PixelTrackerPlugin,
            "SecretionPlugin": SecretionPlugin,
            "VolumePlugin": VolumePlugin,
            "SurfacePlugin": SurfacePlugin
        }
        
        for plugin_name, plugin_class in plugin_classes.items():
            plugin_values = saved_plugins.get(plugin_name, DEFAULTS["Plugins"].get(plugin_name, {}))
            plugin_widget = PluginWidget(plugin_name, plugin_class, plugin_values, self.cell_types)
            self.plugin_widgets[plugin_name] = plugin_widget
            
            # Categorize
            if plugin_name in ["VolumePlugin", "SurfacePlugin", "LengthConstraintPlugin"]:
                constraint_plugins.append(plugin_widget.create_ui())
            elif plugin_name in ["AdhesionFlexPlugin", "ContactPlugin", "ChemotaxisPlugin", 
                                "CurvaturePlugin", "ExternalPotentialPlugin", 
                                "FocalPointPlasticityPlugin"]:
                behavior_plugins.append(plugin_widget.create_ui())
            elif plugin_name in ["BoundaryPixelTrackerPlugin", "PixelTrackerPlugin"]:
                tracker_plugins.append(plugin_widget.create_ui())
            else:
                other_plugins.append(plugin_widget.create_ui())
        
        # Create tab content
        tab_children = []
        tab_titles = []
        
        if behavior_plugins:
            tab_children.append(VBox(behavior_plugins))
            tab_titles.append("Cell Behavior")
        if constraint_plugins:
            tab_children.append(VBox(constraint_plugins))
            tab_titles.append("Constraints")
        if tracker_plugins:
            tab_children.append(VBox(tracker_plugins))
            tab_titles.append("Trackers")
        if other_plugins:
            tab_children.append(VBox(other_plugins))
            tab_titles.append("Other Plugins")
        
        self.widgets["tabs"].children = tab_children
        for i, title in enumerate(tab_titles):
            self.widgets["tabs"].set_title(i, title)
            
        self.widgets["reset_button"] = Button(
            description="Reset Plugins",
            button_style='warning'
        )
        
    def get_config(self):
        config = {}
        for plugin_name, widget in self.plugin_widgets.items():
            plugin_config = widget.get_config()
            if plugin_config is not None:
                config[plugin_name] = plugin_config
        return config
        
    def reset(self):
        for widget in self.plugin_widgets.values():
            widget.reset()
            
    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        for widget in self.plugin_widgets.values():
            if "rows" in widget.widgets:
                for row in widget.widgets["rows"]:
                    if "cell_type" in row:
                        row["cell_type"].options = cell_types
            
    def create_ui(self):
        return VBox([
            self.widgets["tabs"],
            self.widgets["reset_button"]
        ])


class PottsWidget:
    def __init__(self, saved_values):
        self.widgets = {}
        self.defaults = PottsCore().spec_dict
        self.create_widgets(saved_values or self.defaults)
        
    def create_widgets(self, saved_values):
        # Dimension inputs
        self.widgets["dim_x"] = widgets.IntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, description='X Dimension:',
            style={'description_width': 'initial'}
        )
        self.widgets["dim_y"] = widgets.IntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, description='Y Dimension:',
            style={'description_width': 'initial'}
        )
        self.widgets["dim_z"] = widgets.IntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, description='Z Dimension:',
            style={'description_width': 'initial'}
        )
        
        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=1, description='MC Steps:',
            style={'description_width': 'initial'}
        )
        self.widgets["fluctuation_amplitude"] = widgets.FloatText(
            value=saved_values.get("fluctuation_amplitude", self.defaults["fluctuation_amplitude"]),
            min=0.0, description='Fluctuation Amplitude:',
            style={'description_width': 'initial'}
        )
        
        # Boundaries
        boundary_options = ['NoFlux', 'Periodic']
        self.widgets["boundary_x"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_x", self.defaults["boundary_x"]),
            description='X Boundary:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_y"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_y", self.defaults["boundary_y"]),
            description='Y Boundary:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_z"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_z", self.defaults["boundary_z"]),
            description='Z Boundary:',
            style={'description_width': 'initial'}
        )
        
        # Advanced settings
        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", self.defaults["neighbor_order"]),
            min=1, max=20, description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["lattice_type"] = widgets.Dropdown(
            options=['Cartesian', 'Hexagonal'],
            value=saved_values.get("lattice_type", self.defaults["lattice_type"]),
            description='Lattice Type:',
            style={'description_width': 'initial'}
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
            "fluctuation_amplitude": self.widgets["fluctuation_amplitude"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order"].value,
            "lattice_type": self.widgets["lattice_type"].value
        }
    
    def reset(self):
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                widget.value = self.defaults[key]
    
    def create_ui(self):
        return VBox([
            HTML("<h4>Potts Core Parameters</h4>"),
            HBox([
                self.widgets["dim_x"],
                self.widgets["dim_y"],
                self.widgets["dim_z"]
            ]),
            HBox([
                self.widgets["steps"],
                self.widgets["fluctuation_amplitude"]
            ]),
            HTML("<b>Boundary Conditions:</b>"),
            HBox([
                self.widgets["boundary_x"],
                self.widgets["boundary_y"],
                self.widgets["boundary_z"]
            ]),
            HTML("<b>Advanced Settings:</b>"),
            HBox([
                self.widgets["neighbor_order"],
                self.widgets["lattice_type"]
            ]),
            self.widgets["reset_button"]
        ])


class CellTypeWidget:
    def __init__(self, saved_entries):
        self.next_id = 0
        self.celltype_entries =[]

        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("freeze", False))
            else:
                self.add_entry(entry, False)
        
        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", False)

        # self.celltype_entries = saved_entries or DEFAULTS["CellType"].copy()
        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()
    
    def add_entry(self, name, freeze):
        self.celltype_entries.append({
            "Cell type": name,
            "id": self.next_id,
            "freeze": freeze
        })
        self.next_id += 1

    def create_widgets(self):
        # Display area for current cell types
        self.widgets["display_box"] = VBox(
            layout=Layout(padding='10px')
        )
        
        # Input widgets
        self.widgets["name"] = Text(
            placeholder="Cell type name",
            description="Name:",
            style={'description_width': 'initial'}
        )
        self.widgets["freeze"] = Checkbox(
            value=False,
            description="Freeze",
            indent=False
        )
        self.widgets["add_button"] = Button(
            description="Add Cell Type",
            button_style="success"
        )
        
        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Cell Types",
            button_style='warning'
        )
    
    def setup_event_handlers(self):
        self.widgets["add_button"].on_click(self.on_add_clicked)
        self.widgets["reset_button"].on_click(self.reset)
    
    def on_add_clicked(self, _):
        name = self.widgets["name"].value.strip()
        
        if not name:
            self.widgets["name"].placeholder = "Please enter a name!"
            return
        
        # Check for duplicates
        if any(entry["Cell type"] == name for entry in self.celltype_entries):
            self.widgets["name"].value = ""
            self.widgets["name"].placeholder = f"{name} already exists!"
            return
        
        self.celltype_entries.append({
            "Cell type": name, 
            "freeze": self.widgets["freeze"].value
        })
        
        self.update_celltype_display()
        self.widgets["name"].value = ""
        self.widgets["name"].placeholder = "Cell type name"
    
    def update_celltype_display(self):
        items = []
        for i, entry in enumerate(self.celltype_entries):
            label = Label(f"{entry['Cell type']} {'(frozen)' if entry.get('freeze', False) else ''}")
            remove_btn = Button(
                description="Remove", 
                button_style='danger',
                layout=Layout(width='100px')
            )
            
            def make_remove_handler(index):
                def handler(_):
                    del self.celltype_entries[index]
                    self.update_celltype_display()
                return handler
            
            remove_btn.on_click(make_remove_handler(i))
            items.append(HBox(
                [label, remove_btn], 
                layout=Layout(margin='0 0 5px 0')
            ))
        
        self.widgets["display_box"].children = items
    
    def get_config(self):
        return self.celltype_entries.copy()
    
    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]
    
    def reset(self, _=None):
        self.celltype_entries = []
        self.next_id = 0
        for entry in DEFAULTS["CellType"]:
            self.add_entry(entry["Cell type"], entry.get("freeze", False))
        self.update_celltype_display()
    
    def create_ui(self):
        return VBox([
            HTML("<h4>Cell Types</h4>"),
            self.widgets["display_box"],
            HBox([
                self.widgets["name"],
                self.widgets["freeze"],
                self.widgets["add_button"]
            ]),
            self.widgets["reset_button"]
        ])


class SpecificationSetupUI:
    def __init__(self):
        self.widgets = {}
        self.saved_values = self.load_saved_values()
        
        # Initialize core components with defaults
        self.metadata = Metadata()
        self.potts_core = PottsCore()
        self.cell_type_plugin = CellTypePlugin()
        
        # Apply saved values if available
        self.apply_saved_values()
        
        # Initialize widgets
        self.create_metadata_widgets()
        self.potts_widget = PottsWidget(self.saved_values.get("PottsCore"))
        self.celltype_widget = CellTypeWidget(self.saved_values.get("CellType"))
        self.plugins_tab = PluginsTab(
            self.saved_values.get("Plugins", {}),
            self.celltype_widget.get_cell_type_names()
        )

        # Create the UI
        self.create_ui()
        self.setup_event_handlers()

    def apply_saved_values(self):
        """Apply saved values to core components"""
        # Metadata
        if "Metadata" in self.saved_values:
            for key, value in self.saved_values["Metadata"].items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
        
        # Potts Core
        if "PottsCore" in self.saved_values:
            for key, value in self.saved_values["PottsCore"].items():
                if hasattr(self.potts_core, key):
                    setattr(self.potts_core, key, value)
        
        # Cell Types
        if "CellType" in self.saved_values:
            for entry in self.saved_values["CellType"]:
                if isinstance(entry, dict):
                    self.cell_type_plugin.cell_type_append(
                        entry["Cell type"],
                        type_id=entry.get("id", len(self.cell_type_plugin.cell_types)),
                        frozen=entry.get("freeze", False)
                    )
                else:
                    # Old format (string only)
                    self.cell_type_plugin.cell_type_append(entry)

    def create_metadata_widgets(self):
        """Metadata widgets"""
        self.widgets["num_processors"] = widgets.IntText(
            value=self.metadata.num_processors,
            min=1,
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_output_frequency"] = widgets.IntText(
            value=self.metadata.debug_output_frequency,
            min=0,
            description='Debug Output Frequency:',
            style={'description_width': 'initial'}
        )
        self.widgets["reset_metadata_button"] = Button(
            description="Reset Metadata",
            button_style='warning'
        )
        self.widgets["reset_button"] = Button(
            description="Reset All to Defaults",
            button_style='danger'
        )

    def load_saved_values(self):
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, 'r') as f:
                    return json.load(f)
            return json.loads(json.dumps(DEFAULTS))
        except (json.JSONDecodeError, IOError):
            return json.loads(json.dumps(DEFAULTS))

    def current_config(self):
        return {
            "Metadata": self.metadata.spec_dict,
            "PottsCore": self.potts_core.spec_dict,
            "CellType": self.celltype_widget.get_config(),
            "Plugins": self.plugins_tab.get_config()
        }

    def save_to_json(self, _=None):
        try:
            with open(SAVE_FILE, 'w') as f:
                json.dump(self.current_config(), f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def setup_event_handlers(self):
        # Metadata handlers
        self.widgets["num_processors"].observe(
            lambda change: self.update_metadata('num_processors', change.new), 
            names='value'
        )
        self.widgets["debug_output_frequency"].observe(
            lambda change: self.update_metadata('debug_output_frequency', change.new),
            names='value'
        )

        # Connect reset handlers
        self.potts_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_potts_tab()
        )
        self.celltype_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_celltype_tab()
        )
        self.plugins_tab.widgets["reset_button"].on_click(
            lambda _: self.save_to_json()
            # lambda _: self.reset_plugins_tab()
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
            if hasattr(widget, 'observe') and name != "reset_button":
                widget.observe(
                    lambda change, prop=name: self.update_potts_core(prop, change.new),
                    names='value'
                )

        # CellType handlers
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.update_cell_types()
        )

        # Auto-save on changes
        for widget in [self.widgets["num_processors"], self.widgets["debug_output_frequency"]]:
            widget.observe(lambda _: self.save_to_json(), names='value')

    def update_metadata(self, property_name, value):
        setattr(self.metadata, property_name, value)
        self.save_to_json()

    def update_potts_core(self, property_name, value):
        if hasattr(self.potts_core, property_name):
            setattr(self.potts_core, property_name, value)
            self.save_to_json()

    def create_ui(self):
        tabs = Tab()
        tabs.children = [
            self.create_metadata_tab(),
            self.potts_widget.create_ui(),
            self.celltype_widget.create_ui(),
            self.plugins_tab.create_ui()
        ]
        tabs.set_title(0, 'Metadata')
        tabs.set_title(1, 'Potts Core')
        tabs.set_title(2, 'Cell Types')
        tabs.set_title(3, 'Plugins')
        
        display(VBox([
            tabs,
            HBox([
                self.widgets["reset_button"],
                Button(
                    description="Save Configuration",
                    button_style='success',
                    on_click=self.save_to_json
                )
            ])
        ]))

    def create_metadata_tab(self):
        return VBox([
            HTML("<h4>Simulation Metadata</h4>"),
            self.widgets["num_processors"],
            self.widgets["debug_output_frequency"],
            self.widgets["reset_metadata_button"]
        ])

    def reset_potts_tab(self):
        self.potts_widget.reset()
        for prop, value in PottsCore().spec_dict.items():
            if hasattr(self.potts_core, prop):
                setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        self.celltype_widget.reset()
        self.cell_type_plugin = CellTypePlugin()
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

    def reset_plugins_tab(self):
        self.plugins_tab.reset()
        self.save_to_json()

    def reset_metadata_tab(self):
        self.widgets["num_processors"].value = Metadata().num_processors
        self.widgets["debug_output_frequency"].value = Metadata().debug_output_frequency
        self.metadata = Metadata()
        self.save_to_json()

    def reset_all(self):
        self.reset_metadata_tab()
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_plugins_tab()

    def update_cell_types(self):
        self.cell_type_plugin = CellTypePlugin()
        for entry in self.celltype_widget.celltype_entries:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                type_id=entry["id"],
                frozen=entry["freeze"]
            )
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

