import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output, GridspecLayout, Box,
    IntText, SelectMultiple, GridBox
)
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin, BoundaryPixelTrackerPlugin, CellTypePlugin,
    ChemotaxisPlugin, ContactPlugin, CurvaturePlugin,
    ExternalPotentialPlugin, FocalPointPlasticityPlugin,
    LengthConstraintPlugin, PixelTrackerPlugin, SecretionPlugin,
    SurfacePlugin, VolumePlugin, BlobInitializer
)
from cc3d.core.PyCoreSpecs import SpecValueCheckError
from IPython.display import display as ipy_display

from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Get default values from class constructors using .spec_dict
def get_defaults():
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        "CellType": CellTypePlugin().spec_dict,
        "Plugins": {
            "AdhesionFlexPlugin": {},
            "BoundaryPixelTrackerPlugin": {},
            "ChemotaxisPlugin": {},
            "ContactPlugin": {},
            "CurvaturePlugin": {},
            "ExternalPotentialPlugin": {},
            "FocalPointPlasticityPlugin": {},
            "LengthConstraintPlugin": {},
            "PixelTrackerPlugin": {},
            "SecretionPlugin": {},
            "VolumePlugin": {},
            "SurfacePlugin": {}
        }
    }

DEFAULTS = get_defaults()

class PluginWidget:
    def __init__(self, plugin_name, plugin_class, saved_values, cell_types, parent_ui=None, potts_neighbor_order=None):
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.output = Output()
        self.param_cache = {}
        self.parent_ui = parent_ui
        self.potts_neighbor_order = potts_neighbor_order
        self.create_widgets(saved_values if saved_values else {})

    def create_widgets(self, saved_values):
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )

        self.widgets["config_container"] = VBox([], layout=Layout(
            padding='0',
            display='none'  
        ))

        def save_on_toggle(change):
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()
        self.widgets["active"].observe(save_on_toggle, names='value')

        # Plugins with custom UIs
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
        elif self.plugin_name in [
            "CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin",
            "LengthConstraintPlugin", "PixelTrackerPlugin", "SecretionPlugin"
        ]:
            # Not implemented plugins: message will be shown/hidden by toggle_config_visibility
            self.widgets["config_container"].children = []
            self.widgets["config_container"].layout.display = 'none'
        # else: leave as is for future plugins

        self.widgets["active"].observe(self.toggle_config_visibility, names='value')

    def toggle_config_visibility(self, change):
        # For not implemented plugins, show/hide the message based on enable state
        if self.plugin_name in [
            "CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin",
            "LengthConstraintPlugin", "PixelTrackerPlugin", "SecretionPlugin"
        ]:
            if change['new']:
                self.widgets["config_container"].children = [HTML(f"<b style='color: #b00'>{self.plugin_name} not implemented yet</b>")]
                self.widgets["config_container"].layout.display = 'block'
            else:
                self.widgets["config_container"].children = []
                self.widgets["config_container"].layout.display = 'none'
        else:
            if self.plugin_name in ["CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin"]:
                self.widgets["config_container"].layout.display = 'block'
            else:
                self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'

    # VolumePlugin widgets
    def create_volume_widgets(self, saved_values):
        default_params = VolumePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            # For Medium, set target and lambda to 0 and disable
            if ct == "Medium":
                param = {"CellType": ct, "target_volume": 0.0, "lambda_volume": 0.0}
                disabled = True
            else:
                param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                    "CellType": ct,
                    "target_volume": 25.0,
                    "lambda_volume": 2.0
                }
                disabled = False
                
            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            def make_cache_updater(cell_type, field):
                def updater(change):
                    if cell_type not in self.param_cache:
                        self.param_cache[cell_type] = {
                            "CellType": cell_type,
                            "target_volume": row["target_volume"].value,
                            "lambda_volume": row["lambda_volume"].value
                        }
                    self.param_cache[cell_type][field] = change["new"]
                    if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                        self.parent_ui.save_to_json()
                return updater
            row["target_volume"].observe(make_cache_updater(ct, "target_volume"), names='value')
            row["lambda_volume"].observe(make_cache_updater(ct, "lambda_volume"), names='value')
            rows.append(row)
        self.widgets["rows"] = rows
        self.update_volume_ui()

    def update_volume_ui(self):
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Volume:", layout=widgets.Layout(width='100px')),
                row["target_volume"],
                widgets.Label("Lambda Volume:", layout=widgets.Layout(width='100px')),
                row["lambda_volume"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    # SurfacePlugin widgets
    def create_surface_widgets(self, saved_values):
        default_params = SurfacePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            # For Medium, set target and lambda to 0 and disable
            if ct == "Medium":
                param = {"CellType": ct, "target_surface": 0.0, "lambda_surface": 0.0}
                disabled = True
            else:
                param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                    "CellType": ct,
                    "target_surface": 100.0,
                    "lambda_surface": 0.5
                }
                disabled = False
                
            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            def make_cache_updater(cell_type, field):
                def updater(change):
                    if cell_type not in self.param_cache:
                        self.param_cache[cell_type] = {
                            "CellType": cell_type,
                            "target_surface": row["target_surface"].value,
                            "lambda_surface": row["lambda_surface"].value
                        }
                    self.param_cache[cell_type][field] = change["new"]
                    if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                        self.parent_ui.save_to_json()
                return updater
            row["target_surface"].observe(make_cache_updater(ct, "target_surface"), names='value')
            row["lambda_surface"].observe(make_cache_updater(ct, "lambda_surface"), names='value')
            rows.append(row)
        self.widgets["rows"] = rows
        self.update_surface_ui()

    def update_surface_ui(self):
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Surface:", layout=widgets.Layout(width='100px')),
                row["target_surface"],
                widgets.Label("Lambda Surface:", layout=widgets.Layout(width='100px')),
                row["lambda_surface"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    # AdhesionFlexPlugin widgets - use Potts neighbor order
    def create_adhesion_widgets(self, saved_values):
        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 2
        
        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )
        
        defaults = AdhesionFlexPlugin().spec_dict
        self.widgets["max_distance"] = widgets.IntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            min=1,
            description='Max Distance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["max_distance_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        max_distance_box = VBox([self.widgets["max_distance"], self.widgets["max_distance_error"]])
        container = VBox([
            self.widgets["neighbor_order_display"],
            HBox([max_distance_box])
        ])
        self.widgets["config_container"].children = [container]

    # ContactPlugin widgets - use Potts neighbor order
    def create_contact_widgets(self, saved_values):
        from ipywidgets import VBox, HBox, Dropdown, FloatText, Button, IntText, HTML, Layout

        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 1
        
        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )

        # Get cell types from the parent UI
        cell_types = self.cell_types if self.cell_types else []

        # Table for contact energies
        self.widgets["contact_rows"] = []
        self.widgets["contact_table"] = VBox()  # <-- Create this before add_row or update_table
        energies = saved_values.get("energies", {})

        # Helper to add a row
        def add_row(type_1=None, type_2=None, energy=None):
            type_1 = type_1 or (cell_types[0] if cell_types else "")
            type_2 = type_2 or (cell_types[0] if cell_types else "")
            energy = energy if energy is not None else 10.0

            dd1 = Dropdown(options=cell_types, value=type_1, layout=Layout(width='120px'))
            dd2 = Dropdown(options=cell_types, value=type_2, layout=Layout(width='120px'))
            en = FloatText(value=energy, layout=Layout(width='100px'))
            rm = Button(description="Remove", button_style='danger', layout=Layout(width='80px'))

            row = HBox([dd1, dd2, en, rm])
            self.widgets["contact_rows"].append((row, dd1, dd2, en, rm))

            def on_remove(_):
                self.widgets["contact_rows"].remove((row, dd1, dd2, en, rm))
                update_table()
                if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                    self.parent_ui.save_to_json()

            rm.on_click(on_remove)

            # Observe changes to trigger save
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                dd1.observe(lambda change: self.parent_ui.save_to_json(), names='value')
                dd2.observe(lambda change: self.parent_ui.save_to_json(), names='value')
                en.observe(lambda change: self.parent_ui.save_to_json(), names='value')

            update_table()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()

        # Helper to update the table display
        def update_table():
            self.widgets["contact_table"].children = [r[0] for r in self.widgets["contact_rows"]]

        # Add from saved values
        for t1, t2dict in energies.items():
            for t2, param in t2dict.items():
                add_row(type_1=t1, type_2=t2, energy=param.get("energy", 10.0))

        # Add row button
        add_btn = Button(description="Add Row", button_style='success', layout=Layout(width='100px'))
        def on_add_btn(_):
            add_row()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()
        add_btn.on_click(on_add_btn)

        # Table container (already created above)
        update_table()

        # Compose the UI
        container = VBox([
            self.widgets["neighbor_order_display"],
            HTML("<b>Contact Energies</b>"),
            self.widgets["contact_table"],
            add_btn
        ])
        self.widgets["config_container"].children = [container]

    # ChemotaxisPlugin widgets
    def create_chemotaxis_widgets(self, saved_values):
        defaults = ChemotaxisPlugin().spec_dict
        self.widgets["field"] = widgets.Text(
            value=saved_values.get("field", defaults.get("field", "chemoattractant")),
            description='Field Name:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["field_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["lambda_val"] = widgets.FloatText(
            value=saved_values.get("lambda", defaults.get("lambda", 100.0)),
            min=0.0,
            description='Lambda Value:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["lambda_val_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        field_box = VBox([self.widgets["field"], self.widgets["field_error"]])
        lambda_box = VBox([self.widgets["lambda_val"], self.widgets["lambda_val_error"]])
        container = VBox([
            HBox([field_box, lambda_box])
        ])
        self.widgets["config_container"].children = [container]

    # BoundaryPixelTrackerPlugin widgets - use Potts neighbor order
    def create_boundary_tracker_widgets(self, saved_values):
        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 2
        
        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )
        neighbor_box = VBox([self.widgets["neighbor_order_display"]])
        self.widgets["config_container"].children = [neighbor_box]

    def create_curvature_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_external_potential_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_focal_point_plasticity_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_ui(self):
        if self.plugin_name in ["AdhesionFlexPlugin", "ContactPlugin", "ChemotaxisPlugin"]:
            return VBox([
                self.widgets["active"],
                self.widgets["config_container"],
                self.output
            ])
        else:
            return VBox([
                self.widgets["active"],
                self.widgets["config_container"]
            ])

    def get_config(self):
        if not self.widgets["active"].value:
            return None

        config = {}
        if "rows" in self.widgets:
            params = []
            if self.plugin_name == "VolumePlugin":
                for row in self.widgets["rows"]:
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_volume": row["target_volume"].value,
                        "lambda_volume": row["lambda_volume"].value
                    })
                config["params"] = params
            elif self.plugin_name == "SurfacePlugin":
                for row in self.widgets["rows"]:
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_surface": row["target_surface"].value,
                        "lambda_surface": row["lambda_surface"].value
                    })
                config["params"] = params

        # ContactPlugin config
        if self.plugin_name == "ContactPlugin":
            # Use neighbor order from PottsCore
            neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 1
            config = {"neighbor_order": neighbor_order, "energies": {}}
            for _, dd1, dd2, en, _ in self.widgets["contact_rows"]:
                t1, t2 = dd1.value, dd2.value
                if t1 not in config["energies"]:
                    config["energies"][t1] = {}
                config["energies"][t1][t2] = {"type_1": t1, "type_2": t2, "energy": en.value}
            return config

        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn", "contact_rows", "contact_table", "neighbor_order_display"] and hasattr(widget, 'value'):
                config[key] = widget.value

        try:
            plugin_instance = self.plugin_class()
            for k, v in config.items():
                setattr(plugin_instance, k, v)
            if self.parent_ui and hasattr(self.parent_ui, 'potts_core') and hasattr(self.parent_ui, 'cell_type_plugin'):
                plugin_instance.validate(self.parent_ui.potts_core, self.parent_ui.cell_type_plugin)
            else:
                plugin_instance.validate()
            return config
        except Exception as e:
            print(f"Validation error for {self.plugin_name}: {str(e)}")
            return None

    def reset(self):
        self.widgets["active"].value = False
        self.widgets["config_container"].layout.display = 'none'
        default = self.default_instance.spec_dict

        # Clear parameter cache
        self.param_cache = {}

        if self.plugin_name == "ContactPlugin":
            # Clear contact rows and table
            self.widgets["contact_rows"] = []
            if "contact_table" in self.widgets:
                self.widgets["contact_table"].children = []
        elif "rows" in self.widgets:
            self.widgets["rows"] = []
            if "params" in default:
                if self.plugin_name == "VolumePlugin":
                    self.create_volume_widgets(default)
                elif self.plugin_name == "SurfacePlugin":
                    self.create_surface_widgets(default)

        for key, widget in self.widgets.items():
            if key != "active" and key != "config_container" and key != "rows" and key != "add_btn" and key != "contact_rows" and key != "contact_table" and key in default:
                if hasattr(widget, 'value'):
                    widget.value = default[key]

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_volume": 25.0, "lambda_volume": 2.0}) for ct in cell_types]})
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_surface": 100.0, "lambda_surface": 0.5}) for ct in cell_types]})
        elif self.plugin_name == "ContactPlugin":
            # Update dropdown options for all rows
            for row_tuple in self.widgets.get("contact_rows", []):
                _, dd1, dd2, _, _ = row_tuple
                dd1.options = cell_types
                dd2.options = cell_types

class PluginsTab:
    def __init__(self, saved_plugins, cell_types, parent_ui=None, potts_neighbor_order=None):
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
        self.parent_ui = parent_ui
        self.potts_neighbor_order = potts_neighbor_order
        self.create_widgets(saved_plugins or DEFAULTS["Plugins"])

    def create_widgets(self, saved_plugins):
        self.widgets["tabs"] = widgets.Tab()
        behavior_plugins = []
        constraint_plugins = []
        tracker_plugins = []
        other_plugins = []

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
            plugin_widget = PluginWidget(
                plugin_name, 
                plugin_class, 
                plugin_values, 
                self.cell_types, 
                parent_ui=self.parent_ui,
                potts_neighbor_order=self.potts_neighbor_order
            )
            self.plugin_widgets[plugin_name] = plugin_widget

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

        # Connect reset button to save
        def on_reset_clicked(_):
            self.reset()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()

        self.widgets["reset_button"].on_click(on_reset_clicked)

    def get_config(self):
        config = {}
        for plugin_name, widget in self.plugin_widgets.items():
            plugin_config = widget.get_config()
            if plugin_config is not None:
                config[plugin_name] = plugin_config
        return config

    def reset(self, _=None):
        for widget in self.plugin_widgets.values():
            widget.reset()

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        for widget in self.plugin_widgets.values():
            widget.update_cell_types(cell_types)

    def create_ui(self):
        return VBox([
            self.widgets["tabs"],
            HBox([self.widgets["reset_button"]],
                 layout=Layout(justify_content='flex-start', margin='15px 0 0 0'))
        ], layout=Layout(padding='10px'))

class PottsWidget:
    def __init__(self, saved_values):
        self.widgets = {}
        self.defaults = PottsCore().spec_dict
        # Set defaults to 1 for dimensions
        self.defaults["dim_x"] = 1
        self.defaults["dim_y"] = 1
        self.defaults["dim_z"] = 1
        self.create_widgets(saved_values or self.defaults)

    def create_widgets(self, saved_values):
        # Dimension inputs with max=101
        self.widgets["dim_x"] = widgets.BoundedIntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, max=101, description='X Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_x_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_y"] = widgets.BoundedIntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, max=101, description='Y Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_y_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_z"] = widgets.BoundedIntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, max=101, description='Z Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_z_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=0, description='MC Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["steps_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Advanced settings
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

        # Random seed with activation checkbox
        self.widgets["use_random_seed"] = widgets.Checkbox(
            value=saved_values.get("random_seed") is not None,
            description='Use Random Seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["random_seed"] = widgets.BoundedIntText(
            value=saved_values.get("random_seed", 0),
            min=0,
            description='',
            disabled=not (saved_values.get("random_seed") is not None),
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Enable/disable based on checkbox
        def on_use_random_seed_change(change):
            self.widgets["random_seed"].disabled = not change.new
        self.widgets["use_random_seed"].observe(on_use_random_seed_change, names='value')

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
            "random_seed": self.widgets["random_seed"].value if self.widgets["use_random_seed"].value else None
        }

    def reset(self):
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                if key == "random_seed":
                    self.widgets["use_random_seed"].value = False
                    self.widgets["random_seed"].value = 0
                else:
                    widget.value = self.defaults[key]

    def create_ui(self):
        dimensions_row = HBox([
            VBox([self.widgets["dim_x"], self.widgets["dim_x_error"]]),
            VBox([self.widgets["dim_y"], self.widgets["dim_y_error"]]),
            VBox([self.widgets["dim_z"], self.widgets["dim_z_error"]])
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        dimensions_row.add_class('vbox-row-spacing')

        core_params_row = HBox([
            VBox([self.widgets["steps"], self.widgets["steps_error"]]),
            VBox([self.widgets["fluctuation_amplitude"], self.widgets["fluctuation_amplitude_error"]]),
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        core_params_row.add_class('vbox-row-spacing')

        core_params_row2 = HBox([
            VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]]),
            self.widgets["lattice_type"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        core_params_row2.add_class('vbox-row-spacing')

        boundaries_row = HBox([
            self.widgets["boundary_x"],
            self.widgets["boundary_y"],
            self.widgets["boundary_z"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        boundaries_row.add_class('vbox-row-spacing')

        advanced_row1 = HBox([
            VBox([self.widgets["anneal"], self.widgets["anneal_error"]]),
            self.widgets["fluctuation_amplitude_function"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row1.add_class('vbox-row-spacing')

        advanced_row2 = HBox([
            self.widgets["offset"],
            self.widgets["use_random_seed"],
            self.widgets["random_seed"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row2.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Potts Core Parameters</b>", layout=Layout(margin='0 0 10px 0')),
            dimensions_row,
            core_params_row,
            core_params_row2,
            HTML("<b>Boundary Conditions:</b>", layout=Layout(margin='10px 0 5px 0')),
            boundaries_row,
            HTML("<b>Advanced Settings:</b>", layout=Layout(margin='10px 0 5px 0')),
            advanced_row1,
            advanced_row2,
            HBox([self.widgets["reset_button"]], layout=Layout(margin='10px 0 0 0'))
        ], layout=Layout(align_items='flex-start', padding='10px'))

class CellTypeWidget:
    def __init__(self, saved_entries, on_change=None):
        self.on_change = on_change
        self.celltype_entries = []

        # Load saved entries or use defaults
        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("id", None), entry.get("freeze", False))
            else:
                self.add_entry(entry, None, False)

        # Ensure Medium always exists with ID 0
        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", 0, False)

        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()

    def add_entry(self, name, type_id=None, freeze=False):
        # For Medium, force ID to 0
        if name == "Medium":
            type_id = 0
        else:
            # Find the smallest available ID
            if type_id is None:
                used_ids = {entry["id"] for entry in self.celltype_entries}
                type_id = 1
                while type_id in used_ids:
                    type_id += 1

        # Create new entry
        self.celltype_entries.append({
            "Cell type": name,
            "id": type_id,
            "freeze": freeze
        })

        # Sort by ID to maintain order
        self.celltype_entries.sort(key=lambda x: x["id"])

    def create_widgets(self):
        self.widgets["display_box"] = VBox(layout=Layout(padding='10px'))
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
        if any(entry["Cell type"] == name for entry in self.celltype_entries):
            self.widgets["name"].value = ""
            self.widgets["name"].placeholder = f"{name} already exists!"
            return

        self.add_entry(name, None, self.widgets["freeze"].value)
        self.update_celltype_display()
        self.widgets["name"].value = ""
        self.widgets["name"].placeholder = "Cell type name"
        if self.on_change:
            self.on_change()

    def update_celltype_display(self):
        n = len(self.celltype_entries)
        if n == 0:
            self.widgets["display_box"].children = [HTML("<i>No cell types defined.</i>")]
            return

        row_border = '1px solid #e0e0e0'
        header_border = '2px solid #bdbdbd'
        header = [
            HTML(f"<b style='display:block; padding:2px 8px;'>ID</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Cell Type</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Freeze</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Remove</b>", layout=Layout(border=f'0 0 {header_border} 0'))
        ]
        grid = GridspecLayout(n + 1, 4, grid_gap="0px")
        for j, h in enumerate(header):
            grid[0, j] = h

        for i, entry in enumerate(self.celltype_entries):
            border_style = f'0 0 {row_border} 0' if i < n - 1 else '0'

            # ID column
            grid[i + 1, 0] = Label(str(entry['id']), layout=Layout(border=border_style, padding='2px 8px'))

            # Cell Type column
            grid[i + 1, 1] = Label(str(entry['Cell type']), layout=Layout(border=border_style, padding='2px 8px'))

            # Freeze column
            freeze_chk = Checkbox(
                value=entry.get('freeze', False),
                indent=False,
                layout=Layout(border=border_style, padding='2px 8px', width='auto')
            )

            # Handler for freeze checkbox
            def make_freeze_handler(idx):
                def handler(change):
                    self.celltype_entries[idx]['freeze'] = change['new']
                    if self.on_change:
                        self.on_change()
                return handler

            freeze_chk.observe(make_freeze_handler(i), names='value')
            grid[i + 1, 2] = freeze_chk

            # Remove column - disable for Medium
            if entry['Cell type'] == "Medium":
                remove_btn = Button(
                    description="Default",
                    button_style='',
                    disabled=True,
                    layout=Layout(width='100px', border=border_style, padding='2px 8px')
                )
            else:
                remove_btn = Button(
                    description="Remove",
                    button_style='danger',
                    layout=Layout(width='100px', border=border_style, padding='2px 8px')
                )
                # Handler for remove button
                def make_remove_handler(idx):
                    def handler(_):
                        # Skip Medium removal
                        if self.celltype_entries[idx]['Cell type'] == "Medium":
                            return
                        del self.celltype_entries[idx]
                        self.update_celltype_display()
                        if self.on_change:
                            self.on_change()
                    return handler
                remove_btn.on_click(make_remove_handler(i))

            grid[i + 1, 3] = remove_btn

        self.widgets["display_box"].children = [grid]

    def get_config(self):
        return self.celltype_entries.copy()

    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]

    def reset(self, _=None):
        self.celltype_entries = []
        self.add_entry("Medium", 0, False)
        self.update_celltype_display()

    def create_ui(self):
        input_row = HBox([
            self.widgets["name"],
            self.widgets["freeze"],
            self.widgets["add_button"]
        ], layout=Layout(justify_content='flex-start', margin='10px 0'))

        reset_button_box = HBox([self.widgets["reset_button"]],
                               layout=Layout(justify_content='flex-start', margin='10px 0'))

        return VBox([
            HTML("<b>Cell Types</b>", layout=Layout(margin='0 0 10px 0')),
            self.widgets["display_box"],
            input_row,
            reset_button_box
        ], layout=Layout(align_items='flex-start', padding='10px'))

class InitializerWidget:
    def __init__(self, saved_values=None, cell_types=None, parent_ui=None):
        self.cell_types = cell_types or []
        self.saved_values = saved_values or {}
        self.widgets = {}
        self.regions = []  # List of region dicts
        self.regions_box = VBox()
        self.add_region_btn = Button(description="Add Region", button_style="success")
        self.add_region_btn.on_click(self.add_region)
        self.parent_ui = parent_ui

        # Dropdown for initializer type
        self.initializer_type_dropdown = Dropdown(
            options=["BlobInitializer", "PIFInitializer", "UniformInitializer"],
            value=self.saved_values.get("type", "BlobInitializer"),
            description="Initializer Type:",
            style={'description_width': 'initial'},
        )
        self.initializer_type_dropdown.observe(self._on_initializer_type_change, names='value')

        # Load regions from saved_values if present and type is BlobInitializer
        if self.saved_values.get("type", "BlobInitializer") == "BlobInitializer" and self.saved_values.get("regions"):
            for region in self.saved_values["regions"]:
                self._add_region_from_saved(region)

        self.create_ui()

    def _on_initializer_type_change(self, change):
        self.update_regions_box()
        if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
            self.parent_ui.save_to_json()

    def _add_region_from_saved(self, region):
        region_dict = {
            "width": IntText(value=region["width"], description="Width:"),
            "radius": IntText(value=region["radius"], description="Radius:"),
            "center_x": IntText(value=region["center"][0], description="Center X:"),
            "center_y": IntText(value=region["center"][1], description="Center Y:"),
            "center_z": IntText(value=region["center"][2], description="Center Z:"),
            "cell_types": SelectMultiple(
                options=self.cell_types,
                value=tuple(region["cell_types"]),
                description="Cell Types:"
            ),
            "remove_btn": Button(description="Remove", button_style="danger"),
            "selection_note": HTML(
                value="<medium><em>Tip: Hold Ctrl/Cmd to select multiple cell types</em></medium>",
                layout=Layout(margin='0 0 0 10px')
            )
        }
        region_dict["remove_btn"].on_click(lambda btn, r=region_dict: self.remove_region(r))
        for key in ["width", "radius", "center_x", "center_y", "center_z"]:
            region_dict[key].observe(self._trigger_save, names='value')
        region_dict["cell_types"].observe(self._trigger_save, names='value')
        self.regions.append(region_dict)

    def _trigger_save(self, *_):
        if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
            self.parent_ui.save_to_json()

    def add_region(self, _=None):
        width = 5  # Manually assigned default value
        radius = 20
        center_x = 50
        center_y = 50
        center_z = 0
        selected_cell_types = self.cell_types.copy() if self.cell_types else []
        
        region = {
            "width": IntText(value=width, description="Width:"),
            "radius": IntText(value=radius, description="Radius:"),
            "center_x": IntText(value=center_x, description="Center X:"),
            "center_y": IntText(value=center_y, description="Center Y:"),
            "center_z": IntText(value=center_z, description="Center Z:"),
            "cell_types": SelectMultiple(
                options=self.cell_types,
                value=tuple(selected_cell_types),
                description="Cell Types:"
            ),
            "remove_btn": Button(description="Remove", button_style="danger"),
            # Add note about multi-selection
            "selection_note": HTML(
                value="<small><em>Tip: Hold Ctrl/Cmd to select multiple cell types</em></small>",
                layout=Layout(margin='0 0 0 10px')
            )
        }
        
        region["remove_btn"].on_click(lambda btn, r=region: self.remove_region(r))
        
        for key in ["width", "radius", "center_x", "center_y", "center_z"]:
            region[key].observe(self._trigger_save, names='value')
        
        region["cell_types"].observe(self._trigger_save, names='value')
        
        self.regions.append(region)
        self.update_regions_box()
        self._trigger_save()

    def remove_region(self, region):
        self.regions.remove(region)
        self.update_regions_box()
        self._trigger_save()

    def update_regions_box(self):
        selected_type = self.initializer_type_dropdown.value
        if selected_type == "BlobInitializer":
            region_vboxes = []
            for region in self.regions:
                vbox = VBox([
                    HBox([region["center_x"], region["center_y"], region["center_z"]]),
                    HBox([region["width"], region["radius"]]),
                    VBox([
                        region["cell_types"],
                        region["selection_note"]  
                    ]),
                    region["remove_btn"]
                ], layout=Layout(border="1px solid #ccc", margin="10px 0", padding="10px"))
                region_vboxes.append(vbox)
            self.regions_box.children = region_vboxes + [self.add_region_btn]
        else:
            msg = f"{selected_type} not implemented yet"
            self.regions_box.children = [HTML(f"<b style='color: #b00'>{msg}</b>")]

    def get_config(self):
        selected_type = self.initializer_type_dropdown.value
        if selected_type == "BlobInitializer":
            regions = []
            for region in self.regions:
                regions.append({
                    "width": region["width"].value,
                    "radius": region["radius"].value,
                    "center": (region["center_x"].value, region["center_y"].value, region["center_z"].value),
                    "cell_types": list(region["cell_types"].value)
                })
            return {
                "type": "BlobInitializer",
                "regions": regions
            }
        else:
            return {
                "type": selected_type
            }

    def create_ui(self):
        self.update_regions_box()
        self.widget = VBox([
            self.initializer_type_dropdown,
            Label("Initializer Configuration:"),
            self.regions_box
        ])

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        for region in self.regions:
            region["cell_types"].options = self.cell_types

    def get_widget(self):
        return self.widget

class SpecificationSetupUI:
    def __init__(self):
        self._initializing = True  # Flag to suppress save_to_json during init
        self.widgets = {}
        self.saved_values = self.load_saved_values()
        self.metadata = Metadata()
        self.potts_core = PottsCore()
        self.cell_type_plugin = CellTypePlugin()
        self.apply_saved_values()
        self.create_metadata_widgets()
        self.potts_widget = PottsWidget(self.saved_values.get("PottsCore"))
        self.celltype_widget = CellTypeWidget(
            self.saved_values.get("CellType"),
            on_change=self.cell_types_changed
        )
        self.plugins_tab = PluginsTab(
            self.saved_values.get("Plugins", {}),
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self,
            potts_neighbor_order=self.potts_widget.widgets["neighbor_order"]
        )
        self.initializer_widget = InitializerWidget(
            saved_values=self.saved_values.get("Initializer"),
            cell_types=self.celltype_widget.get_cell_type_names(),
            parent_ui=self
        )
        self.cc3d_sim = None
        self.visualization_output = widgets.Output()
        self.create_ui()
        self.setup_event_handlers()
        self._initializing = False  # Allow save_to_json after init

    def print_specs_summary(self):
        """Print summary of loaded specs for debugging"""
        print("Loaded Specifications:")
        print(f"Metadata: {self.metadata.spec_dict}")
        print(f"PottsCore: {self.potts_core.spec_dict}")
        print(f"Cell Types: {[ct['Cell type']] for ct in self.celltype_widget.get_config()}")
        
        active_plugins = [name for name, config in self.plugins_tab.get_config().items() if config]
        print(f"Active Plugins: {active_plugins}")
        
        init_config = self.initializer_widget.get_config()
        print(f"Initializer: {init_config['type']} with {len(init_config.get('regions', []))} regions")

    @property
    def specs(self):
        """
        Returns a list of all specification objects for the simulation.
        This includes Metadata, PottsCore, CellTypePlugin, enabled plugins, and initializer.
        """
        import copy
        import inspect
        print("Generating simulation specifications...")
        
        config = self.current_config()
        specs = []
        
        # Metadata
        specs.append(Metadata(**config["Metadata"]))
        
        # PottsCore (filter out invalid keys)
        pottscore_args = inspect.signature(PottsCore.__init__).parameters
        pottscore_config = {k: v for k, v in config["PottsCore"].items() if k in pottscore_args}
        specs.append(PottsCore(**pottscore_config))
        
        # CellTypePlugin
        cell_types = [entry["Cell type"] for entry in config["CellType"]]
        specs.append(CellTypePlugin(*cell_types))
        
        # VolumePlugin
        if "params" in config["Plugins"].get("VolumePlugin", {}):
            volume_plugin = VolumePlugin()
            for param in config["Plugins"]["VolumePlugin"]["params"]:
                volume_plugin.param_new(param["CellType"], param["target_volume"], param["lambda_volume"])
            specs.append(volume_plugin)
        
        # SurfacePlugin
        if "params" in config["Plugins"].get("SurfacePlugin", {}):
            surface_plugin = SurfacePlugin()
            for param in config["Plugins"]["SurfacePlugin"]["params"]:
                surface_plugin.param_new(param["CellType"], param["target_surface"], param["lambda_surface"])
            specs.append(surface_plugin)
        
        # ContactPlugin
        if "energies" in config["Plugins"].get("ContactPlugin", {}):
            cp_conf = config["Plugins"]["ContactPlugin"]
            contact_plugin = ContactPlugin(cp_conf.get("neighbor_order", 1))
            for t1, t2dict in cp_conf["energies"].items():
                for t2, param in t2dict.items():
                    contact_plugin.param_new(type_1=param["type_1"], type_2=param["type_2"], energy=param["energy"])
            specs.append(contact_plugin)
        
        # Other plugins
        for plugin_name in ["AdhesionFlexPlugin", "BoundaryPixelTrackerPlugin", "ChemotaxisPlugin"]:
            if plugin_name in config["Plugins"] and config["Plugins"][plugin_name]:
                plugin_class = globals()[plugin_name]
                plugin_config = config["Plugins"][plugin_name]
                plugin_args = inspect.signature(plugin_class.__init__).parameters
                filtered_config = {k: v for k, v in plugin_config.items() if k in plugin_args}
                specs.append(plugin_class(**filtered_config))
        
        # BlobInitializer
        if "Initializer" in config and config["Initializer"].get("type") == "BlobInitializer":
            blob_init = BlobInitializer()
            for region in config["Initializer"].get("regions", []):
                blob_init.region_new(
                    width=region["width"],
                    radius=region["radius"],
                    center=region["center"],
                    cell_types=region["cell_types"]
                )
            specs.append(blob_init)
        
        print(f"Generated {len(specs)} specification objects")
        return specs

    def cell_types_changed(self):
        self.update_plugin_cell_types()
        self.save_to_json()

    def update_plugin_cell_types(self):
        cell_types = self.celltype_widget.get_config()
        cell_type_names = [entry["Cell type"] for entry in cell_types]
        # Update VolumePlugin
        volume_widget = self.plugins_tab.plugin_widgets.get("VolumePlugin")
        if volume_widget:
            volume_widget.update_cell_types(cell_type_names)
        # Update SurfacePlugin
        surface_widget = self.plugins_tab.plugin_widgets.get("SurfacePlugin")
        if surface_widget:
            surface_widget.update_cell_types(cell_type_names)
        # Update ContactPlugin
        contact_widget = self.plugins_tab.plugin_widgets.get("ContactPlugin")
        if contact_widget:
            contact_widget.update_cell_types(cell_type_names)
        # Update BlobInitializer
        if hasattr(self.initializer_widget, "update_cell_types"):
            self.initializer_widget.update_cell_types(cell_type_names)

    def apply_saved_values(self):
        if "Metadata" in self.saved_values:
            for key, value in self.saved_values["Metadata"].items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
        if "PottsCore" in self.saved_values:
            for key, value in self.saved_values["PottsCore"].items():
                if hasattr(self.potts_core, key):
                    setattr(self.potts_core, key, value)
        if "CellType" in self.saved_values:
            for entry in self.saved_values["CellType"]:
                if isinstance(entry, dict):
                    cell_type_name = entry["Cell type"]
                    if cell_type_name == "Medium":
                        continue
                    self.cell_type_plugin.cell_type_append(
                        cell_type_name,
                        frozen=entry.get("freeze", False)
                    )

    def create_metadata_widgets(self):
        self.widgets["num_processors"] = widgets.IntText(
            value=self.metadata.num_processors,
            min=1,
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["num_processors_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["debug_output_frequency"] = widgets.IntText(
            value=self.metadata.debug_output_frequency,
            min=0,
            description='Debug Output Frequency:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_output_frequency_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["reset_metadata_button"] = Button(
            description="Reset Metadata",
            button_style='warning'
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
            "Plugins": self.plugins_tab.get_config(),
            "Initializer": self.initializer_widget.get_config()
        }

    def save_to_json(self, _=None):
        if getattr(self, '_initializing', False):
            return  # Don't save during initialization
        config = self.current_config()
        if config:
            with open(SAVE_FILE, 'w') as f:
                json.dump(config, f, indent=4)

    def setup_event_handlers(self):
        self.widgets["num_processors"].observe(
            lambda change: self.update_metadata('num_processors', change.new),
            names='value'
        )
        self.widgets["debug_output_frequency"].observe(
            lambda change: self.update_metadata('debug_output_frequency', change.new),
            names='value'
        )
        self.potts_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_potts_tab()
        )
        self.celltype_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_celltype_tab()
        )
        self.widgets["reset_metadata_button"].on_click(
            lambda _: self.reset_metadata_tab()
        )
        for name, widget in self.potts_widget.widgets.items():
            if hasattr(widget, 'observe') and name != "reset_button":
                widget.observe(
                    lambda change, prop=name: self.update_potts_core(prop, change.new),
                    names='value'
                )
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.update_cell_types()
        )
        for widget in [self.widgets["num_processors"], self.widgets["debug_output_frequency"]]:
            widget.observe(lambda _: self.save_to_json(), names='value')

    def update_metadata(self, property_name, value):
        try:
            setattr(self.metadata, property_name, value)
            self.save_to_json()
            self.clear_metadata_error(property_name)
        except Exception as e:
            self.show_metadata_error(property_name, str(e))

    def show_metadata_error(self, property_name, message):
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_metadata_error(self, property_name):
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def update_potts_core(self, property_name, value):
        try:
            setattr(self.potts_core, property_name, value)
            self.save_to_json()
            self.clear_potts_error(property_name)
        except Exception as e:
            self.show_potts_error(property_name, str(e))

    def show_potts_error(self, property_name, message):
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_potts_error(self, property_name):
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def run_and_visualize(self, _=None):
        from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
        from IPython.display import display
        import traceback
        import time
        
        # Get current configuration
        specs = self.specs

        self.visualization_output.clear_output()
        with self.visualization_output:
            try:
                print("Simulation still work in progress...")
                print("Initializing simulation...")
                
                # Initialize simulation service
                self.cc3d_sim = CC3DSimService()
                self.cc3d_sim.register_specs(specs)
                
                # Compile and run simulation
                self.cc3d_sim.run()
                self.cc3d_sim.init()
                
                # Start simulation without visualization first
                self.cc3d_sim.start()
                
                print("Simulation started. Preparing visualization...")
                
                # Create visualization widget
                viewer = self.cc3d_sim.visualize(plot_freq=10)
                
                # Display the viewer
                display(viewer)
                
                # Give it a moment to initialize
                time.sleep(1)
                
                # Call draw to ensure visualization updates
                viewer.draw()
                
                # Display run button
                run_button = self.cc3d_sim.jupyter_run_button()
                if run_button:
                    display(run_button)
                else:
                    print("Run button not available")
                    
                print("Simulation running. Use the run button to pause/resume.")
                
            except Exception as e:
                print("Error during simulation setup:")
                traceback.print_exc()

    def create_ui(self):
        tabs = Tab(layout=Layout(width='100%'))

        # Create tab containers
        metadata_tab = VBox([
            self.create_metadata_tab()
        ], layout=Layout(width='100%'))

        potts_tab = VBox([
            self.potts_widget.create_ui()
        ], layout=Layout(width='100%'))

        celltype_tab = VBox([
            self.celltype_widget.create_ui()
        ], layout=Layout(width='100%'))

        plugins_tab = VBox([
            self.plugins_tab.create_ui()
        ], layout=Layout(width='100%'))

        initializer_tab = VBox([
            self.initializer_widget.get_widget()
        ], layout=Layout(width='100%'))

        steppable_tab = VBox([
            HTML("<h3>Steppable Configuration</h3>"),
            HTML("<b style='color: #b00'>Steppable configuration is work in progress.</b style='color: #b00'>"),
            HTML("<b style='color: #b00'>This feature will be implemented in a future release.</b style='color: #b00'>")
        ], layout=Layout(width='100%'), padding='15px')


        tabs.children = [metadata_tab, potts_tab, celltype_tab, plugins_tab, initializer_tab, steppable_tab]
        tabs.set_title(0, 'Metadata')
        tabs.set_title(1, 'Potts Core')
        tabs.set_title(2, 'Cell Types')
        tabs.set_title(3, 'Plugins')
        tabs.set_title(4, 'Initializer')
        tabs.set_title(5, 'Steppable')

        run_button = Button(
            description="Run Simulation",
            button_style='success',
            layout=Layout(width='100%', margin='20px 0 10px 0')
        )
        run_button.on_click(self.run_and_visualize)

        run_container = VBox([
            run_button,
            self.visualization_output
        ], layout=Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            margin='20px 0 0 0'
        ))

        # Wrap in container for consistent styling
        container = VBox([
            tabs,
            # run_container
        ], layout=Layout(
            align_items='flex-start',
            width='100%',
            padding='15px')
        )

        container.add_class('widget-container')
        ipy_display(container)

    def create_metadata_tab(self):
        num_processors_box = VBox([
            self.widgets["num_processors"],
            self.widgets["num_processors_error"]
        ], layout=Layout(align_items='flex-start'))

        debug_frequency_box = VBox([
            self.widgets["debug_output_frequency"],
            self.widgets["debug_output_frequency_error"]
        ], layout=Layout(align_items='flex-start'))

        return VBox([
            HTML("<b>Simulation Metadata</b>", layout=Layout(margin='0 0 10px 0')),
            num_processors_box,
            debug_frequency_box,
            HBox([self.widgets["reset_metadata_button"]], layout=Layout(justify_content='flex-start', margin='10px 0'))
        ], layout=Layout(align_items='flex-start', padding='10px'))

    def reset_potts_tab(self):
        self.potts_widget.reset()
        for prop, value in PottsCore().spec_dict.items():
            if hasattr(self.potts_core, prop):
                setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        self.celltype_widget.reset()
        self.cell_type_plugin = CellTypePlugin()
        self.update_plugin_cell_types()
        self.save_to_json()

    def reset_plugins_tab(self):
        self.plugins_tab = PluginsTab(
            DEFAULTS["Plugins"],
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self,
            potts_neighbor_order=self.potts_widget.widgets["neighbor_order"]
        )
        self.save_to_json()

    def reset_metadata_tab(self):
        self.widgets["num_processors"].value = Metadata().num_processors
        self.widgets["debug_output_frequency"].value = Metadata().debug_output_frequency
        self.metadata = Metadata()
        self.save_to_json()

    def update_cell_types(self):
        self.cell_type_plugin = CellTypePlugin()
        for entry in self.celltype_widget.celltype_entries:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                frozen=entry["freeze"]
            )
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

class SimulationVisualizer:
    def __init__(self, specs):
        self.specs = specs
        self.cc3d_sim = None
    
    def show_initializer(self):
        """Visualize the initial cell configuration"""
        # Create temporary simulation to generate initial state
        temp_sim = CC3DSimService()
        temp_sim.register_specs(self.specs)
        temp_sim.run()
        temp_sim.init()  # Initialize but don't start simulation
        
        # Visualize at step 0 (initial state)
        viewer = temp_sim.visualize(plot_freq=1)
        viewer.show()  # Fixed: use show() instead of draw()
        return viewer
    
    def show_simulation(self):
        """Run and visualize the live simulation"""
        # Create and start simulation service
        self.cc3d_sim = CC3DSimService()
        self.cc3d_sim.register_specs(self.specs)
        self.cc3d_sim.run()
        self.cc3d_sim.init()
        self.cc3d_sim.start()
        
        # Create visualization widget
        viewer = self.cc3d_sim.visualize(plot_freq=10)
        viewer.show()  # Fixed: use show() instead of draw()
        
        # Add simulation controls
        run_button = self.cc3d_sim.jupyter_run_button()
        if run_button:
            display(run_button)
        
        return viewer
    
    def stop_simulation(self):
        """Stop the running simulation"""
        if self.cc3d_sim:
            self.cc3d_sim.stop()
            self.cc3d_sim = None