import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output, GridspecLayout, Box
)
from IPython.display import display, HTML as IPythonHTML
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS, CellTypePlugin
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin, BoundaryPixelTrackerPlugin,
    ChemotaxisPlugin, ContactPlugin, CurvaturePlugin,
    ExternalPotentialPlugin, FocalPointPlasticityPlugin,
    LengthConstraintPlugin, PixelTrackerPlugin, SecretionPlugin,
    SurfacePlugin, VolumePlugin
)
from cc3d.core.PyCoreSpecs import SpecValueCheckError

# Inject custom CSS for styling
css = '''
<style>
.vbox-row-spacing { margin-bottom: 10px; }
.custom-grid { grid-gap: 10px; }
</style>
'''

def apply_css():
    display(IPythonHTML(css))

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Build defaults directly from plugin spec_dict

def get_defaults():
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        # Direct raw spec_dict for CellType
        "CellType": CellTypePlugin().spec_dict,
        "Plugins": {name: {} for name in PLUGINS.keys()}
    }

DEFAULTS = get_defaults()

class SpecProperty:
    def __init__(self, plugin_obj, prop_name):
        self.plugin = plugin_obj
        self.name = prop_name

    def __get__(self, instance, owner):
        return getattr(self.plugin, self.name)

    def __set__(self, instance, value):
        try:
            setattr(self.plugin, self.name, value)
        except SpecValueCheckError as e:
            instance.error_output.clear_output()
            with instance.error_output:
                print(f"Invalid value for {self.name}: {e}")
            raise

class CellTypeWidget:
    def __init__(self):
        self.celltype_entries = []
        self.next_id = 0
        self.error_output = Output()
        self.reset()

    def reset(self, _=None):
        """
        Rebuild entries directly from CellTypePlugin spec_dict
        """
        plugin = CellTypePlugin()
        self.celltype_entries = []
        self.next_id = 0
        for name, type_id, frozen in plugin.spec_dict.get("cell_types", []):
            self.add_entry(name, type_id, frozen)
        self.update_display()

    def add_entry(self, name, id, freeze):
        entry = {
            "Cell type": Text(value=name),
            "id": BoundedIntText(value=id, min=0, max=999),
            "freeze": Checkbox(value=freeze)
        }
        # Validation hooks
        entry["id"].observe(lambda change: self.validate(id=change['new']), names='value')
        self.celltype_entries.append(entry)

    def validate(self, **kwargs):
        # Example validation: id must be unique
        ids = [e['id'].value for e in self.celltype_entries]
        if len(ids) != len(set(ids)):
            raise SpecValueCheckError("Cell type IDs must be unique.")

    def update_display(self):
        rows = []
        for e in self.celltype_entries:
            row = HBox([e['Cell type'], e['id'], e['freeze']], layout=Layout(justify_content='flex-start'))
            row.add_class('vbox-row-spacing')
            rows.append(row)
        display(VBox(rows + [self.error_output]))

class PluginWidget:
    def __init__(self, plugin_name):
        self.plugin_name = plugin_name
        self.plugin_obj = globals()[plugin_name]()
        self.widgets = {}
        self.error_output = Output()
        self.create_widgets()

    def create_widgets(self):
        spec = self.plugin_obj.spec_dict
        for prop, default in spec.items():
            if isinstance(default, int):
                w = BoundedIntText(value=default)
            elif isinstance(default, float):
                w = FloatText(value=default)
            elif isinstance(default, bool):
                w = Checkbox(value=default)
            else:
                continue
            # Attach validation
            prop_descr = SpecProperty(self.plugin_obj, prop)
            w.observe(lambda change, prop=prop_descr: setattr(prop, 'plugin', change['new']), names='value')
            self.widgets[prop] = w

    def display(self):
        items = []
        for prop, w in self.widgets.items():
            items.append(HBox([Label(prop), w], layout=Layout(justify_content='space-between')))
        display(VBox(items + [self.error_output]))

class SpecificationSetupUI:
    def __init__(self):
        apply_css()
        self.metadata_plugin = Metadata()
        self.potts_plugin = PottsCore()
        self.celltype_widget = CellTypeWidget()
        self.plugins_tab = Tab()
        self.build_ui()

    def build_ui(self):
        self.tabs = Tab()
        self.tabs.children = [
            self.build_metadata_tab(),
            self.build_potts_tab(),
            self.build_celltype_tab(),
            self.build_plugins_tab(),
            self.build_actions_tab()
        ]
        titles = ['Metadata', 'Potts', 'Cell Types', 'Plugins', 'Actions']
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)
        display(self.tabs)

    def build_metadata_tab(self):
        # metadata UI code unchanged
        return VBox([])

    def build_potts_tab(self):
        # potts UI code unchanged
        return VBox([])

    def build_celltype_tab(self):
        btn_reset = Button(description='Reset Cell Types')
        btn_reset.on_click(self.celltype_widget.reset)
        return VBox([btn_reset, self.celltype_widget.update_display])

    def build_plugins_tab(self):
        plugin_widgets = []
        for pname in PLUGINS.keys():
            pw = PluginWidget(pname)
            plugin_widgets.append(VBox([Label(pname), pw.display]))
        self.plugins_tab.children = plugin_widgets
        for i, pname in enumerate(PLUGINS.keys()):
            self.plugins_tab.set_title(i, pname)
        return self.plugins_tab

    def build_actions_tab(self):
        btn_save = Button(description='Save Setup')
        btn_save.on_click(self.save_to_json)
        btn_run = Button(description='Run Simulation')
        btn_run.on_click(lambda _: self.run_simulation())
        return VBox([HBox([btn_save, btn_run])])

    def save_to_json(self, _=None):
        data = {
            'Metadata': self.metadata_plugin.spec_dict,
            'PottsCore': self.potts_plugin.spec_dict,
            'CellType': CellTypePlugin().spec_dict,
            'Plugins': {p: globals()[p]().spec_dict for p in PLUGINS.keys()}
        }
        with open(SAVE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Configuration saved to {SAVE_FILE}")

    def run_simulation(self):
        print("Simulation is ready to run with current configuration")

# Instantiate and display the UI
ui = SpecificationSetupUI()
