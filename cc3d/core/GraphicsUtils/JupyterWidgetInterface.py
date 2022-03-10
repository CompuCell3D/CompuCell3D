from IPython.display import display
import ipywidgets as widgets
from ipywidgets.widgets.widget_box import HBox

class JupyterWidgetInterface():
    """
    Helper class for creating consistent ipywidget interface
    """
    def __init__(self):
        self.values = {}
        self.widgets = {}
        self.tabs = None


    def set_disabled(self, value=True):
        for w in self.widgets.values():
            w.disabled = value


    def add_toggle(self, name, value=False, callback=None, show=False):
        self.values[name] = value
        button = widgets.Checkbox(value=value, description=name)
        self.widgets[name] = button

        def toggle_handler(change):
            if change['name'] == 'value':
                self.values[name] = change['new']
                if callback:
                    callback(self.values[name])

        button.observe(toggle_handler)
        if show:
            display(button)
        return button


    def add_int(self, name, value=0, min=0, max=100, step=1, callback=None, show=False):
        self.values[name] = value
        slider = widgets.IntSlider(value, min, max, step, description=name)
        self.widgets[name] = slider 

        def int_handler(change):
            if change['name'] == 'value':
                self.values[name] = change['new']
                if callback:
                    callback(self.values[name])

        slider.observe(int_handler)
        if show:
            display(slider)
        return slider


    def add_select(self, name, options=[], value=None, callback=None, show=False):
        self.values[name] = value
        select = widgets.Dropdown(options=options, value=value, description=name)
        self.widgets[name] = select

        def select_handler(change):
            if change['name'] == 'value':
                self.values[name] = change['new']
                if callback:
                    callback(self.values[name])

        select.observe(select_handler)
        if show:
            display(select)
        return select


    def add_multiselect(self, name, options=[], value=None, callback=None, show=False):
        value = tuple(value)
        self.values[name] = value
        select = widgets.SelectMultiple(options=options, value=value, description=name)
        self.widgets[name] = select

        def select_handler(change):
            if change['name'] == 'value':
                self.values[name] = change['new']
                if callback:
                    callback(self.values[name])

        select.observe(select_handler)
        if show:
            display(select)
        return select

    
    def add_color(self, name, value='#ffffff', callback=None, show=False):
        self.values[name] = value
        picker = widgets.ColorPicker(value=value, description=name, concise=True)
        self.widgets[name] = picker

        def color_handler(change):
            if change['name'] == 'value':
                self.values[name] = change['new']
                if callback:
                    callback(name, self.values[name])

        picker.observe(color_handler)
        if show:
            display(picker)
        return picker


    def add_text(self, name, value=None, callback=None, show=False):
        self.values[name] = value
        label = widgets.Label(value=name)
        text = widgets.Text(value=value)
        submit = widgets.Button(description='Save')
        box = widgets.HBox([label, text, submit])
        self.widgets[name] = box

        def text_handler(button):
            if callback:
                callback(text.value)

        submit.on_click(text_handler)
        if show:
            display(box)
        return box


    def make_tab(self, tab_name, *widget_names):
        """
        Add widgets to a tab. Widgets should be added to panel beforehand.

        example:
        panel.add_toggle('widget1')
        panel.add_toggle('widget2')
        panel.add_toggle('widget3')
        panel.make_tab('tab_name', ['widget1'], ['widget2', 'widget3'])
        creates tab with 2 columns
        """
        if not self.tabs:
            self.tabs = widgets.Tab()
            display(self.tabs)

        columns = []
        for wn in widget_names:
            tab_widgets = [self.widgets[n] for n in wn]
            columns.append(widgets.VBox(tab_widgets))
        children = list(self.tabs.children)
        box = widgets.HBox(columns)
        children.append(box)
        self.tabs.children = children
        i = len(children) - 1
        self.tabs.set_title(i, tab_name)

        
