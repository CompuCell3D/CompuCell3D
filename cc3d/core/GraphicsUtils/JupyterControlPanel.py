from IPython.display import display
import ipywidgets as widgets


class JupyterControlPanel():
    """
    Helper class for creating consistent ipywidget interface
    """
    def __init__(self):
        self.values = {}
        self.widgets = {}
        self.tabs = None

    def add_toggle(self, name, value=False, callback=None, show=True):
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


    def add_int(self, name, value=0, min=0, max=100, step=1, callback=None, show=True):
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


    def add_select(self, name, options=[], value=None, callback=None, show=True):
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

    def add_tab(self, tab_name, widget_names):
        if not self.tabs:
            self.tabs = widgets.Tab()
            display(self.tabs)
        tab_widgets = [self.widgets[n] for n in widget_names]
        box = widgets.VBox(tab_widgets)
        children = list(self.tabs.children)
        children.append(box)
        self.tabs.children = children
        i = len(children) - 1
        self.tabs.set_title(i, tab_name)

        
