import ipywidgets as widgets
from ipywidgets.widgets.widget_box import HBox, VBox

import base64
import hashlib
from typing import Callable
from IPython.display import HTML, display


class DownloadButton(widgets.Button):
    """
    Download button with dynamic content
    The content is generated using a callback when the button is clicked.
    """

    def __init__(self, filename: str, contents: Callable[[], str], **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.filename = filename
        self.contents = contents
        self.on_click(self.__on_click)

    def __on_click(self, b):
        contents: bytes = self.contents().encode('utf-8')
        b64 = base64.b64encode(contents)
        payload = b64.decode()
        digest = hashlib.md5(contents).hexdigest()  # bypass browser cache
        id = f'dl_{digest}'

        display(HTML(f"""
		<html>
			<body>
			<a id="{id}" download="{self.filename}" href="data:text/csv;base64,{payload}" download> </a>
			<script>
				(function download() {{
					document.getElementById('{id}').click();
				}})()
			</script>
			</body>
		</html>
		"""))


class JupyterWidgetInterface:
    """
    Helper class for creating consistent ipywidget interface
    """
    def __init__(self):
        self.data = {}
        self.tabs = None

    def set_disabled(self, keyname, value=True):
        self.data[keyname]['disabled'] = value
        self.data[keyname]['widget'].disabled = value

    def set_all_disabled(self, value=True):
        for w in self.data.values():
            # if not value and w['disabled']:
            if value or not w['disabled']:
                w['widget'].disabled = value

    def update_values(self, data):
        for key in self.data.keys():
            if key in data:
                if not isinstance(self.data[key]['widget'], (VBox, HBox)):
                    new_value = data[key]
                    self.data[key]['value'] = new_value
                    self.data[key]['widget'].value = new_value

    def add_grid(self, description, rows=1, cols=1, value=True, callback=None, name_callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        datavalue = [[value for j in range(cols)] for i in range(rows)]
        
        def toggle_handler(change):
            pos = change['owner'].description.split()[0].split(',')
            row = int(pos[0])
            col = int(pos[1])

            if change['name'] == 'value':
                self.data[keyname]['value'][row][col] = change['new']
                self._try_cb(keyname, callback)

        # call callback once to initialize
        callback(keyname, [ [value]*cols for i in range(rows)] )

        hboxes = []
        for i in range(rows):
            buttons = []
            for j in range(cols):
                button = widgets.ToggleButton(value=value, description=f"{i},{j}{' '+name_callback(i,j) if name_callback else ''}")
                button.observe(toggle_handler)
                buttons.append(button)
            hboxes.append( HBox(buttons) )
        
        label = widgets.Label(value=description)
        vbox = VBox([label] + hboxes, layout=widgets.Layout(width='20%'))

        self._add_data(keyname, datavalue, vbox)

        if show:
            display(vbox)
        return vbox

    def add_toggle(self, description, value=False, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        button = widgets.Checkbox(value=value, description=description)

        def toggle_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        button.observe(toggle_handler)
        self._add_data(keyname, value, button)

        if show:
            display(button)
        return button

    def add_button(self, description, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description

        button = widgets.Button(description=description)

        button.on_click(callback)
        self._add_data(keyname, None, button)

        if show:
            display(button)
        return button

    def add_download_button(self, description, filename, filecontents, keyname=None, show=False):
        if keyname is None:
            keyname = description

        button = DownloadButton(filename, filecontents, description=description)

        self._add_data(keyname, None, button)

        if show:
            display(button)
        return button

    def add_upload_button(self, description, accept='', multiple=False, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description

        button = widgets.FileUpload(accept=accept, multiple=multiple, description=description)

        def upload_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        button.observe(upload_handler)
        self._add_data(keyname, None, button)

        if show:
            display(button)
        return button

    def add_int(self, description, value=0, min=0, max=100, step=1, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description

        slider = widgets.IntSlider(value, min, max, step, description=description)

        def int_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        slider.observe(int_handler)
        self._add_data(keyname, value, slider)

        if show:
            display(slider)
        return slider

    def add_float(self, description, value=0, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        if value is None:
            value = 0

        floattext = widgets.FloatText(value, description=description)

        def float_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        floattext.observe(float_handler)
        self._add_data(keyname, value, floattext)

        if show:
            display(floattext)
        return floattext

    def add_range(self, description, value=None, minf=0, maxf=1, step=0.01, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        if value is None:
            value = [0, 1]

        slider = widgets.FloatRangeSlider(value=value, min=minf, max=maxf, step=step, description=description, continuous_update=False, readout=True)

        def range_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        slider.observe(range_handler)
        self._add_data(keyname, value, slider)

        if show:
            display(slider)
        return slider

    def add_select(self, description, options=None, value=None, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        if options is None:
            options = []

        select = widgets.Dropdown(options=options, value=value, description=description)

        def select_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        select.observe(select_handler)
        self._add_data(keyname, value, select)

        if show:
            display(select)
        return select

    def add_multiselect(self, description, options=None, value=None, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        if options is None:
            options = []

        value = tuple(value)
        select = widgets.SelectMultiple(options=options, value=value, description=description)

        def select_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        select.observe(select_handler)
        self._add_data(keyname, value, select)

        if show:
            display(select)
        return select

    def add_color(self, description, value='#ffffff', callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description
        # style = {'description_width': 'initial'}
        style = {'description_width': '200px'}

        picker = widgets.ColorPicker(value=value, description=description, concise=True, style=style)

        def color_handler(change):
            if change['name'] == 'value':
                self.data[keyname]['value'] = change['new']
                self._try_cb(keyname, callback)

        picker.observe(color_handler)
        self._add_data(keyname, value, picker)

        if show:
            display(picker)
        return picker

    def add_text(self, description, value=None, callback=None, keyname=None, show=False):
        if keyname is None:
            keyname = description

        label = widgets.Label(value=description)
        text = widgets.Text(value=value)
        submit = widgets.Button(description='Save')
        box = widgets.HBox([label, text, submit])

        def text_handler(button):
            self._try_cb(text.value)

        submit.on_click(text_handler)
        self._add_data(keyname, value, box)

        if show:
            display(box)
        return box

    def add_accordion(self, keyname, widget_objects, section_names, show=False):
        accordion = widgets.Accordion(children=widget_objects)
        for i, sn in enumerate(section_names):
            accordion.set_title(i, sn)

        self._add_data(keyname, None, accordion)

        if show:
            display(accordion)
        return accordion

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

        columns = []
        for wn in widget_names:
            tab_widgets = [self.data[n]['widget'] for n in wn]
            columns.append(widgets.VBox(tab_widgets))
        children = list(self.tabs.children)
        box = widgets.HBox(columns)
        children.append(box)
        self.tabs.children = children
        i = len(children) - 1
        self.tabs.set_title(i, tab_name)

    def add_listener(self, keyname, callback):
        self.data[keyname]['listeners'].append(callback)

    def _add_data(self, keyname, value, widget):
        self.data[keyname] = dict(value=value, widget=widget, listeners=[], disabled=False)

    def _try_cb(self, keyname, cb):
        self._call_listeners(keyname)
        if cb:
            cb(keyname, self.data[keyname]['value'])

    def _call_listeners(self, keyname):
        for l in self.data[keyname]['listeners']:
            l(keyname, self.data[keyname]['value'])

    @property
    def values(self):
        vals = {}
        for keyname, d in self.data.items():
            vals[keyname] = d['value']
        return vals
