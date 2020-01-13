import re

from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLListPy
from cc3d.cpp.CC3DXML import *


def modules_dicts():
    return [{"Module": "CompuCell3D", "Attribute": "version"},
            {"Module": "Potts", "Attribute": None},
            {"Module": "Metadata", "Attribute": None},
            {"Module": "Plugin", "Attribute": "Name"},
            {"Module": "Steppable", "Attribute": "Type"}]


def pretty_text(text):

    no_space_chars = ['<', '/', '>']
    comment_beginning_chars = '<!--'

    text_f = ''

    text_split = text.split()
    for el in text_split:
        text_f += el
        if not any(el.endswith(no_space_char) for no_space_char in no_space_chars):
            text_f += ' '

    if text_f.__len__() >= 4 and text_f[0:3] == comment_beginning_chars:
        return text
    else:
        return text_f


def search_beginning_line(text: str):
    return re.search('^[\s\S]*<(.*)>[\s\S]*', pretty_text(text))


def search_closing_line(text: str):
    return re.search('^[\s\S]*</(.*)>[\s\S]*', pretty_text(text))


def is_beginning_module_line(text: str) -> bool:
    search = search_beginning_line(text)
    if search is None:
        return False

    contents = search.group(0)[1:-1]
    if contents[-1] == '/':
        contents = contents[0:-1]

    text_module = contents.split()[0]
    return any([text_module == module_entry["Module"] for module_entry in modules_dicts()])


def is_module_single_line(text: str) -> bool:
    search = search_beginning_line(text)
    if search is None:
        return False

    contents = search.group(0)[1:-1]
    if contents[-1] == '/':
        contents = contents[0:-1]
    else:
        return False

    text_module = contents.split()[0]
    return any([text_module == module_entry["Module"] for module_entry in modules_dicts()])


def is_closing_module_line(text: str) -> bool:
    search = search_closing_line(text)
    if search is None:
        return False

    contents = search.group(0)[2:-1]

    text_module = contents.split()[0]
    return any([text_module == module_entry["Module"] for module_entry in modules_dicts()])


def get_module_type(text: str):
    text_module = None
    if is_beginning_module_line(text=text):
        search = search_beginning_line(text)
        if search is None:
            return None
        contents = search.group(0)[1:-1]
        if contents[-1] == '/':
            contents = contents[0:-1]
        text_module = contents.split()[0]
    elif is_closing_module_line(text=text):
        search = search_closing_line(text)
        if search is None:
            return None
        contents = search.group(0)[2:-1]
        text_module = contents.split()[0]

    return text_module


def get_module_name(text: str):

    if not is_beginning_module_line(text=text):
        return None

    module_type = get_module_type(text=text)
    if module_type is None:
        return None
    elif module_type in ["CompuCell3D", "Potts", "Metadata"]:
        return module_type

    att = [val["Attribute"] for val in modules_dicts() if val["Module"] == module_type]

    if att.__len__() != 1:
        return None

    attribute = att[0]
    regex = re.compile('^[\s\S]*<[\s]*' + module_type + '[\s]*' + attribute +
                       '[\s]*=[\s]*"[\s]*(.*)[\s]*"[\s]*>')
    regex_sl = re.compile('^[\s\S]*<[\s]*' + module_type + '[\s]*' + attribute +
                          '[\s]*=[\s]*"[\s]*(.*)[\s]*"[\s]*/[\s]*>')

    search_sl = re.search(regex_sl, text)
    try:
        if search_sl is not None:
            return search_sl.group(1)
        else:
            search = re.search(regex, text)
            if search is not None:
                return search.group(1)
    except (AttributeError, IndexError):
        return None

    return None


def get_closing_line(text: str):
    if not is_beginning_module_line(text=text):
        return None
    elif re.search('^[\s\S]*<[\s]*(.*)[\s]*/>', text) is not None:
        return None

    module_type = get_module_type(text=text)
    if module_type is None:
        return None

    for module_entry in modules_dicts():
        if module_type == module_entry["Module"]:
            return '</' + module_type + '>'

    return None


def scan_xml_model(text: str) -> []:
    scanned_blocks = []

    # First split and format
    text_split = [pretty_text(line) for line in text.splitlines()]

    # Generate list of modules info
    module_beginnings = []
    module_closings = []

    # Find single simulation element, or reject otherwise
    for line in range(text_split.__len__()):
        text_line = text_split[line]
        if is_beginning_module_line(text_line):
            module_type = get_module_type(text_line)
            if module_type is not None and module_type == "CompuCell3D":
                module_beginnings.append((line, module_type))
        elif is_closing_module_line(text_line):
            module_type = get_module_type(text_line)
            if module_type is not None and module_type == "CompuCell3D":
                module_closings.append((line, module_type))

    problematic_sim = False
    if module_beginnings.__len__() != 1 or module_closings.__len__() != 1:
        module_beginnings = [(0, None)]
        module_closings = [(text_split.__len__() - 1, None)]
        problematic_sim = True

    sb = ScannedBlock()
    sb.beginning_line = module_beginnings[0][0]
    sb.closing_line = module_closings[0][0]
    sb.module_name = "CompuCell3D"
    sb.block_text = text_split[sb.beginning_line:sb.closing_line]
    scanned_blocks.append(sb)
    if problematic_sim:
        sb.is_problematic = True
        return scanned_blocks

    # Find module element pairs of beginning and closing lines
    in_element = None
    for line in range(text_split.__len__()):
        text_line = text_split[line]
        if is_module_single_line(text_line):
            module_type = get_module_type(text_line)
            if module_type is not None:
                module_beginnings.append((line, module_type))
                module_closings.append((line, module_type))
        elif is_beginning_module_line(text_line):
            module_type = get_module_type(text_line)
            if module_type is not None and module_type != "CompuCell3D":
                module_beginnings.append((line, module_type))
                if in_element is not None:
                    if module_beginnings.__len__() == 1:
                        prev_line = 0
                    else:
                        prev_line = module_beginnings[-1][0] + 1
                        if prev_line == module_beginnings[0][0] + 1:
                            prev_line = module_closings[0][0] - 1
                    module_closings.append((prev_line, None))
                in_element = module_type
        elif is_closing_module_line(text_line):
            module_type = get_module_type(text_line)
            if module_type is not None and module_type != "CompuCell3D":
                module_closings.append((line, module_type))
                if in_element is None:
                    if module_closings.__len__() == 1:
                        prev_line = 0
                    else:
                        prev_line = module_closings[-2][0] + 1
                        if prev_line == module_closings[0][0] + 1:
                            prev_line = module_beginnings[0][0] + 1
                    module_beginnings.append((prev_line, None))
                in_element = None

    if in_element:
        module_closings.append((module_closings[0][0] - 1, None))

    # Generate module element info
    for index in range(1, module_beginnings.__len__()):
        line_b, type_b = module_beginnings[index]
        line_c, type_c = module_closings[index]
        sb = ScannedBlock()
        sb.beginning_line = line_b
        sb.closing_line = line_c
        sb.block_text = text_split[line_b:line_c + 1]
        if type_b is None or type_b != type_c:
            sb.is_problematic = True
        elif type_c is None:
            sb.is_problematic = True
            sb.module_name = get_module_name(text_split[line_b])
        else:
            sb.module_name = get_module_name(text_split[line_b])

        if line_c < scanned_blocks[0].beginning_line or line_b > scanned_blocks[0].closing_line:
            sb.is_problematic = True

        scanned_blocks.append(sb)

    # Sweep for unidentified lines of text and group them into problematic blocks
    lines_with_blocks = [scanned_blocks[0].beginning_line, scanned_blocks[0].closing_line]
    for sb in [sb for sb in scanned_blocks if sb is not scanned_blocks[0]]:
        lines_with_blocks += list(range(sb.beginning_line, sb.closing_line + 1))

    line = 0
    while line in range(text_split.__len__()):
        if text_split[line].__len__() > 0 and line not in lines_with_blocks:
            sb = ScannedBlock()
            sb.beginning_line = line
            while line + 1 not in lines_with_blocks and line < text_split.__len__() - 1:
                if text_split[line].__len__() > 0:
                    sb.closing_line = line

                line += 1
            sb.is_problematic = True
            scanned_blocks.append(sb)

        line += 1

    return scanned_blocks


class ScannedBlock:

    WF_OUTSIDE_SIM = 0
    WF_MISSING_REQUISITE = 1

    def __init__(self):
        self.beginning_line = -1
        self.closing_line = -1
        self.module_name = None
        self.is_problematic = False
        self.block_text = []
        self.warning_flags = []
        self.warning_msgs = {}
        self.error_msgs = []

    def clear_warning_flags(self) -> None:
        self.warning_flags = []
        self.warning_msgs = {}

    def set_warn_missing_requisite(self, msg: str = None) -> None:
        if self.WF_MISSING_REQUISITE not in self.warning_flags:
            self.warning_flags.append(self.WF_MISSING_REQUISITE)
        if msg is not None:
            if self.WF_MISSING_REQUISITE not in self.warning_msgs.keys():
                self.warning_msgs[self.WF_MISSING_REQUISITE] = []
            self.warning_msgs[self.WF_MISSING_REQUISITE].append(msg)

    def clear_warn_missing_requisite(self) -> None:
        if self.WF_MISSING_REQUISITE in self.warning_flags:
            self.warning_flags.remove(self.WF_MISSING_REQUISITE)

        self.warning_msgs[self.WF_MISSING_REQUISITE] = []

    def is_missing_requisite(self) -> bool:
        return self.WF_MISSING_REQUISITE in self.warning_flags

    def msgs_missing_requisite(self) -> []:
        if not self.warning_msgs[self.WF_MISSING_REQUISITE]:
            self.warning_msgs[self.WF_MISSING_REQUISITE] = []
        return self.warning_msgs[self.WF_MISSING_REQUISITE]

    def set_warn_outside_sim_element(self) -> None:
        if self.WF_OUTSIDE_SIM not in self.warning_flags:
            self.warning_flags.append(self.WF_OUTSIDE_SIM)

    def clear_warn_outside_sim_element(self) -> None:
        if self.WF_OUTSIDE_SIM in self.warning_flags:
            self.warning_flags.remove(self.WF_OUTSIDE_SIM)

    def is_outside_sim_element(self) -> bool:
        return self.WF_OUTSIDE_SIM in self.warning_flags

    def contains_line(self, line):
        if self.beginning_line == -1 and self.closing_line == -1:
            return None
        return self.beginning_line <= line <= self.closing_line


class ElementCC3DX(ElementCC3D):
    def __init__(self, name: str = None, attributes: dict = None, cdata: str = None,
                 cc3d_xml_element: CC3DXMLElement = None, map_children: bool = True):

        self.element_list = []
        if cc3d_xml_element is not None:
            self.childrenList = []
            self.CC3DXMLElement = cc3d_xml_element
        elif name is not None:
            if attributes is None:
                attributes = {}
            if cdata is None:
                cdata = ""
            super(ElementCC3DX, self).__init__(name, attributes, cdata)
        else:
            raise ValueError

        if map_children:
            self._add_children()

    def ElementCC3DX(self, name: str = None, attributes: dict = None, cdata: str = None,
                     cc3d_xml_element: CC3DXMLElement = None, map_children: bool = True):
        child = ElementCC3DX(name=name, attributes=attributes, cdata=cdata, cc3d_xml_element=cc3d_xml_element,
                             map_children=map_children)
        self.childrenList.append(child.CC3DXMLElement)
        self.element_list.append(child)

    def _add_children(self):
        children = CC3DXMLListPy(self.CC3DXMLElement.getElements())
        [self.ElementCC3DX(cc3d_xml_element=child) for child in children if children is not None]
