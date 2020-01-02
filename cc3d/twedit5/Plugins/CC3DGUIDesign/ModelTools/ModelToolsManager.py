from importlib.machinery import SourceFileLoader
import os
import sys
import traceback

import cc3d.twedit5.twedit as twedit


class BasicToolData(object):
    def __init__(self, _name: str = '', _file_name: str = ''):

        self.attribute_names = ['name', 'file_name', 'author', 'version', 'class_name', 'module_type',
                                'short_description', 'long_description', 'tool_tip']

        self.name = _name

        self.file_name = _file_name

        self.author = ''

        self.version = ''

        self.class_name = ''

        self.module_type = ''

        self.short_description = ''

        self.long_description = ''

        self.tool_tip = ''

    def __str__(self):

        ret_str = ''

        for attribute_name in self.attribute_names:

            try:

                ret_str += attribute_name + '=' + str(getattr(self, attribute_name)) + '\n'

            except TypeError as e:

                ret_str += attribute_name + '=' + getattr(self, attribute_name) + '\n'

        return ret_str


class ModelToolsManager:

    tools_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(twedit.__file__)),
                                              "Plugins", "CC3DGUIDesign", "ModelTools"))

    def __init__(self):

        self.model_tools_dict = {}

        self.__tool_queries = {}

        self.__initialize()

    def __initialize(self):

        sys.path.insert(2, self.tools_path)

        tool_module_file_names = self.__get_tool_module_file_names(self.tools_path)

        self.__load_tool_modules(tool_module_file_names)

    def __load_tool_modules(self, tool_module_file_names) -> None:
        """
        Private method to load model tools
        :param tool_module_file_names: list of model tool file names
        :return: None
        """

        for tool_module_file_name in tool_module_file_names:

            tool_module_dir, tool_module_file_name_rel = os.path.split(tool_module_file_name)

            tool_name = tool_module_file_name_rel.replace("Tool.py", "")

            self.__tool_queries[tool_name] = self.query_tool(tool_name, tool_module_dir)

        keys_swap = []

        for tool_name, btd in self.__tool_queries.items():

            if btd is None:

                self.model_tools_dict[tool_name] = None

            else:

                btd: BasicToolData

                module = self.__load_tool_source(btd.name, btd.file_name)

                self.model_tools_dict[btd.name] = getattr(module, btd.class_name)

            if tool_name != btd.name:

                keys_swap.append((tool_name, btd.name))

        for key_swap in keys_swap:

            self.__tool_queries[key_swap[1]] = self.__tool_queries.pop(key_swap[0])

    def __get_tool_module_file_names(self, tools_path):

        return [f for f in self.__tool_module_file_names_in_dir(tools_path)]

    def __tool_module_file_names_in_dir(self, dir_path):

        dir_list = os.listdir(dir_path)

        for f in dir_list:

            f_abs = os.path.abspath(os.path.join(dir_path, f))

            if os.path.isdir(f_abs):

                for ff in self.__tool_module_file_names_in_dir(f_abs):

                    yield ff

            elif self.is_valid_tool_filename(f_abs):

                yield f_abs

    @staticmethod
    def is_valid_tool_filename(tool_filename):
        """
        Public method to determine if source is a valid model tool file name
        :param tool_filename: model tool file name
        :return:{bool} true if model file name is valid
        """

        return tool_filename.endswith("Tool.py")

    def query_tool(self, name, directory):
        """
        Public method to query model tool
        :param name: Name of model tool
        :param directory: directory containing model tool source
        :return: model tool basic tool data
        """

        attribute_names = BasicToolData().attribute_names

        attribute_names.remove('file_name')

        print('\nTool name=', name)

        file_name = "%sTool.py" % os.path.join(directory, name)

        print("\n ##################loading source for ", file_name)

        module = self.__load_tool_source(name, file_name)

        if module is not None:

            btd = BasicToolData(name, file_name)

            for attribute_name in attribute_names:

                setattr(btd, attribute_name, getattr(module, attribute_name))

            print(btd)

        else:

            btd = None

        return btd

    def __load_tool_source(self, name, file_name):
        """
        Private method to load a model tool from source
        :param name: Name to give module
        :param file_name: absolute file path to module
        :return: Loaded module
        """
        try:

            tool_dir = os.path.split(file_name)[0]

            sys.path.insert(2, tool_dir)

            import imp

            module = imp.load_source(name, file_name)

        except Exception as e:

            module = None

            print('Error loading tool: ')

            print(str(e))

            traceback.print_exc(file=sys.stdout)

        return module

    def active_tools(self):
        """
        Public method to yield active model tools and info
        :return: tool key, tool class, basic tool data
        """
        for key in self.model_tools_dict.keys():
            if self.model_tools_dict[key] is not None:
                yield key, self.model_tools_dict[key], self.__tool_queries[key]

    def active_tool_query(self):
        """
        Public method to yield active model tool info
        :return: basic tool data key, basic tool data
        """
        for name, btd in self.__tool_queries.items():
            if btd is not None:
                yield name, btd

    def add_tool(self, tool_directory) -> bool:
        """
        Public method to add tool to active model tools list
        :param tool_directory: directory containing model tool
        :return:{bool} low if something was not successfully loaded
        """
        t_dir = os.path.abspath(tool_directory)

        if not os.path.isdir(t_dir):

            return False

        tool_module_file_names = self.__get_tool_module_file_names(t_dir)

        if not tool_module_file_names:

            return False

        mtd_keys = list(self.model_tools_dict.keys())

        tq_keys = list(self.__tool_queries.keys())

        self.__load_tool_modules(tool_module_file_names)

        something_loaded = True

        for key, btd in self.__tool_queries.items():

            if key not in tq_keys and btd is None:

                self.__tool_queries.pop(key)

        for key, tool in self.model_tools_dict.items():

            if key not in mtd_keys and tool is None:

                self.model_tools_dict.pop(key)

                something_loaded = False

        return something_loaded
