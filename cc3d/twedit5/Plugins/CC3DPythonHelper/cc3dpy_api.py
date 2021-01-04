"""Generates cc3dpy.api"""

import importlib
import inspect
from os.path import abspath, dirname, join


api_name = "cc3d_py"
"""Name of api file"""

api_filename = join(dirname(abspath(__file__)), api_name + ".api")
"""Absolute path to api file"""


def collect_parts(_mod, processed_modules: list) -> dict:
    """
    Recursively collects unique members of a module

    :param _mod: a module
    :param processed_modules: list of previously processed modules
    :type processed_modules: list
    :return: objects of modules by unique name string
    :rtype: dict of [str: object]
    """

    collected_parts = {}
    module_name = _mod.__name__

    if not module_name.startswith("cc3d"):
        return collected_parts

    if module_name.startswith("_"):
        return collected_parts

    if module_name in processed_modules:
        return collected_parts

    processed_modules.append(module_name)

    print("Collecting module members", module_name, _mod)

    for name, obj in inspect.getmembers(_mod):

        if name.startswith("_"):
            continue
        elif name == "cvar":
            continue

        if inspect.ismodule(obj):
            collected_parts.update(collect_parts(obj, processed_modules))

        elif inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
            key = obj.__module__ + "." + obj.__name__
            val = obj
            collected_parts[key] = val

        elif inspect.isbuiltin(obj):
            # Handle swig stuff
            suffix_patterns = ["_swigregister", "_destroy", "_getInstance", "_instantiate"]
            if any([obj.__name__.endswith(x) for x in suffix_patterns]):
                continue

            key = module_name + "." + name
            val = obj
            collected_parts[key] = val

        else:
            key = module_name + "." + name
            collected_parts[key] = obj

    return collected_parts


def filter_parts(collected_parts: dict) -> None:
    """
    Filters dictionary returned by :func:`collect_parts`

    :param collected_parts: collected module parts
    :type collected_parts: dict
    :return: None
    """
    keys = list(collected_parts.keys())
    [collected_parts.pop(k) for k in keys if not k.startswith("cc3d")]
    [collected_parts.pop(k) for k in keys if k.startswith("cc3d.player5")]
    [collected_parts.pop(k) for k in keys if k.startswith("cc3d.twedit")]
    [collected_parts.pop(k) for k in keys if k.startswith("cc3d.cpp.PlayerPython")]
    [collected_parts.pop(k) for k in keys if k.startswith("cc3d.cpp.SerializerDEPy")]


def annotate(obj, key: str, annotated_dict: dict) -> None:
    """
    Generates api signature info for an object

    :param obj: an object to annotate
    :type obj: Any
    :param key: unique name in module; corresponds to the key in annotated_dict
    :type key: str
    :param annotated_dict: dictionary to store annotations
    :type annotated_dict: dict of [str: str]
    :return: None
    """
    try:
        sig = inspect.signature(obj)
        s = str(sig)
    except TypeError:
        # Assumed value type
        s = ""
    except ValueError:
        # Assumed callable type
        s = "(??)"

    annotated_dict[key] = s

    if inspect.isclass(obj):
        for a in inspect.classify_class_attrs(obj):
            if a.name.startswith("_"):
                continue

            if a.kind in ["data", "property"]:
                annotated_dict[key + "." + a.name] = ""
            elif a.kind.find("method") >= 0:
                annotate(getattr(obj, a.name), key + "." + a.name, annotated_dict)
            else:
                raise NotImplementedError


def main():
    """
    Main call to generate cc3dpy.api

    :return: None
    """

    module_patterns = [
        "CompuCellSetup",
        "core.iterators",
        "core.PySteppables",
        "core.RoadRunnerPy",
        "core.XMLUtils",
        "cpp.CompuCell"
    ]

    modules = [importlib.import_module(f"cc3d.{mp}") for mp in module_patterns]

    collected_parts = {}
    processed_modules = []
    [collected_parts.update(collect_parts(mp, processed_modules)) for mp in modules]

    filter_parts(collected_parts)

    annotated_dict = {k: "" for k in collected_parts.keys()}

    [annotate(v, k, annotated_dict) for k, v in collected_parts.items()]

    with open(api_filename, "w") as f:
        f.writelines([k + v + "\n" for k, v in annotated_dict.items()])


if __name__ == "__main__":
    main()
