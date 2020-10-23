from multiprocessing import current_process


def name_mangler(process_name: str, name: str) -> str:
    """
    Standard name mangling of this module by process name
    :param process_name: process name (e.g., hex( os.getpid() ) )
    :param name: name to mangle
    :return: {str} mangled name
    """
    return f"{process_name}_{name}"


def standard_process_name() -> str:
    """
    Get standard process name
    :return: {str} standard process name
    """
    return hex(current_process().pid)
