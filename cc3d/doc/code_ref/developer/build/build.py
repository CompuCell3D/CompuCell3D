from os import mkdir
from os.path import abspath, dirname, isdir, join
from shutil import rmtree

#: source directory for API source code auto-generation
api_source_dir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
#: build directory for API source code auto-generation
api_build_dir = join(dirname(abspath(__file__)), "api")
#: source directory for doc generation
man_source_dir = dirname(abspath(__file__))
#: build directory for doc generation
man_build_dir = dirname(man_source_dir)


def build(builder: str = 'html'):
    """
    Builds documentation using Sphinx

    :param str builder: Sphinx builder
    :return: None
    """
    from sphinx.cmd import build as sphinx_build
    from sphinx.cmd.make_mode import BUILDERS

    if not isdir(api_build_dir):
        build_api()

    if builder not in [x[1] for x in BUILDERS]:
        raise EnvironmentError("Builder not supported")

    build_dir = join(man_build_dir, builder)

    if not isdir(build_dir):
        mkdir(build_dir)

    argv = [man_source_dir, build_dir,
            '-b', builder]

    sphinx_build.main(argv)


def build_api():
    """
    Auto-generates API documentation source code and stores it in ./api

    :return: None
    """
    if not isdir(api_build_dir):
        mkdir(api_build_dir)

    exclude_pattern = ['doc', "experimental", "gui_plugins", "player5", "tests", "twedit5", 'version_fetcher.py',
                       'core/envVarSanitizer.py', 'core/param_scan', 'core/Validation',
                       "cpp/bin", "cpp/CompuCell3DPlugins", "cpp/CompuCell3DSteppables",
                       "cpp/lib", "cpp/PlayerPython*", "cpp/SerializerDE*"]
    source_prefix = "../../../../"
    exclude_pattern = [source_prefix + x for x in exclude_pattern]
    argv = [api_source_dir, *exclude_pattern,
            '-o', api_build_dir,
            '--implicit-namespaces',
            '--separate']

    from sphinx.ext import apidoc
    apidoc.main(argv)


def clear_api():
    """
    Clears auto-generated API documentation source code

    :return: None
    """
    if isdir(api_build_dir):
        rmtree(api_build_dir)


def main():
    """
    Generates HTML manual with fresh API and stores in *man_build_dir*/html

    :return: None
    """
    clear_api()
    build()


if __name__ == '__main__':
    main()
