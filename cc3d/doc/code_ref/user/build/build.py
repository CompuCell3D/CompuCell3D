from os import mkdir
from os.path import abspath, dirname, isdir, join

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

    if builder not in [x[1] for x in BUILDERS]:
        raise EnvironmentError("Builder not supported")

    build_dir = join(man_build_dir, builder)

    if not isdir(build_dir):
        mkdir(build_dir)

    argv = [man_source_dir, build_dir,
            '-b', builder]

    sphinx_build.main(argv)


def main():
    """
    Generates HTML manual and stores in *man_build_dir*/html

    :return: None
    """
    build()


if __name__ == '__main__':
    main()
