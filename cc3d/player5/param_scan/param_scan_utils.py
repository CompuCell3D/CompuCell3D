import sys
import traceback


def run_main_player_run_script(arg_list_local: list):
    """
    Runs simulation via helper script cc3d.core.param_scan.main_player_run
    Experimental function .
    :param arg_list_local: {list} list of cml options for the player
    :return: None
    """

    from . import main_player_run

    sys.argv = arg_list_local

    with open(main_player_run) as sim_fh:
        try:
            code = compile(sim_fh.read(), main_player_run, 'exec')

        except:
            code = None
            traceback.print_exc(file=sys.stdout)

        if code is not None:
            try:
                exec(code)
            except:
                traceback.print_exc(file=sys.stdout)
