
def check_nanohub_and_count():
    """
    Function for proper counting of number of runs in NanoHub
    Checks for NANOHUB_SIM in the environment variables
    If NANOHUB_SIM is there will attempt to call TOOL_HOME/bin/cc3d_count.sh
    NANOHUB_SIM must be TOOL_HOME/bin/
    :return:
    """
    from os import environ
    import subprocess
    if 'NANOHUB_SIM' in environ:
        # NANOHUB_SIM will be the path to the sh script that starts the nanohub run. That info is already in the .sh
        # files under binDir
        bin_dir = environ.get('NANOHUB_SIM')
        if bin_dir is not None:
            count_sh = join(bin_dir, r'cc3d_count.sh')
            try:
                subprocess.call(['submit', '--local', count_sh])
                return
            except:
                print(f"Couldn't call {count_sh}! Is the file there?\nProceeding.")
                return
        else:
            print("Coundn't find NANOHUB_SIM in the environment variables.\nProceeding")
            return
    return
