import os

import roadrunner


_rr_reg = {}


def get_roadrunner_inst():
    pid = os.getpid()
    try:
        rr = _rr_reg[pid]
    except KeyError:
        rr = roadrunner.RoadRunner()
        _rr_reg[pid] = rr
    return rr
