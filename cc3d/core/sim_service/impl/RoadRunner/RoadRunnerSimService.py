import antimony
import copyreg
import os

from ...PySimService import PySimService


from .roadrunner_registry import get_roadrunner_inst


class RoadRunnerSimService(PySimService):
    def __init__(self, step_size: float = 1.0):
        super().__init__()

        self._step_size = step_size
        self.beginning_step = 0

    def _run(self):

        from .roadrunner_types import COPY_REGS
        [copyreg.pickle(*crp) for crp in COPY_REGS]

        rr = get_roadrunner_inst()

        # Grab RoadRunner public methods for exposure on wrap
        import inspect
        from cc3d.core.sim_service import service_function

        methods = [x for x in inspect.getmembers(rr, predicate=inspect.ismethod)]
        methods_excluded = ["load"]  # we override load with antimony support
        methods_absorbed = [x for x in methods
                            if not x[0].startswith("_") and x[0] not in methods_excluded]
        for x in methods_absorbed:
            service_function(x[1])

    def _init(self) -> bool:
        return True

    def _start(self) -> bool:
        return True

    def _step(self) -> bool:

        start_time = self.current_step * self._step_size
        finish_time = start_time + self._step_size

        rr = get_roadrunner_inst()
        rr.simulate(start=start_time, end=finish_time, points=2)
        return True

    def _finish(self):
        pass

    def load(self, _input):

        kwargs = {'model_file': None, 'model_string': None}
        if os.path.isfile(_input):
            kwargs['model_file'] = _input
        else:
            kwargs['model_string'] = _input

        sbml_string = translate_to_sbml_string(**kwargs)[0]
        rr = get_roadrunner_inst()
        rr.load(sbml_string)


class AntimonyTranslatorError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def translate_to_sbml_string(model_file: str = None, model_string: str = None):
    """
    Returns string of SBML model specification translated from Antimony or CellML model specification file or string
    :param model_file: absolute path to model specification file
    :param model_string: model specification string
    :return {str,str}: string of SBML model specification, string of main module name
    """
    assert model_file is not None or model_string is not None

    # Just to be sure, call clear previous loads
    antimony.clearPreviousLoads()

    # Loading from model string or file?
    if model_file is None:
        loader, arg = antimony.loadString, model_string
    else:
        loader, arg = antimony.loadFile, model_file

    if loader(arg) == -1:
        raise AntimonyTranslatorError(antimony.getLastError())

    # Get main loaded module
    main_module_name = antimony.getMainModuleName()
    if not main_module_name:
        raise AntimonyTranslatorError(antimony.getLastError())

    # Return string of SBML model specification
    translated_model_string = antimony.getSBMLString(main_module_name)
    if not translated_model_string:
        raise AntimonyTranslatorError(antimony.getLastError())
    else:
        return translated_model_string, main_module_name
