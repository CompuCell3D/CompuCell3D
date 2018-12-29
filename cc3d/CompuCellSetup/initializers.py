import CompuCell
from cc3d.CompuCellSetup import init_modules, parseXML
import cc3d.CompuCellSetup as CompuCellSetup


def getCoreSimulationObjects():

    # xml_fname = r'd:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_2D.xml'
    # cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    # CompuCell.initializePlugins()

    sim = CompuCell.Simulator()
    simthread = None
    xml_fname = CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript

    cc3d_xml2_obj_converter = parseXML(xml_fname=xml_fname)
    #  cc3d_xml2_obj_converter cannot be garbage colected hence goes to persisten storage declared at the global level
    # in CompuCellSetup
    CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter = cc3d_xml2_obj_converter

    # cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

    print ('CompuCellSetup.cc3dSimulationDataHandler=',CompuCellSetup.cc3dSimulationDataHandler)
    print('cc3dSimulationDataHandler.cc3dSimulationData.pythonScript=',CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
    init_modules(sim, cc3d_xml2_obj_converter)
    #
    # # sim.initializeCC3D()
    # # at this point after initialize cc3d stepwe can start querieg sim object.
    # # print('num_steps=', sim.getNumSteps())
    #
    # # sim.start()
    #
    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    CompuCell.initializePlugins()
    print ("sim=",sim)
    sim.initializeCC3D()
    # sim.extraInit()

    return sim, simthread

def initializeSimulationObjects(sim, simthread):
    """

    :param sim:
    :param simthread:
    :return:
    """
    sim.extraInit()
    print('THIS IS num_steps=', sim.getNumSteps())
    max_num_steps = sim.getNumSteps()

    sim.start()
    cur_step = 0
    while cur_step < max_num_steps / 100:
        sim.step(cur_step)
        cur_step += 1

    # print("sim=", sim)
    # sim.initializeCC3D()
