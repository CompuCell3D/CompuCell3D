import unittest



class TestAPI(unittest.TestCase):

    def setUp(self):
        pass

    def test_core_functionality(self):
        try:
            import CompuCell
        except:
            self.fail("Could not import CompuCell module")

        # sim = CompuCell.Simulator()
        pt_1 = CompuCell.Point3D()
        pt_2 = CompuCell.Point3D(1, 1, 1)
        print('pt=', pt_1)

        potts = CompuCell.Potts3D()
        print('potts=', potts)
        print('potts.getTemperature()=', potts.getTemperature())

        baby_sim = CompuCell.BabySim()

        print('potts=', baby_sim.getPotts())


        # print('_b_strategy=', bs.getBoundaryStrategy())

        boundary_strategy = baby_sim.getBoundaryStrategy()
        # bbs = baby_sim.getBabyBoundaryStrategy()
        bbs = baby_sim.getBoundaryStrategy()



        print('_baby_strategy=', bbs)


        # s = CompuCell.Simulator()
        print('ps=',baby_sim.ps)

        print('bppd=',baby_sim.bppd)

        baby_sim.bppd.Boundary_x("demo")

        sim = CompuCell.Simulator()

        print

        # boundary_strategy = baby_sim.getBoundaryStrategy()
    def test_core_object_creation_api(self):
        import CompuCell
        from cc3d.CompuCellSetup import init_modules, parseXML

        xml_fname = r'd:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_2D.xml'
        cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

        # # this loads all plugins/steppables - need to recode it to make loading on-demand only
        # CompuCell.initializePlugins()

        sim = CompuCell.Simulator()
        # boundary_strategy = sim.getBoundaryStrategy()



        init_modules(sim, cc3dXML2ObjConverter)

        # sim.initializeCC3D()
        # at this point after initialize cc3d stepwe can start querieg sim object.
        # print('num_steps=', sim.getNumSteps())

        # sim.start()


        # this loads all plugins/steppables - need to recode it to make loading on-demand only
        CompuCell.initializePlugins()

        sim.initializeCC3D()
        # at this point after initialize cc3d stepwe can start querieg sim object.
        print('num_steps=', sim.getNumSteps())
        max_num_steps = sim.getNumSteps()

        sim.extraInit()

        sim.start()

        cur_step = 0
        while cur_step < max_num_steps/100:

            sim.step(cur_step)
            cur_step += 1




if __name__ == '__main__':
    unittest.main()