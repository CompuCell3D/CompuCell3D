from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCellExtraModules


class CustomCellAttributePythonSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.custom_attr_steppable_cpp = None

    def start(self):
        self.custom_attr_steppable_cpp = CompuCellExtraModules.getCustomCellAttributeSteppable()

    def step(self, mcs):
        print('mcs=', mcs)

        for cell in self.cell_list:
            custom_cell_attr_data = self.custom_attr_steppable_cpp.getCustomCellAttribute(cell)
            print('custom_cell_attr_data=', custom_cell_attr_data)
            print('custom_cell_attr_data.x=', custom_cell_attr_data.x)

            custom_cell_attr_data.x = cell.id * mcs ** 2

            print('after modification custom_cell_attr_data.x=', custom_cell_attr_data.x)

            print('custom_cell_attr_data.array=', custom_cell_attr_data.array)
            print('custom_cell_attr_data.array[0]=', custom_cell_attr_data.array[0])

            if len(custom_cell_attr_data.array) < 5:
                custom_cell_attr_data.array.push_back(100.0)
            print('custom_cell_attr_data.array[len(custom_cell_attr_data.array)-1] = ',
                  custom_cell_attr_data.array[len(custom_cell_attr_data.array) - 1])

            simple_map = custom_cell_attr_data.simple_map

            print('simple_map.size()=', simple_map.size())
            vec = CompuCellExtraModules.vector_int()
            vec.push_back(20)
            vec.push_back(30)
            simple_map[cell.id] = vec

            print('simple_map[cell.id]=', simple_map[cell.id])

            break
