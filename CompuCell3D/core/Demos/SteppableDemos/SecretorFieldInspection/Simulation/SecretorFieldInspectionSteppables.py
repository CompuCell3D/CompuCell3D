from cc3d.core.PySteppables import *


class SecretorFieldInspectionSteppable(SteppableBasePy):

    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)


    def step(self,mcs):
        fgf_secretor = self.get_field_secretor("FGF")
        
        for i, cell in enumerate(self.cell_list):
           
            amount_see_by_cell = fgf_secretor.amountSeenByCell(cell)
            
            print(f'cell.id={cell.id} saw {amount_see_by_cell} of FGF')
            
            if i > 10:
                # we only print data for first 10 cells                       
                break

        total_field_integral = fgf_secretor.totalFieldIntegral()
                
        print(f'total amount of chemical in FGF is {total_field_integral}')

        