''' 
ISSUE: There is no way to set the random number seed in the Gillespie solver. 
All instances of the same model run in step with each other.
This bug appears to have been introduced somewhere around CC3D 4.2.5
https://github.com/CompuCell3D/CompuCell3D/issues/652

There are two identical cells each with copies of the same simple ODE model, A->B. 
The models use the gilliespie solver. During the run the amount of A and B in both 
cells are printed every MCS. 
They values are in lock-step. 
Below is the programs output:

MCS CellID    A    B
0     1    100.0  0.0
0     2    100.0  0.0

1     1    97.0   3.0
1     2    97.0   3.0

2     1    95.0   5.0
2     2    95.0   5.0

3     1    91.0   9.0
3     2    91.0   9.0

4     1    89.0  11.0
4     2    89.0  11.0

5     1    87.0  13.0
5     2    87.0  13.0

6     1    85.0  15.0
6     2    85.0  15.0

7     1    83.0  17.0
7     2    83.0  17.0   ... 
'''

from cc3d.core.PySteppables import *
import numpy as np

class Gillespie_random_numberSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)
        self.set_gillespie_integrator_seed(20)

    def start(self):
        # Antimony model string
        model_string = '''        
        //Equations
        A -> B;  k1*A
        //Parameters        
        k1 = 0.02;
        //Initial Conditions
        A = 100
        B = 0
        '''  
        
        # create two cells and add Antimony
        newCell1 = self.new_cell(self.ACELL)
        self.cell_field[45:50, 45:50, 0] = newCell1
        newCell1.targetVolume = newCell1.volume
        newCell1.lambdaVolume = 10.
        RRinst1 = self.add_antimony_to_cell(model_string=model_string, model_name='firstOrder', \
                                            cell=newCell1, step_size=1, integrator='gillespie' )
       
        newCell2 = self.new_cell(self.ACELL)
        self.cell_field[50:55, 45:50, 0] = newCell2
        newCell2.targetVolume = newCell2.volume
        newCell2.lambdaVolume = 10.
        RRinst2 = self.add_antimony_to_cell(model_string=model_string, model_name='firstOrder', \
                                            cell=newCell2, step_size=1, integrator='gillespie' )
        
    def step(self, mcs):
        self.timestep_sbml()
        for cell in self.cell_list_by_type(self.ACELL):
            A = cell.sbml.firstOrder['A']              
            B = cell.sbml.firstOrder['B']              
            print(mcs,cell.id,A,B)
        print()


