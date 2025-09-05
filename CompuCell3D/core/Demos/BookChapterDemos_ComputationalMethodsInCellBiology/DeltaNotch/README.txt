This demonstrates the use of ODEs in SBML to control single-cell behaviors.

Expected result: A checkerboard pattern where a few cells have high concentrations of Delta but low values of Notch. The cells in between them should have the opposite: high Notch but low Delta. 

One of the most important mechanisms of cell signaling is mediated by Notch, a transmembrane receptor that coordinates a signaling system known as the Notch pathway. Notch signaling regulates cell fates and pattern formation.

Delta and Notch are transmembrane proteins, and Delta is the ligand of Notch. Delta in the signaling cell works as the ligand for Notch in the signal-receiving cell. Notch activation leads to downstream inhibition of Delta expression in the signal-receiving cell. 

Relevant documentation: https://compucell3dreferencemanual.readthedocs.io/en/latest/sbml_solver.html