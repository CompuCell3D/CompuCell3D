"""
MaBoSS Cell Cycle Model in CC3D
===============================

Written by T.J. Sego, Ph.D.

Biocomplexity Institute

Indiana University

Bloomington, IN

Overview
========

This simulation implements a multicellular spatial version of the cell cycle model in

Fauré A, Naldi A, Chaouiya C, Thieffry D.
Dynamical analysis of a generic Boolean model for the control of the mammalian cell cycle.
Bioinformatics. 2006 Jul 15;22(14):e124-31. doi: 10.1093/bioinformatics/btl210. PMID: 16873462.

The files /Simulation/cellcycle.bnd and /Simulation/cellcycle_runcfg.cfg were curated from https://maboss.curie.fr/.

"""

from cc3d import CompuCellSetup
        

from MaBoSSCellCycleSteppables import MaBoSSCellCycleSteppable

CompuCellSetup.register_steppable(steppable=MaBoSSCellCycleSteppable(frequency=1))


CompuCellSetup.run()
