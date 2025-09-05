Angiogenesis demonstrates the formation of blood vessels. There are many ways to achieve angiogenesis in CC3D, but this one is a Python-only simulation with only one cell type. It leverages secretion, chemotaxis, and cell-cell adhesion.

Expected result: A branching pattern of interlinked capillaries that changes shape (for better or worse) as the simulation progresses. 

To use the model, set the display mode to 3D, and create a second graphics window to display VEGF concentration. 

First, endothelial cells have low contact energy with other endothelial cells, which will encourage them to adhere. Their contact energy with the medium should be relatively higher so that each cell exposes part of its surface area to the outside. Contact energy alone is sufficient to form branching patterns, but adding secretion and chemotaxis will make this process happen more consistently. 

Next, endothelial cells also secrete VEGF to encourage nearby neighbors to migrate. We use the FlexibleDiffusionSolverFE plugin to do this. The `DecayConstant` must be set low enough so that VEGF can reach nearby cells but also high enough so that it cannot reach not distant cells. We also set periodic boundary conditions so that secretions spread to the opposite sides of the simulation when they reach the edge of the screen. 

Relevant documentation: https://compucell3dreferencemanual.readthedocs.io/en/latest/diffusion_solver.html