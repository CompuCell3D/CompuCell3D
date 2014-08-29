/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#include "AdvectionDiffusionSolverFE.h"
#include "GPUEnabled.h"
#include "FlexibleDiffusionSolverFE.h"
// #include "FiPySolver.h"
#include "FlexibleDiffusionSolverADE.h"
#include "KernelDiffusionSolver.h"
#include "ReactionDiffusionSolverFE_SavHog.h"
#include "ReactionDiffusionSolverFE.h"
#include "FlexibleReactionDiffusionSolverFE.h"
#include "FastDiffusionSolver2DFE.h"
#include "SteadyStateDiffusionSolver2D.h"
#include "SteadyStateDiffusionSolver.h"
// #include "ReactionAdvectionDiffusionSolverFE.h"
// #include "ReactionAdvectionDiffusionSolverFE_TagBased.h"
// #include "ScalableFlexibleDiffusionSolverFE.h"
#include "DiffusionSolverFE.h"
#include "DiffusionSolverFE_CPU.h"
#include "DiffusionSolverFE_CPU_Implicit.h"

#if OPENCL_ENABLED == 1
#include "FlexibleDiffusionSolverFE_GPU.h"
#include "OpenCL/DiffusionSolverFE_OpenCL.h"
//#include "OpenCL/DiffusionSolverFE_OpenCL_Implicit.h"
#include "OpenCL/ReactionDiffusionSolverFE_OpenCL_Implicit.h"
#endif

// #include "ReactionDiffusionFile.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Steppable, AdvectionDiffusionSolverFE> 
advectionDiffusionSolverProxy("AdvectionDiffusionSolverFE", "Solves advection diffusion equation on the cell field",
	    &Simulator::steppableManager);

BasicPluginProxy<Steppable, FlexibleDiffusionSolverFE> 
flexibleDiffusionSolverProxy("FlexibleDiffusionSolverFE", "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference",
	    &Simulator::steppableManager);

// BasicPluginProxy<Steppable, FiPySolver> 
// fiPySolverProxy("FiPySolver", "Solves diffusion equation on the lattice. Uses FiPy python library",
	    // &Simulator::steppableManager);

BasicPluginProxy<Steppable, FlexibleDiffusionSolverADE> 
flexibleDiffusionSolverADEProxy("FlexibleDiffusionSolverADE", "Solves diffusion equation on the lattice. Uses Alternate Direction Explicit method -  finite difference",
	    &Simulator::steppableManager);

 BasicPluginProxy<Steppable, KernelDiffusionSolver> 
 kernelDiffusionSolverProxy("KernelDiffusionSolver", "Solves diffusion equation on the lattice. Uses diffusion equation Green's Function to solve the equation ",
	     &Simulator::steppableManager);


BasicPluginProxy<Steppable, ReactionDiffusionSolverFE_SavHog> 
reactionDiffusion_SavHogSolverProxy("ReactionDiffusionSolverFE_SavHog", "Solves reaction-diffusion equation on the lattice - used in dictyostelium simulation",
	    &Simulator::steppableManager);

 BasicPluginProxy<Steppable, ReactionDiffusionSolverFE> 
 reactionDiffusionSolverProxy("ReactionDiffusionSolverFE", "Solves reaction-diffusion system of equations on the lattice ",
	     &Simulator::steppableManager);

BasicPluginProxy<Steppable, FlexibleReactionDiffusionSolverFE> 
flexibleReactionDiffusionSolverProxy("FlexibleReactionDiffusionSolverFE", "Solves reaction-diffusion system of equations on the lattice ",
	     &Simulator::steppableManager);
         
         
// BasicPluginProxy<Steppable, ReactionAdvectionDiffusionSolverFE> 
// reactionAdvectionDiffusionSolverProxy("ReactionAdvectionDiffusionSolverFE", "Solves reaction-diffusion system of equations on the lattice ",
            // &Simulator::steppableManager);

// BasicPluginProxy<Steppable, ReactionAdvectionDiffusionTagsSolverFE> 
// reactionAdvectionDiffusionTagsSolverProxy("ReactionAdvectionDiffusionTagsSolverFE", "Solves reaction-diffusion system of equations on the lattice ",
            // &Simulator::steppableManager);
       
BasicPluginProxy<Steppable, FastDiffusionSolver2DFE> 
fastDiffusionSolverProxy("FastDiffusionSolver2DFE", "Solves diffusion equation on the lattice. Provides limited flexibility but is faster than FlexibleDiffusionSolver however operates only in the xy plane and is 2D only",
	     &Simulator::steppableManager);


BasicPluginProxy<Steppable, SteadyStateDiffusionSolver2D> 
steadyStateDiffusionSolver2DProxy("SteadyStateDiffusionSolver2D", "Solves for steady state of diffusion equation in 2D.",
	    &Simulator::steppableManager);

BasicPluginProxy<Steppable, SteadyStateDiffusionSolver> 
steadyStateDiffusionSolverProxy("SteadyStateDiffusionSolver", "Solves for steady state of diffusion equation in 3D.",
	    &Simulator::steppableManager);

// BasicPluginProxy<Steppable, ScalableFlexibleDiffusionSolverFE> 
// scalableFlexibleDiffusionSolverProxy("ScalableFlexibleDiffusionSolverFE", "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
            // &Simulator::steppableManager);

BasicPluginProxy<Steppable, DiffusionSolverFE_CPU> 
diffusionSolverFEProxy("DiffusionSolverFE", "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
            &Simulator::steppableManager);

BasicPluginProxy<Steppable, DiffusionSolverFE_CPU_Implicit> 
diffusionSolverFEImplicitProxy("DiffusionSolverFE_Implicit", "Solves diffusion equation on the lattice. Uses Implicit method - finite difference.",
            &Simulator::steppableManager);
			
#if OPENCL_ENABLED == 1
BasicPluginProxy<Steppable, DiffusionSolverFE_OpenCL> 
diffusionSolverOpenCLProxy("DiffusionSolverFE_OpenCL", "Solves diffusion equation on the lattice with OpenCL. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
            &Simulator::steppableManager);

//BasicPluginProxy<Steppable, DiffusionSolverFE_OpenCL_Implicit> 
//diffusionSolverOpenCLImplicitProxy("DiffusionSolverFE_OpenCL_Implicit", "Solves diffusion equation on the lattice with OpenCL. Uses Implicit method - finite difference.",
//            &Simulator::steppableManager);

BasicPluginProxy<Steppable, ReactionDiffusionSolverFE_OpenCL_Implicit>
reactionDiffusionSolverOpenCLImplicitProxy("ReactionDiffusionSolverFE_OpenCL_Implicit", "Solves diffusion equation on the lattice with OpenCL. Uses Implicit method - finite difference.",
            &Simulator::steppableManager);

#endif

//BasicPluginProxy<Steppable, ReactionDiffusionFile>
//reactionDiffusionFileProxy("ReactionDiffusionFile", "Loads from files precalculated reaction-diffusion fields",
//            &Simulator::steppableManager);
	    