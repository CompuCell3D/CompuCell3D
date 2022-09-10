

#include "AdvectionDiffusionSolverFE.h"
#include "GPUEnabled.h"
#include "FlexibleDiffusionSolverFE.h"
// #include "FiPySolver.h"
#include "FlexibleDiffusionSolverADE.h"
#include "KernelDiffusionSolver.h"
#include "ReactionDiffusionSolverFE_SavHog.h"
#include "ReactionDiffusionSolverFE.h"
#include "ReactionDiffusionSolverFVM.h"
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
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto advectionDiffusionSolverProxy = registerPlugin<Steppable, AdvectionDiffusionSolverFE>(
        "AdvectionDiffusionSolverFE",
        "Solves advection diffusion equation on the cell field",
        &Simulator::steppableManager
);

auto flexibleDiffusionSolverProxy = registerPlugin<Steppable, FlexibleDiffusionSolverFE>(
        "FlexibleDiffusionSolverFE",
        "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference",
        &Simulator::steppableManager
);

// auto fiPySolverProxy = registerPlugin<Steppable, FiPySolver>(
// 	"FiPySolver",
// 	"Solves diffusion equation on the lattice. Uses FiPy python library",
// 	&Simulator::steppableManager
// );

auto flexibleDiffusionSolverADEProxy = registerPlugin<Steppable, FlexibleDiffusionSolverADE>(
        "FlexibleDiffusionSolverADE",
        "Solves diffusion equation on the lattice. Uses Alternate Direction Explicit method -  finite difference",
        &Simulator::steppableManager
);

auto kernelDiffusionSolverProxy = registerPlugin<Steppable, KernelDiffusionSolver>(
        "KernelDiffusionSolver",
        "Solves diffusion equation on the lattice. Uses diffusion equation Green's Function to solve the equation ",
        &Simulator::steppableManager
);

auto reactionDiffusion_SavHogSolverProxy = registerPlugin<Steppable, ReactionDiffusionSolverFE_SavHog>(
        "ReactionDiffusionSolverFE_SavHog",
        "Solves reaction-diffusion equation on the lattice - used in dictyostelium simulation",
        &Simulator::steppableManager
);

auto reactionDiffusionFVMSolverProxy = registerPlugin<Steppable, ReactionDiffusionSolverFVM>(
        "ReactionDiffusionSolverFVM",
        "Solves reaction-diffusion system of equations on the lattice using the finite volume method",
        &Simulator::steppableManager
);

auto reactionDiffusionSolverProxy = registerPlugin<Steppable, ReactionDiffusionSolverFE>(
        "ReactionDiffusionSolverFE",
        "Solves reaction-diffusion system of equations on the lattice ",
        &Simulator::steppableManager
);

auto flexibleReactionDiffusionSolverProxy = registerPlugin<Steppable, FlexibleReactionDiffusionSolverFE>(
        "FlexibleReactionDiffusionSolverFE",
        "Solves reaction-diffusion system of equations on the lattice ",
        &Simulator::steppableManager
);

// auto reactionAdvectionDiffusionSolverProxy = registerPlugin<Steppable, ReactionAdvectionDiffusionSolverFE>(
// 	"ReactionAdvectionDiffusionSolverFE",
// 	"Solves reaction-diffusion system of equations on the lattice ",
// 	&Simulator::steppableManager
// );

// auto reactionAdvectionDiffusionTagsSolverProxy = registerPlugin<Steppable, ReactionAdvectionDiffusionTagsSolverFE>(
// 	"ReactionAdvectionDiffusionTagsSolverFE",
// 	"Solves reaction-diffusion system of equations on the lattice ",
// 	&Simulator::steppableManager
// );

auto fastDiffusionSolverProxy = registerPlugin<Steppable, FastDiffusionSolver2DFE>(
        "FastDiffusionSolver2DFE",
        "Solves diffusion equation on the lattice. Provides limited flexibility but is faster than FlexibleDiffusionSolver however operates only in the xy plane and is 2D only",
        &Simulator::steppableManager
);

auto steadyStateDiffusionSolver2DProxy = registerPlugin<Steppable, SteadyStateDiffusionSolver2D>(
        "SteadyStateDiffusionSolver2D",
        "Solves for steady state of diffusion equation in 2D.",
        &Simulator::steppableManager
);

auto steadyStateDiffusionSolverProxy = registerPlugin<Steppable, SteadyStateDiffusionSolver>(
        "SteadyStateDiffusionSolver",
        "Solves for steady state of diffusion equation in 3D.",
        &Simulator::steppableManager
);

// auto scalableFlexibleDiffusionSolverProxy = registerPlugin<Steppable, ScalableFlexibleDiffusionSolverFE>(
// 	"ScalableFlexibleDiffusionSolverFE", "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
// 	&Simulator::steppableManager
// );

auto diffusionSolverFEProxy = registerPlugin<Steppable, DiffusionSolverFE_CPU>(
        "DiffusionSolverFE",
        "Solves diffusion equation on the lattice. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
        &Simulator::steppableManager
);

auto diffusionSolverFEImplicitProxy = registerPlugin<Steppable, DiffusionSolverFE_CPU_Implicit>(
        "DiffusionSolverFE_Implicit",
        "Solves diffusion equation on the lattice. Uses Implicit method - finite difference.",
        &Simulator::steppableManager
);

#if OPENCL_ENABLED == 1
auto diffusionSolverOpenCLProxy = registerPlugin<Steppable, DiffusionSolverFE_OpenCL>(
    "DiffusionSolverFE_OpenCL",
    "Solves diffusion equation on the lattice with OpenCL. Uses Forward Euler method - finite difference.  Also, uses automatic scaling.",
    &Simulator::steppableManager
);

// auto diffusionSolverOpenCLImplicitProxy = registerPlugin<Steppable, DiffusionSolverFE_OpenCL_Implicit>(
// 	"DiffusionSolverFE_OpenCL_Implicit",
// 	"Solves diffusion equation on the lattice with OpenCL. Uses Implicit method - finite difference.",
// 	&Simulator::steppableManager
// );

auto reactionDiffusionSolverOpenCLImplicitProxy = registerPlugin<Steppable, ReactionDiffusionSolverFE_OpenCL_Implicit>(
    "ReactionDiffusionSolverFE_OpenCL_Implicit",
    "Solves diffusion equation on the lattice with OpenCL. Uses Implicit method - finite difference.",
    &Simulator::steppableManager
);

#endif

// auto reactionDiffusionFileProxy = registerPlugin<Steppable, ReactionDiffusionFile>(
// 	"ReactionDiffusionFile",
// 	"Loads from files precalculated reaction-diffusion fields",
// 	&Simulator::steppableManager
// );
