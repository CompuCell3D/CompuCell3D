Release Notes
=============

Version 4.2.2
-------------
**2019-07-24*

This is a bug-fix release featuring the following improvements:

Bug Fixes:

- Fixed saving of windows layout - including plots and steering panel
- Fixed screenshot color issues - as of now coloring is based on specification inside
screenshot description file, and not on current player settings
- Fixed handling of secretion in ReactionDiffusionSolverFE
- Fixed "Check for New Version" functionality in the player
- Fixed behavior of simulations that use plots but are run in gui-less mode
- Fixed Twedit++ zoom in / zoom out issues
- Fixed CC3D version printout issues

New Features:

- Added ability top open zipped .cc3d project directly from Twedit++ - no need to
do unzipping using 3rd party tools
- Added saving of simulation layout on simulation stop or simulation finish events
- Added automatic zip file name fill-in in Twedit++ when zipping .cc3d project
-Added Simulation menu action to reset global settings - as of now users can reset simulation-specific and global settings directly from Player menu
- Added option to reset Twedit++ settings directly from Twedit++ GUI


Version 4.2.1
-------------
**2019-05-18*

This is mainly bug-fix release featuring the following improvements:

- Added new convenience function to FieldSecretor class to compute total concentration "seen" by a cell
as well as total amount of field in the entire lattice

- Multiple bug fixes including:
    - Fixed Replay of saved simulation snapshots
    - Fixed simulation shutdown function call sequence to avoid crashes after last MCS was not a multiple of
    screen update frequency


Version 4.2.0
-------------
**2019-04-18*

The list of new features added in this release includes the following:

- Multiple bug fixes including:
    - fixing CC3D GUI behavior with multiple monitors
    - fixing contour lines plotting
    - fixing display of chemical/scalar fields
    - floating windows layout now supported on all platforms
    - dmg-based installer for OSX 10.14+. Solves previous issues with CC3D installations on newer OSX systems

- New floating layout that limits windows clutter (important for OSX users)

- Added persistent bias to Bias Vector Steppable

- Added Screenshot API

- Added cell type name accessor to Python steppable

- Added Fluctuation Compensator to DiffusionSolverFE and ReactionDiffusionSolverFE

- Added effective energy data Python accessor

- Added Focal Point Plasticity time tracking data

- Added Focal Point Plasticity link initiator data

- Added PDE test-suite

- Improvements to CallableCC3D module (input passing)

Known Issues:
- GPU solvers on OSX 10.14 or higher may not work properly


Version 4.1.1
-------------
**2019-01-18*

This release adds support for Antimony (see examples in Demos/SBMLSolverExamples/SBMLSolverAntimony)
and has also multiple bug-fixes:

- Fixed parameter scan to allow runs with multiple workers. See example script - Demos/ParameterScan/pscan_loop.sh
- Added callable API allowing CC3d to be called as a function returning values. See documentation and example in Demos/CallableCC3D.
- Fixed restart files issue
- fixed PIFF dumper
- fixed hover over text in Player
- Added support for developing custom C++ steppables and plugins on OSX - see
https://compucell3ddevelopersmanual.readthedocs.io/en/latest/setting_up_compiler_on_osx.html
- Improved compilation on linux , windows and osx but adding extra conda packages that fix issues
with incomplete packaging of vtk from conda-forge
- Expanded compilation documentation for all 3 platforms


Version 4.1.0
-------------
**2019-09-21**

This is mainly bug-fix release that fixes many of the issues we observed in 4.0.0.
In addition to this we also added the following features:

- New , intuitive way to launch parameter scans
- Added 3D vascularized tumor demo from Shirinifard PLoS One 2009
- Added basic, in-player simulation stats output
- Added "weightEnergyByDistance" in all contact energy plugins
- Expanded Developer's manual and added new , documented DeveloperZone steppables examples
- Added convenience Michaelis-Menten and Hill functions to SteppableBasePy
- Multiple bug fixes (including ability to resize screenshots)

Version 4.0.0
-------------
**2019-08-11**

Major version change migrated to Python 3.6+

- Python 3 - based code
- Much simpler specification of simulation - new , more intuitive API
- More intuitive specification of parameter scans
- Better support and integration with 3rd party Python packages (numpy, pandas, scipy)
- Multiple bug fixes

Version 3.7.7
-------------

**2017-11-12**

- Improved handling of Player settings - based on SQLite database
- Significantly faster connectivity plugin that works in 2D , 3D and on any type of lattice
- Multiple bug-fixes

Version 3.7.6
--------------

**2017-05-12**

- New PLayer - based on PyQt5
- New plotting backend based on PyQtGraph
- Multiple bug-fixes

Version 3.7.5
--------------

**2016-05-14**

- Improved player and many convenience features in Python scripting that make model development much easier.
- Windows versions ship with bundled Python distributions
- support for OSX 10.11 - ElCapitan
- Starting from this version we will be only supporting Long Term Support Ubuntu releases (12.04, 14.0 16.04 etc)
- Player has been improved and users can add axes
- RoadRunner was upgraded to the latest version. **IMPORTANT:** The RR upgrade eliminates
  the need to set steps options in in the Steppable file. If you have step options set remove it from your script


Version 3.7.4
--------------

**2015-05-17**

- Improved player and many convenience features in Python scripting that make model development much easier.
- Player has been improved and has new layout with floating windows. This is the default and recommended setting for Mac users
- Player settings are stored individually with each simulation.
  Thus several simulations running in parallel may have different set of settings.
  Previously there was one global setting file which made it
  inconvenient to run multiple simultaneous simulations with different settings
- Window layout is saved in the settings each time user stops the simulation.
  This feature allows simulation to open in exactly the same state it was before user stopped simulation run.
- Automatic cell labeling using scalar or vector cell attribute
- Simplified access to cell python dictionary - not you type cell.dict
- Simplified histograms and scientific plots setup
- Added ability to subscribe/unsubscribe to CompuCell3D mailing list from the Player

Version 3.7.3
--------------

**2014-09-14**

- paramScan script that runs parameter scan in a fault-tolerant way. Even if simulation crashes for whatever reason, the next one in the parameter scan will be started
- Added new format to save plot data (csv)
- Added hex2Cartesian and cartesin2Hex functions
- Added option to turn off comments in Python snippets inserted from CC3D Python menu
- Added support for VTK6
- Stopped requiring PyQt/Qt for command line runs
- Added some XML code checkers which do sanity checks for XML part of simulation description
- Fixed saving plots and plots data
- Fixed saving .cc3d projects in the new directory aka Save Project As ...
- Fixed visualization scaling for 2D projectsion on hex lattice
- Fixed generation of higher neighbor order on demand. Current implementation was good up to 8th nearest neighbor. Now we can use 20 or 30 or even higher
- Fixed how secretion plugin is handled in openMP - now when user does all secretion in Python there is no thread blocking in open mp to execute fixed stepper - see manual for more details

Version 3.7.2
-------------

**2014-07-04**

- Made secretion in the GPU solvers run on GPU not on CPU as before - performance gain
- Improved roadrunner SBML Solver - faster than before and with more user-configurable options
- Improved GPU and CPU PDE Solvers - fixed small bugs on hex lattice with non-periodic boundary conditions
- Updated Twedit helper menu
- Fixed OSX player freeze when replaying VTK files
- Added min/max functions to the chemical field for faster performance
- Fixed memory leaks in some field-accessing functions (swig-wrapped functions)
- Fixed GPU solvers for 3D
- Fixed Hex lattice solvers in general for 3D
- Fixed hex lattice transformation formulas for 3D - this might have been done already in 3.7.1
- Improved performance of GPU solvers
- Imiproved VTK file replay - now it runs smoothly on all platforms

Version 3.7.1
-------------

- LLVM-based RoadRunner as a backend for SBML Solver
- Parameter Scans
- Improved Twedit
- On Windows switched compilers from VS2008 to VS 2010
- Added Serialization of SBMLSolver objects
- Fixed memory leaks in the Player
- Added proper cleanup functions to Simulator
- Fixed sneaky bug related to cell inventory ordering - affected windows only and when cells were deleted it could cause CC3D crash. Same for FocalPOintPLasticity plugin ordering of the links was buggy on windows.

Version 3.7.0
-------------

- GPU Reaction-Diffusion Solvers (explicit and implicit)
- RoadRunner-basedSBMLsolvers
- Simplified and improved Steppable API (backward compatibility maintained)
- Numpy-based syntax for field manipulation
- Demo integration with Dolfin- works on linux only

Version 3.6.2
-------------

- Added CC3DML Helper to Twedit
- GPU Diffusion solver

Version 3.6.0
-------------

- Integrated Twedit++ with CC3D
- Added more functionality to plotting in CC3D, modified startup scripts related to twedit++
- Separated internal energies and external energies - all contact plugins by default include only
  terms from neighboring pixels belonging to different clusters.
  Added ContactInternal plugin which calculates energy between neighboring pixels belongint to
  different cells but within the same cluster. This allows replacement of Compartment plugin with
  combination of ContactInternal+Contact, ContactInternal+AdhesionFlex etc.

- modified clusterEnergy example to show how the new approach will work
- Added extra functionality to PySteppables SteppableBasePy module allowing simple cell manipulation and better access to cell within cluster
- Fixed Python iterators - see bug-fixes below for more details
- Bundled BionetSolver with CC3D - Windows OSX, coming soon
- Introduced new style CCC3D project files (as of now each CC3D simulation can be stored as a
  self-contained directory containing all the files necessaruy to run simulations).
  All file locations are w.r.t to directory containing main CC3D project file *.cc3d
- Introduced new storage place. By default all the simulations results are now saved to <homeDirectory>/CC3DWorkspace
- Added CC3D project management tool to Twedit ++
- Added CC3D simulation wizard to Twedit
- Added new boundary condition specification and a llowed mixed BC for most of
 the PDE's (Kernel and AdvectionDiffusion solver are not included in this change)
- Fixed instability issues in the SteadyStateDiffusionSolver associated with floats - Change solver to work with doubles
- Fixed the following problem:

SWIG has problems correctly generating/handling STL iterators (or in general any iterators)
Once there are more than one SWIG-generated modules loaded in Python and each of those modules contains STL containers
then iterators generated by SWIG () like those returneb by itervalues, iter, iterator iterkeys etc) will caus segfault during iteration
This is well documented below and here:

http://permalink.gmane.org/gmane.comp.programming.swig.devel/20140
//here is a reference found on the web to the bug in Swig
// # 1. Workaround for SWIG bug #1863647: Ensure that the PySwigIterator class
// #    (SwigPyIterator in 1.3.38 or later) is renamed with a module-specific
// #    prefix, to avoid collisions when using multiple modules
// # 2. If module names contain '.' characters, SWIG emits these into the CPP
// #    macros used in the director header. Work around this by replacing them
// #    with '_'. A longer term fix is not to call our modules "IMP.foo" but
// #    to say %module(package=IMP) foo but this doesn't work in SWIG stable
// #    as of 1.3.36 (Python imports incorrectly come out as 'import foo'
// #    rather than 'import IMP.foo'). See also IMP bug #41 at
// #    https://salilab.org/imp/bugs/show_bug.cgi?id=41

The bottom line is that instead of relying on SWIG to generate iterators for you it is
much better to write your own iterator wrapper like the one included in the CC3D code.
This is a bit of the overhead but not too much and if necessary it can be further simplified
(for the convenience of coding)

Version 3.5.0
-------------

- Added OpenMP support
- Added new algorithm to External potential - delta E can be now calculated based on changes in COM position
- Added functionality to SteppableBasePy - now it detects which Python available plugins are loaded and
  based on this it makes them callable directly from any steppable which inherits SteppableBasePy.
- Added COM based algorithm to cell orientation plugin
- Modified COM plugin to make center of mass coordinates easier to access without doing any calculations
- Reworked viscosity plugin, added new attributes to CellG - true COM coordinates and COM for one spin flip before
- Added Secretion Plugin which replaces (this is optional and up to modeler) secretion syntax of PDE solver.
 Secretion plugin has better functionality than secretion functions in PDE-solver
- Implemented Chemotaxis by cell id. "Per-cell" chemotaxis parameters override XML based definitions.
  Users still have to list in XML which fields participate in chemotaxis
- Implemented fluctuation amplitude on per-cell basis. Replaced "with" statement in Graphics/GraphicsFrameWidget.py
  with equivalent try/except statement
- Changed Temperature/Cell motility to FluctuationAmplitude - we still support old definitions
  however we should deprecate old terminology
- Added accessor functions to LengthConstraintLocalFlex/LengthConstraintLocalFlexPlugin.cpp
- Implemented text stream redirection so that output from C++ and Python can be displayed in Player console
- Fixed significant bug in parallel Potts section - had to allow nested omp regions as PDE solver caller calls
  PDE solver from within parallel section . PDESolver though instantiates its own parallel section to solve PDE
  so there are nested parallel regions





