# Compiled ClusterSurface demo

This demo is based on `CompuCellPythonTutorial/clusterSurface`. It initializes
the same compartmentalized cell from a PIFF file and uses a compiled steppable
to set and read the ClusterSurface target and lambda values through the C ABI.

The mitosis portion of the Python tutorial is intentionally omitted.

Build the native module with the compiled-steppable helper, then run
`ClusterSurfaceCompiledDemo.cc3d`. The native output prints the values read
back for individual cells at each compiled-steppable call.
