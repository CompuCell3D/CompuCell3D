What are these dll's for?


The RoadRunner library depends on libsbml, which in turn depends on various 3rd party libraries, e.g. libxml2, iconv etc..

Currently, libsbml is not statically linked to these third party libs, requiring distribution of these dll's being present on the target system. 

These dll's are copied here, and installed to a final bin folder by the CMake system. If this were not done, the end user would have to do that manually.