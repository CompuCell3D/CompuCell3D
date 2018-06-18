
#ifndef NEIGHBORSURFACECONSTRAINT_EXPORT_H
#define NEIGHBORSURFACECONSTRAINT_EXPORT_H

    #if defined(_WIN32)
      #ifdef NeighborSurfaceConstraintShared_EXPORTS
          #define NEIGHBORSURFACECONSTRAINT_EXPORT __declspec(dllexport)
          #define NEIGHBORSURFACECONSTRAINT_EXPIMP_TEMPLATE
      #else
          #define NEIGHBORSURFACECONSTRAINT_EXPORT __declspec(dllimport)
          #define NEIGHBORSURFACECONSTRAINT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NEIGHBORSURFACECONSTRAINT_EXPORT
         #define NEIGHBORSURFACECONSTRAINT_EXPIMP_TEMPLATE
    #endif

#endif
