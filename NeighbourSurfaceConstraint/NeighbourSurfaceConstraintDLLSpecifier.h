
#ifndef NEIGHBOURSURFACECONSTRAINT_EXPORT_H
#define NEIGHBOURSURFACECONSTRAINT_EXPORT_H

    #if defined(_WIN32)
      #ifdef NeighbourSurfaceConstraintShared_EXPORTS
          #define NEIGHBOURSURFACECONSTRAINT_EXPORT __declspec(dllexport)
          #define NEIGHBOURSURFACECONSTRAINT_EXPIMP_TEMPLATE
      #else
          #define NEIGHBOURSURFACECONSTRAINT_EXPORT __declspec(dllimport)
          #define NEIGHBOURSURFACECONSTRAINT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define NEIGHBOURSURFACECONSTRAINT_EXPORT
         #define NEIGHBOURSURFACECONSTRAINT_EXPIMP_TEMPLATE
    #endif

#endif
