#ifndef CURVATURE_EXPORT_H
#define CURVATURE_EXPORT_H

    #if defined(_WIN32)
      #ifdef CurvatureShared_EXPORTS
          #define CURVATURE_EXPORT __declspec(dllexport)
          #define CURVATURE_EXPIMP_TEMPLATE
      #else
          #define CURVATURE_EXPORT __declspec(dllimport)
          #define CURVATURE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CURVATURE_EXPORT
         #define CURVATURE_EXPIMP_TEMPLATE
    #endif

#endif
