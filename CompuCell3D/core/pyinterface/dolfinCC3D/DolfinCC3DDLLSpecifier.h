#ifndef DOLFINCC3D_EXPORT_H
#define DOLFINCC3D_EXPORT_H

    #if defined(_WIN32)
      #ifdef dolfinCC3D_EXPORTS
          #define DOLFINCC3D_EXPORT __declspec(dllexport)
          #define DOLFINCC3D_EXPIMP_TEMPLATE
      #else
          #define DOLFINCC3D_EXPORT __declspec(dllimport)
          #define DOLFINCC3D_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define DOLFINCC3D_EXPORT
         #define DOLFINCC3D_EXPIMP_TEMPLATE
    #endif

#endif
