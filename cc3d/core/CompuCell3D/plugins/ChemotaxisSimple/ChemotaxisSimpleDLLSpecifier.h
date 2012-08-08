#ifndef CHEMOTAXISSIMPLE_EXPORT_H
#define CHEMOTAXISSIMPLE_EXPORT_H

    #if defined(_WIN32)
      #ifdef ChemotaxisSimpleShared_EXPORTS
          #define CHEMOTAXISSIMPLE_EXPORT __declspec(dllexport)
          #define CHEMOTAXISSIMPLE_EXPIMP_TEMPLATE
      #else
          #define CHEMOTAXISSIMPLE_EXPORT __declspec(dllimport)
          #define CHEMOTAXISSIMPLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CHEMOTAXISSIMPLE_EXPORT
         #define CHEMOTAXISSIMPLE_EXPIMP_TEMPLATE
    #endif

#endif
