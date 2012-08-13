#ifndef CHEMOTAXIS_EXPORT_H
#define CHEMOTAXIS_EXPORT_H

    #if defined(_WIN32)
      #ifdef ChemotaxisShared_EXPORTS
          #define CHEMOTAXIS_EXPORT __declspec(dllexport)
          #define CHEMOTAXIS_EXPIMP_TEMPLATE
      #else
          #define CHEMOTAXIS_EXPORT __declspec(dllimport)
          #define CHEMOTAXIS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CHEMOTAXIS_EXPORT
         #define CHEMOTAXIS_EXPIMP_TEMPLATE
    #endif

#endif
