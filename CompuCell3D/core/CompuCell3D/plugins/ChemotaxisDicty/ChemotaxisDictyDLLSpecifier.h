#ifndef CHEMOTAXISDICTY_EXPORT_H
#define CHEMOTAXISDICTY_EXPORT_H

    #if defined(_WIN32)
      #ifdef ChemotaxisDictyShared_EXPORTS
          #define CHEMOTAXISDICTY_EXPORT __declspec(dllexport)
          #define CHEMOTAXISDICTY_EXPIMP_TEMPLATE
      #else
          #define CHEMOTAXISDICTY_EXPORT __declspec(dllimport)
          #define CHEMOTAXISDICTY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CHEMOTAXISDICTY_EXPORT
         #define CHEMOTAXISDICTY_EXPIMP_TEMPLATE
    #endif

#endif
