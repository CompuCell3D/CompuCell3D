#ifndef CONTACTMULTICAD_EXPORT_H
#define CONTACTMULTICAD_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactMultiCadShared_EXPORTS
          #define CONTACTMULTICAD_EXPORT __declspec(dllexport)
          #define CONTACTMULTICAD_EXPIMP_TEMPLATE
      #else
          #define CONTACTMULTICAD_EXPORT __declspec(dllimport)
          #define CONTACTMULTICAD_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACTMULTICAD_EXPORT
         #define CONTACTMULTICAD_EXPIMP_TEMPLATE
    #endif

#endif
