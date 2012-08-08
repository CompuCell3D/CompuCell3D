#ifndef MITOSIS_EXPORT_H
#define MITOSIS_EXPORT_H

    #if defined(_WIN32)
      #ifdef MitosisShared_EXPORTS
          #define MITOSIS_EXPORT __declspec(dllexport)
          #define MITOSIS_EXPIMP_TEMPLATE
      #else
          #define MITOSIS_EXPORT __declspec(dllimport)
          #define MITOSIS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define MITOSIS_EXPORT
         #define MITOSIS_EXPIMP_TEMPLATE
    #endif

#endif
