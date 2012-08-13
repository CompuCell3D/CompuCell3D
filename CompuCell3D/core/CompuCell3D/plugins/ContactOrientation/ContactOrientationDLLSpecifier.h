
#ifndef CONTACTORIENTATION_EXPORT_H
#define CONTACTORIENTATION_EXPORT_H

    #if defined(_WIN32)
      #ifdef ContactOrientationShared_EXPORTS
          #define CONTACTORIENTATION_EXPORT __declspec(dllexport)
          #define CONTACTORIENTATION_EXPIMP_TEMPLATE
      #else
          #define CONTACTORIENTATION_EXPORT __declspec(dllimport)
          #define CONTACTORIENTATION_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONTACTORIENTATION_EXPORT
         #define CONTACTORIENTATION_EXPIMP_TEMPLATE
    #endif

#endif
