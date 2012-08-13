#ifndef TEMPLATESTEPPABLE_EXPORT_H
#define TEMPLATESTEPPABLE_EXPORT_H

    #if defined(_WIN32)
      #ifdef PIFInitializerShared_EXPORTS
          #define TEMPLATESTEPPABLE_EXPORT __declspec(dllexport)
          #define TEMPLATESTEPPABLE_EXPIMP_TEMPLATE
      #else
          #define TEMPLATESTEPPABLE_EXPORT __declspec(dllimport)
          #define TEMPLATESTEPPABLE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define TEMPLATESTEPPABLE_EXPORT
         #define TEMPLATESTEPPABLE_EXPIMP_TEMPLATE
    #endif

#endif
