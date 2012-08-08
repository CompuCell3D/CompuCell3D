#ifndef CONVERGENTEXTENSION_EXPORT_H
#define CONVERGENTEXTENSION_EXPORT_H

    #if defined(_WIN32)
      #ifdef ConvergentExtensionShared_EXPORTS
          #define CONVERGENTEXTENSION_EXPORT __declspec(dllexport)
          #define CONVERGENTEXTENSION_EXPIMP_TEMPLATE
      #else
          #define CONVERGENTEXTENSION_EXPORT __declspec(dllimport)
          #define CONVERGENTEXTENSION_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CONVERGENTEXTENSION_EXPORT
         #define CONVERGENTEXTENSION_EXPIMP_TEMPLATE
    #endif

#endif
