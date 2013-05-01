
#ifndef CGALMESHDUMPER_EXPORT_H
#define CGALMESHDUMPER_EXPORT_H

    #if defined(_WIN32)
      #ifdef CGALMeshDumperShared_EXPORTS
          #define CGALMESHDUMPER_EXPORT __declspec(dllexport)
          #define CGALMESHDUMPER_EXPIMP_TEMPLATE
      #else
          #define CGALMESHDUMPER_EXPORT __declspec(dllimport)
          #define CGALMESHDUMPER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CGALMESHDUMPER_EXPORT
         #define CGALMESHDUMPER_EXPIMP_TEMPLATE
    #endif

#endif
