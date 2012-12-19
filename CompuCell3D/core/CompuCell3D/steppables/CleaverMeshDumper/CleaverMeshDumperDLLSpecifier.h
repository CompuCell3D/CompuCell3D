
#ifndef CLEAVERMESHDUMPER_EXPORT_H
#define CLEAVERMESHDUMPER_EXPORT_H

    #if defined(_WIN32)
      #ifdef CleaverMeshDumperShared_EXPORTS
          #define CLEAVERMESHDUMPER_EXPORT __declspec(dllexport)
          #define CLEAVERMESHDUMPER_EXPIMP_TEMPLATE
      #else
          #define CLEAVERMESHDUMPER_EXPORT __declspec(dllimport)
          #define CLEAVERMESHDUMPER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CLEAVERMESHDUMPER_EXPORT
         #define CLEAVERMESHDUMPER_EXPIMP_TEMPLATE
    #endif

#endif
