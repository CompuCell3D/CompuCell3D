#ifndef PERSISTENCE_EXPORT_H
#define PERSISTENCE_EXPORT_H

    #if defined(_WIN32)
      #ifdef PersistenceShared_EXPORTS
          #define PERSISTENCE_EXPORT __declspec(dllexport)
          #define PERSISTENCE_EXPIMP_TEMPLATE
      #else
          #define PERSISTENCE_EXPORT __declspec(dllimport)
          #define PERSISTENCE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PERSISTENCE_EXPORT
         #define PERSISTENCE_EXPIMP_TEMPLATE
    #endif

#endif
