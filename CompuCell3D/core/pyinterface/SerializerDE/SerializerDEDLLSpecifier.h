#ifndef SERIALIZERDE_EXPORTS_H
#define SERIALIZERDE_EXPORTS_H

    #if defined(_WIN32)
      #ifdef SerializerDE_EXPORTS
          #define SERIALIZERDE_EXPORT __declspec(dllexport)
          #define SERIALIZERDE_EXPIMP_TEMPLATE
      #else
          #define SERIALIZERDE_EXPORT __declspec(dllimport)
          #define SERIALIZERDE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define SERIALIZERDE_EXPORT
         #define SERIALIZERDE_EXPIMP_TEMPLATE
    #endif

#endif
