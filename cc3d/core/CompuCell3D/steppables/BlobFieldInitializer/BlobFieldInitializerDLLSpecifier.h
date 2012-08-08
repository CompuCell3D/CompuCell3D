#ifndef BLOBFIELDINITIALIZER_EXPORT_H
#define BLOBFIELDINITIALIZER_EXPORT_H

    #if defined(_WIN32)
      #ifdef BlobFieldInitializerShared_EXPORTS
          #define BLOBFIELDINITIALIZER_EXPORT __declspec(dllexport)
          #define BLOBFIELDINITIALIZER_EXPIMP_TEMPLATE
      #else
          #define BLOBFIELDINITIALIZER_EXPORT __declspec(dllimport)
          #define BLOBFIELDINITIALIZER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define BLOBFIELDINITIALIZER_EXPORT
         #define BLOBFIELDINITIALIZER_EXPIMP_TEMPLATE
    #endif

#endif
