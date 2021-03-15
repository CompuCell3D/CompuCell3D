#ifndef CC3DMABOSS_EXPORT_H
#define CC3DMABOSS_EXPORT_H

    #if defined(_WIN32)
      #ifdef MaBoSSCC3DShared_EXPORTS
          #define CC3DMABOSS_EXPORT __declspec(dllexport)
          #define CC3DMABOSS_EXPIMP_TEMPLATE
      #else
          #define CC3DMABOSS_EXPORT __declspec(dllimport)
          #define CC3DMABOSS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define CC3DMABOSS_EXPORT
         #define CC3DMABOSS_EXPIMP_TEMPLATE
    #endif

#endif
