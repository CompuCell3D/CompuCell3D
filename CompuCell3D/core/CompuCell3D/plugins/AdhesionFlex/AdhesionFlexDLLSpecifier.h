#ifndef ADHESIONFLEX_EXPORT_H
#define ADHESIONFLEX_EXPORT_H

    #if defined(_WIN32)
      #ifdef AdhesionFlexShared_EXPORTS
          #define ADHESIONFLEX_EXPORT __declspec(dllexport)
          #define ADHESIONFLEX_EXPIMP_TEMPLATE
      #else
          #define ADHESIONFLEX_EXPORT __declspec(dllimport)
          #define ADHESIONFLEX_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ADHESIONFLEX_EXPORT
         #define ADHESIONFLEX_EXPIMP_TEMPLATE
    #endif

#endif
