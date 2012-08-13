#ifndef BOXWATCHER_EXPORT_H
#define BOXWATCHER_EXPORT_H

    #if defined(_WIN32)
      #ifdef BoxWatcherShared_EXPORTS
          #define BOXWATCHER_EXPORT __declspec(dllexport)
          #define BOXWATCHER_EXPIMP_TEMPLATE
      #else
          #define BOXWATCHER_EXPORT __declspec(dllimport)
          #define BOXWATCHER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define BOXWATCHER_EXPORT
         #define BOXWATCHER_EXPIMP_TEMPLATE
    #endif

#endif
