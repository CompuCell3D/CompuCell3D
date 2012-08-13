#ifndef PLAYERSETTINGS_EXPORT_H
#define PLAYERSETTINGS_EXPORT_H

    #if defined(_WIN32)
      #ifdef PlayerSettingsShared_EXPORTS
          #define PLAYERSETTINGS_EXPORT __declspec(dllexport)
          #define PLAYERSETTINGS_EXPIMP_TEMPLATE
      #else
          #define PLAYERSETTINGS_EXPORT __declspec(dllimport)
          #define PLAYERSETTINGS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PLAYERSETTINGS_EXPORT
         #define PLAYERSETTINGS_EXPIMP_TEMPLATE
    #endif

#endif
