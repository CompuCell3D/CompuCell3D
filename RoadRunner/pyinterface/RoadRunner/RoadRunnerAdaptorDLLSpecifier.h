#ifndef ROADRUNNERADAPTOR_EXPORTS_H
#define ROADRUNNERADAPTOR_EXPORTS_H

    #if defined(_WIN32)
      #ifdef RoadRunnerAdaptor_EXPORTS
          #define ROADRUNNERADAPTOR_EXPORT __declspec(dllexport)
          #define ROADRUNNERADAPTOR_EXPIMP_TEMPLATE
      #else
          #define ROADRUNNERADAPTOR_EXPORT __declspec(dllimport)
          #define ROADRUNNERADAPTOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ROADRUNNERADAPTOR_EXPORT
         #define ROADRUNNERADAPTOR_EXPIMP_TEMPLATE
    #endif

#endif
