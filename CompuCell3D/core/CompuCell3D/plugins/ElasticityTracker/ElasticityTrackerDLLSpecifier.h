#ifndef ELASTICITYTRACKER_EXPORT_H
#define ELASTICITYTRACKER_EXPORT_H

    #if defined(_WIN32)
      #ifdef ElasticityTrackerShared_EXPORTS
          #define ELASTICITYTRACKER_EXPORT __declspec(dllexport)
          #define ELASTICITYTRACKER_EXPIMP_TEMPLATE
      #else
          #define ELASTICITYTRACKER_EXPORT __declspec(dllimport)
          #define ELASTICITYTRACKER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define ELASTICITYTRACKER_EXPORT
         #define ELASTICITYTRACKER_EXPIMP_TEMPLATE
    #endif

#endif
