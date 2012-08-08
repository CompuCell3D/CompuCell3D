#ifndef FIELDEXTRACTOR_EXPORTS_H
#define FIELDEXTRACTOR_EXPORTS_H

    #if defined(_WIN32)
      #ifdef FieldExtractor_EXPORTS
          #define FIELDEXTRACTOR_EXPORT __declspec(dllexport)
          #define FIELDEXTRACTOR_EXPIMP_TEMPLATE
      #else
          #define FIELDEXTRACTOR_EXPORT __declspec(dllimport)
          #define FIELDEXTRACTOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define FIELDEXTRACTOR_EXPORT
         #define FIELDEXTRACTOR_EXPIMP_TEMPLATE
    #endif

#endif
