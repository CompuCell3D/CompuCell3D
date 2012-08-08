#ifndef FIPYINTERFACEEXTRACTOR_EXPORTS_H
#define FIPYINTERFACEEXTRACTOR_EXPORTS_H

    #if defined(_WIN32)
      #ifdef FieldExtractor_EXPORTS
          #define FIPYINTERFACEEXTRACTOR_EXPORT __declspec(dllexport)
          #define FIPYINTERFACEEXTRACTOR_EXPIMP_TEMPLATE
      #else
          #define FIPYINTERFACEEXTRACTOR_EXPORT __declspec(dllimport)
          #define FIPYINTERFACEEXTRACTOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define FIPYINTERFACEEXTRACTOR_EXPORT
         #define FIPYINTERFACEEXTRACTOR_EXPIMP_TEMPLATE
    #endif

#endif