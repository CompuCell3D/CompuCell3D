#ifndef FOCALPOINTPLASTICITY_EXPORT_H
#define FOCALPOINTPLASTICITY_EXPORT_H

    #if defined(_WIN32)
      #ifdef FocalPointPlasticityShared_EXPORTS
          #define FOCALPOINTPLASTICITY_EXPORT __declspec(dllexport)
          #define FOCALPOINTPLASTICITY_EXPIMP_TEMPLATE
      #else
          #define FOCALPOINTPLASTICITY_EXPORT __declspec(dllimport)
          #define FOCALPOINTPLASTICITY_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define FOCALPOINTPLASTICITY_EXPORT
         #define FOCALPOINTPLASTICITY_EXPIMP_TEMPLATE
    #endif

#endif
