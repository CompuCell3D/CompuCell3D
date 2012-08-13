#ifndef FOAMDATAOUTPUT_EXPORT_H
#define FOAMDATAOUTPUT_EXPORT_H

    #if defined(_WIN32)
      #ifdef FoamDataOutputShared_EXPORTS
          #define FOAMDATAOUTPUT_EXPORT __declspec(dllexport)
          #define FOAMDATAOUTPUT_EXPIMP_TEMPLATE
      #else
          #define FOAMDATAOUTPUT_EXPORT __declspec(dllimport)
          #define FOAMDATAOUTPUT_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define FOAMDATAOUTPUT_EXPORT
         #define FOAMDATAOUTPUT_EXPIMP_TEMPLATE
    #endif

#endif
