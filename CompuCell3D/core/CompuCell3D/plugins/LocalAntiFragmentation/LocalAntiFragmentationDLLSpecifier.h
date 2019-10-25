

#ifndef LOCALANTIFRAGMENTATION_EXPORT_H

#define LOCALANTIFRAGMENTATION_EXPORT_H



    #if defined(_WIN32)

      #ifdef LocalAntiFragmentationShared_EXPORTS

          #define LOCALANTIFRAGMENTATION_EXPORT __declspec(dllexport)

          #define LOCALANTIFRAGMENTATION_EXPIMP_TEMPLATE

      #else

          #define LOCALANTIFRAGMENTATION_EXPORT __declspec(dllimport)

          #define LOCALANTIFRAGMENTATION_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define LOCALANTIFRAGMENTATION_EXPORT

         #define LOCALANTIFRAGMENTATION_EXPIMP_TEMPLATE

    #endif



#endif

