	
#ifndef SOSLIB_CVODESETTINGS_H
#define SOSLIB_CVODESETTINGS_H
#include "BionetworkDLLSpecifier.h"

typedef struct cvodeSettings cvodeSettings_t;

   class BIONETWORK_EXPORT soslib_CvodeSettings{
        private:
            cvodeSettings_t *settings;
            double nextTimeStep;
        public:
            soslib_CvodeSettings();
            soslib_CvodeSettings(double);
            soslib_CvodeSettings(double, unsigned int);
            soslib_CvodeSettings* cloneSettings() const ;
            ~soslib_CvodeSettings();
            cvodeSettings_t* getSettings() const ;
            void setSettings(cvodeSettings_t *);
            void createSettings();
            void createSettings(double, unsigned int);
            void printSettings();
            
            double getTimeStep() const ;
            void setTimeStep(double);
            
            double getEndTime() const ;
            void setEndTime(double);
            
            int getPrintSteps() const;
            void setPrintSteps(int);
            
            void setIndefiniteIntegration(int);
            int indefiniteIntegrationIsSet() const;
   };

#endif




   