


#ifndef PLAYERSETTINGSPLUGIN_H
#define PLAYERSETTINGSPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/plugins/PlayerSettings/PlayerSettings.h>
#include "PlayerSettingsDLLSpecifier.h"


namespace CompuCell3D {


  class PLAYERSETTINGS_EXPORT PlayerSettingsPlugin : public Plugin {


	 CC3DXMLElement *xmlData;
  public:

    PlayerSettings playerSettings;
    PlayerSettings *playerSettingsPtr;

    PlayerSettingsPlugin();
    virtual ~PlayerSettingsPlugin();

    
    ///SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);
    
    PlayerSettings getPlayerSettings() const {return *(playerSettingsPtr);} 
	 
	//steerable interface
	virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
	virtual std::string steerableName();
	virtual std::string toString();    
  };


};
#endif


