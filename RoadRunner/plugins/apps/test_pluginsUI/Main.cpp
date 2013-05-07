//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop
#include "rrUtils.h"
#include "Main.h"
#include "rrc_api.h"
#include "rrException.h"
#include "utils.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TMainF *MainF;

//---------------------------------------------------------------------------
__fastcall TMainF::TMainF(TComponent* Owner)
:
TForm(Owner)
{
	startupTimer->Enabled = true;
}
//---------------------------------------------------------------------------
void __fastcall TMainF::FormKeyDown(TObject *Sender, WORD &Key, TShiftState Shift)
{
	if(Key == VK_ESCAPE)
	{
		Close();
	}
}

void __fastcall TMainF::startupTimerTimer(TObject *Sender)
{
	startupTimer->Enabled = false;
	mTheAPI = createRRInstance();
	if(!mTheAPI)
	{

	}
	else
	{
    	char* fileName = "R:\\installs\\cg\\xe3\\debug\\models\\test_1.xml";
    	loadSBMLFromFile(mTheAPI, fileName);
		apiVersionLBL->Caption = getVersion(mTheAPI);
		buildDateLbl->Caption  = getBuildDate();
		buildTimeLbl->Caption  = getBuildTime();
		string info 		   = getInfo(mTheAPI);

		vector<string> lines = rr::SplitString(info, "\n");
		for(int i =0; i < lines.size(); i++)
		{
			infoMemo->Lines->Add(lines[i].c_str());
		}
	}
}

//---------------------------------------------------------------------------
void __fastcall TMainF::loadPluginsAExecute(TObject *Sender)
{
	if(!loadPlugins(mTheAPI))
	{
		Log() << "failed loading plugins..";
		throw Exception(getLastError());
	}

	//Populate list box with plugins
	RRStringArray* pluginNames = getPluginNames(mTheAPI);

	for(int i = 0; i < pluginNames->Count; i++)
	{
		pluginList->AddItem(pluginNames->String[i], NULL);
	}
	Log() << "Loaded plugins..";
    Button1->Action = unloadPlugins;
}

void __fastcall TMainF::unloadPluginsExecute(TObject *Sender)
{
	if(!unLoadPlugins(mTheAPI))
	{
		Log() << "failed un-loading plugins..";
		throw Exception(getLastError());
	}

    pluginList->Clear();
	Log() << "Un-Loaded plugins..";
    Button1->Action = loadPluginsA;
}

void __fastcall TMainF::ApplicationEvents1Exception(TObject *Sender, Exception *E)
{
	Log() << std_str(E->ToString()) <<endl;
}

void __fastcall TMainF::pluginListClick(TObject *Sender)
{
	//Retrieve the name of the plugin that was clicked
    if(pluginList->ItemIndex == -1)
    {
    	return;
    }

    string pluginName = std_str(pluginList->Items->Strings[pluginList->ItemIndex]);
    Log()<<std_str(pluginName);

    string test = getPluginInfo(mTheAPI, pluginName.c_str());

    infoMemo->Clear();
    Log()<<test;

    //Populate plugin frame (not a frame yet..)
	pluginCapsCB->Clear();
    pluginParasCB->Clear();

    RRStringArray* caps = getPluginCapabilities(mTheAPI, pluginName.c_str());


    if(!caps)
    {
	    GroupBox5->Enabled = false;
    	return;
    }

    GroupBox5->Enabled = true;
    for(int i =0; i < caps->Count; i++)
    {
        pluginCapsCB->AddItem(caps->String[i], NULL);
    }

    pluginCapsCB->ItemIndex = 0;
    pluginCBChange(pluginCapsCB);
}

void __fastcall TMainF::clearMemoExecute(TObject *Sender)
{
	infoMemo->Clear();
}
//---------------------------------------------------------------------------


void __fastcall TMainF::getPluginInfoAExecute(TObject *Sender)
{
	string pName = getCurrentPluginName();
    Log()<<getPluginInfo(mTheAPI, pName.c_str());
}

//---------------------------------------------------------------------------
string TMainF::getCurrentPluginName()
{
    if(pluginList->ItemIndex == -1)
    {
    	return "";
    }

    string pluginName = std_str(pluginList->Items->Strings[pluginList->ItemIndex]);
	return pluginName;
}

string TMainF::getCurrentSelectedParameter()
{
    if(pluginParasCB->ItemIndex == -1)
    {
    	return "";
    }

    string pluginParaName = std_str(pluginParasCB->Items->Strings[pluginParasCB->ItemIndex]);
	return pluginParaName;

}

void __fastcall TMainF::pluginCBChange(TObject *Sender)
{

	if(pluginCapsCB == (TComboBox*)(Sender))
    {
        string pluginName = std_str(pluginList->Items->Strings[pluginList->ItemIndex]);

		//Change available parameters in the parsCB...
		pluginParasCB->Clear();

        string capa = std_str(pluginCapsCB->Items->Strings[pluginCapsCB->ItemIndex]);
        RRStringArray* paras = getPluginParameters(mTheAPI, pluginName.c_str(), capa.c_str());
        pluginParasCB->Clear();

        if(!paras)
        {
            Log()<<"No parameters!";
            pluginCBChange(NULL);
            return;

        }

        for(int i =0; i < paras->Count; i++)
        {
            pluginParasCB->AddItem(paras->String[i], NULL);
        }
        pluginParasCB->ItemIndex = 0;


    }

    if(pluginParasCB == (TComboBox*)(Sender))
    {
        //Query the current plugin for the current value of selected parameter
        RRParameterHandle para = getPluginParameter(mTheAPI, getCurrentPluginName().c_str(), getCurrentSelectedParameter().c_str());
        if(!para)
        {
            paraEdit->Enabled = false;
            return;
        }

        paraEdit->Enabled = true;
        pluginParasCB->Hint = para->mHint;

        if(para->ParaType == ptInteger)
        {
            paraEdit->Text = rr::ToString(para->data.iValue).c_str();
        }
        else
        {
            paraEdit->Text = "not implemented";
        }
    }
}

//---------------------------------------------------------------------------
void __fastcall TMainF::SetParaBtnClick(TObject *Sender)
{
	setPluginParameter(mTheAPI, getCurrentPluginName().c_str(), getCurrentSelectedParameter().c_str(), std_str(paraEdit->Text).c_str());
}
//---------------------------------------------------------------------------

void __fastcall TMainF::executePluginAExecute(TObject *Sender)
{
	executePlugin(mTheAPI, getCurrentPluginName().c_str());
}
//---------------------------------------------------------------------------

void __fastcall TMainF::getLastErrorAExecute(TObject *Sender)
{
	Log()<<getLastError();
}
