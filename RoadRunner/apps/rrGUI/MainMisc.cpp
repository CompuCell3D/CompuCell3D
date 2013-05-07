#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrLogger.h"
#include "rrUtils.h"
#include "MainForm.h"
#include "rrStringUtils.h"
#include "mtkTreeComponentUtils.h"
#pragma package(smart_init)

using namespace rr;
void __fastcall TMForm::SetupINIParameters()
{
    mIniFileC->Init();

    //General Parameters
    mGeneralParas.SetIniSection("GENERAL");
    mGeneralParas.SetIniFile(mIniFileC->GetFile());
    mGeneralParas.Insert( &mStartTimeE->SetupIni("START_TIME", 0));
    mGeneralParas.Insert( &mEndTimeE->SetupIni("END_TIME", 40));
    mGeneralParas.Insert( &mNrOfSimulationPointsE->SetupIni("NR_OF_SIMULATION_POINTS", 100));

    mGeneralParas.Insert( &mPageControlHeight.Setup("LOWER_PAGE_CONTROL_HEIGHT", 300));
    mGeneralParas.Insert( &mSelectionListHeight.Setup("SEL_LB_HEIGHT", 30));
    mGeneralParas.Insert( &mConservationAnalysis.Setup("CONSERVATION_ANALYSIS", "false"));
    mGeneralParas.Insert( &mCurrentModelsFolder.Setup("MODEL_FOLDER", ""));
    mGeneralParas.Insert( &mCurrentModelFileName.Setup("MODEL_FILE_NAME", ""));
    mGeneralParas.Insert( &mLogLevel.Setup("LOG_LEVEL", rr::lInfo));
    mModelFolders.SetIniSection("MODEL_FOLDERS");
    mModelFolders.SetIniFile(mIniFileC->GetFile());

    mIniFileC->Load();
    mGeneralParas.Read();
    mModelFolders.Read();

    Log(rr::lInfo)<<"Reading settings..";
    mtkIniSection* folders = mIniFileC->GetSection("MODEL_FOLDERS");
    if(folders)
    {
        //Fill out combo box
        for(int i = 0; i < folders->KeyCount(); i++)
        {
            mtkIniKey* aKey = folders->mKeys[i];
            Log(rr::lInfo)<<*aKey;
            modelFoldersCB->Items->Add(aKey->mValue.c_str());
        }
    }
    //Select 'current' model folder
    int index = modelFoldersCB->Items->IndexOf(mCurrentModelsFolder.GetValueAsString().c_str());

    if(index != -1)
    {
        modelFoldersCB->ItemIndex = index;
    }
    else
    {
        if (modelFoldersCB->Items->Count)
        {
            modelFoldersCB->ItemIndex = 0;
        }
    }
    //Update UI
    mStartTimeE->Update();
    mEndTimeE->Update();
    mNrOfSimulationPointsE->Update();
    SelList->Height = mSelectionListHeight;
    PageControl1->Height = mPageControlHeight;
    LogLevelCB->ItemIndex = mLogLevel;
    ConservationAnalysisCB->Checked = mConservationAnalysis == "true" ? true : false;
}

void __fastcall TMForm::LogMessage()
{
    if(mLogString)
    {
        mLogMemo->Lines->Add(mLogString->c_str());
        delete mLogString;
        // Signal to the data sink thread that we can now accept another message...
        mLogString = NULL;
    }
}
//---------------------------------------------------------------------------
void __fastcall TMForm::FormKeyDown(TObject *Sender, WORD &Key, TShiftState Shift)
{
    if(Key == VK_ESCAPE)
    {
        Close();
    }
}

void __fastcall TMForm::startupTimerTimer(TObject *Sender)
{
    startupTimer->Enabled = false;
    //Select  models folder
    modelFoldersCBSelect(NULL);

    TTreeNode* aNode = FindTreeNodeBasedOnLabel(FSF->TreeView1->Items, rr::ExtractFileName(mCurrentModelFileName).c_str());

    if(aNode)
    {
        aNode->Selected = true;
    }
}

void __fastcall TMForm::logModelFileAExecute(TObject *Sender)
{
    string fName = FSF->GetSelectedFileInTree();
    if(fName.size())
    {
        Log(rr::lInfo)<<"Model File: "<<fName;
        if(!rr::FileExists(fName))
        {
            return;
        }

        ifstream aFile(fName.c_str());
        string  str((std::istreambuf_iterator<char>(aFile)), std::istreambuf_iterator<char>());

        vector<string> strings = rr::SplitString(str,"\n");
        for(int i = 0; i < strings.size(); i++)
        {
            Log(rr::lInfo)<<strings[i];
        }
    }
}

TColor TMForm::GetColor(int i)
{
    switch(i)
    {
        case 0: return clRed;
        case 1: return clGreen;
        case 2: return clBlue;
        case 3: return clPurple;
        case 4: return clOlive;
        case 5: return clCream;
        case 6: return clBlack;
        default: return clBlue;
    }
}
