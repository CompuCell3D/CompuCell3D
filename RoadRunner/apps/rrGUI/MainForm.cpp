#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "MainForm.h"
#include "rrRoadRunner.h"
#include "rrLogger.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "mtkStopWatch.h"

//---------------------------------------------------------------------------
#pragma link "mtkFloatLabeledEdit"
#pragma link "mtkIniFileC"
#pragma link "mtkIntLabeledEdit"
#pragma link "TFileSelectionFrame"
#pragma link "mtkSTDStringEdit"
#pragma resource "*.dfm"

#pragma package(smart_init)

TMForm *MForm;
//---------------------------------------------------------------------------
using namespace rr;

__fastcall TMForm::TMForm(TComponent* Owner)
: TForm(Owner),
mLogFileSniffer("", this),
mSimulateThread(NULL, this),
mLogString(NULL)
{
    LogOutput::mLogToConsole = (false);
    LogOutput::mShowLogLevel = true;
    gLog.SetCutOffLogLevel(rr::lDebug5);
    mTempDataFolder = "R:\\temp";

    //This is roadrunners logger
    mRRLogFileName = rr::JoinPath(mTempDataFolder, "RoadRunnerUI.log");
    gLog.Init("", gLog.GetLogLevel(), new LogFile(mRRLogFileName ));

    //Setup a logfile sniffer and propagate logs to memo...
    mLogFileSniffer.SetFileName(mRRLogFileName);
    mLogFileSniffer.Start();
    SetupINIParameters();

    gLog.SetCutOffLogLevel(mLogLevel.GetValue());
    FSF->TreeView1->OnClick     =   FSFTreeView1Click;
	FSF->TreeView1->OnDblClick  =  LoadFromTreeViewAExecute;
	FSF->TreeView1->PopupMenu   =  TVPopupMenu;

    FSF->FSToolBar->Visible = false;
    FSF->TreeView1->ShowRoot = false;
    startupTimer->Enabled = true;

    //Setup road runner
    mRR = new RoadRunner("r:\\temp");
    mRR->setTempFileFolder(mTempDataFolder);
//    mSimulateThread.AssignRRInstance(mRR);
}

__fastcall TMForm::~TMForm()
{
	//FSF->TreeView1->Selected
    mLogLevel.SetValue(rr::GetLogLevel(LogLevelCB->ItemIndex));
    mPageControlHeight = PageControl1->Height;

    mConservationAnalysis = ConservationAnalysisCB->Checked ? true : false;

    mSelectionListHeight = SelList->Height;
    mGeneralParas.Write();
    mModelFolders.Write();
    mIniFileC->Save();
    delete mRR;
}

//---------------------------------------------------------------------------
void __fastcall TMForm::modelFoldersCBChange(TObject *Sender)
{
    Log(rr::lInfo)<<"Model folder is changing..";
}

//---------------------------------------------------------------------------
void __fastcall TMForm::modelFoldersCBSelect(TObject *Sender)
{
    if(modelFoldersCB->ItemIndex > -1 && modelFoldersCB->ItemIndex <= modelFoldersCB->Items->Count)
    {
        mCurrentModelsFolder = ToSTDString(modelFoldersCB->Text);
        FSF->RemoveMonitoredFolders();

        Log(rr::lInfo)<<"Model folder: "<<mCurrentModelsFolder<<" is selected..";

        FSF->MonitorFolder(mCurrentModelsFolder, filterEdit->GetString());
    	FSF->ReScanDataFolderAExecute(NULL);
    }
}

void __fastcall TMForm::selectModelsFolderExecute(TObject *Sender)
{
    //Browse for folder
    String folder = BrowseForDir(NULL);

    if(!folder.Length())
    {
        Log(rr::lInfo)<<"Bad folder...";
        return;
    }

    Log(rr::lInfo)<<"Selected folder "<<ToSTDString(folder.c_str());
    string fldr = ToSTDString(folder);
    fldr = RemoveTrailingSeparator(fldr, '\\');
    fldr = RemoveTrailingSeparator(fldr, '\\');
    if(!rr::FolderExists(fldr))
    {
        return;
    }

    mCurrentModelsFolder = ToSTDString(fldr.c_str());

    //Check if present in CBox
    int indx = modelFoldersCB->Items->IndexOf(folder) ;
    if(indx == -1)
    {
        modelFoldersCB->Items->Add(mCurrentModelsFolder.GetValueAsString().c_str());
        modelFoldersCB->ItemIndex = modelFoldersCB->Items->IndexOf(folder);
        mtkIniSection* folders = mIniFileC->GetSection("MODEL_FOLDERS");
        if(!folders)
        {
            if(mIniFileC->CreateSection("MODEL_FOLDERS"))
            {
                folders = mIniFileC->GetSection("MODEL_FOLDERS");
            }
        }

        if(folders)
        {
            string  keyName = "Item" + mtk::ToString(folders->KeyCount() + 1);
            folders->CreateKey(keyName, mCurrentModelsFolder);
        }
    }
}

//---------------------------------------------------------------------------
void __fastcall TMForm::LoadModelAExecute(TObject *Sender)
{
    LoadFromTreeViewAExecute(Sender);
    LoadModelA->Update();
}

void __fastcall TMForm::FSFTreeView1Click(TObject *Sender)
{
    //If a valid model file is selected, enable Load action
    string fName = FSF->GetSelectedFileInTree();
    if(rr::FileExists(fName))
    {
        LoadModelA->Enabled = true;
        UpdateTestSuiteInfo();
    }
    else
    {
        LoadModelA->Enabled = false;
    }
}

void __fastcall TMForm::LoadFromTreeViewAExecute(TObject *Sender)
{
    ClearMemoA->Execute();
    string fName = FSF->GetSelectedFileInTree();
    if(fName.size())
    {
        mCurrentModelFileName = fName;
        Log(rr::lInfo)<<"Loading model: "<<  fName;

        try
        {
            if(!mRR)
            {
                //delete mRR;
                mRR = new RoadRunner;
            }

            mRR->computeAndAssignConservationLaws(ConservationAnalysisCB->Checked);

            if(mRR->loadSBMLFromFile(fName, true))
            {
                Log(rr::lInfo)<<"Loaded model with no exception";
                loadAvailableSymbolsA->Execute();
                Log(rr::lInfo)<<mRR->getAvailableTimeCourseSymbols();

                //Enable simulate action
                SimulateA->Enabled = true;
                mModelNameLbl->Caption = mRR->getModelName().c_str();
            }
            else
            {
                Log(rr::lInfo)<<"There was problems loading model from file: "<<fName;
                SimulateA->Enabled = false;
            }
        }
        catch(const rr::Exception& ex)
        {
            Log(rr::lInfo)<<"RoadRunner Exception :"<<ex.what();
            SimulateA->Enabled = false;
        }

    }
}

void __fastcall TMForm::ClearMemoAExecute(TObject *Sender)
{
    mLogMemo->Clear();
}

//---------------------------------------------------------------------------
void __fastcall TMForm::SimulateAExecute(TObject *Sender)
{
    if(!mRR)
    {
        return;
    }
    try
    {
        //Setup selection list
        StringList list = GetCheckedSpecies();
        string selected = list.AsString();
        mRR->setTimeCourseSelectionList(selected);

        Log(rr::lInfo)<<"Currently selected species: "<<mRR->getTimeCourseSelectionList().AsString();

        mRR->simulateEx(mStartTimeE->GetValue(), *mEndTimeE, mNrOfSimulationPointsE->GetValue());

        SimulationData data = mRR->getSimulationResult();
        string resultFileName(rr::JoinPath(mRR->getTempFolder(), mRR->getModelName()));
        resultFileName = rr::ChangeFileExtensionTo(resultFileName, ".csv");

        Log(rr::lInfo)<<"Saving result to file: "<<resultFileName;
        ofstream fs(resultFileName.c_str());
        fs << data;
        fs.close();
        Plot(data);
   }
    catch(const rr::Exception& e)
    {
        Log(rr::lInfo)<<"RoadRunner exception: "<<e.what();
    }
}

StringList TMForm::GetCheckedSpecies()
{
    //Go trough the listbox and return checked items
    StringList checked;
    for(int i = 0; i < SelList->Count; i++)
    {
        if(SelList->Checked[i])
        {
            String anItem = SelList->Items->Strings[i];
            checked.Add(ToSTDString(anItem));
        }
    }
    return checked;
}

//---------------------------------------------------------------------------
void __fastcall TMForm::loadAvailableSymbolsAExecute(TObject *Sender)
{
    if(mRR)
    {
        SelList->Clear();
        string settingsFile = GetSettingsFile();
        if(FileExists(settingsFile))
        {
            if(mSettings.LoadFromFile(settingsFile))
            {
                mStartTimeE->SetNumber(mSettings.mStartTime);
                mEndTimeE->SetNumber(mSettings.mEndTime);
                mNrOfSimulationPointsE->SetNumber(mSettings.mSteps + 1);
                StringList symbols = getSelectionListFromSettings(mSettings);
                AddItemsToListBox(symbols);
            }
        }
        else
        {
        	SelList->Items->Add("Time");
	        SelList->Checked[0] = true;
            NewArrayList	symbols = mRR->getAvailableTimeCourseSymbols();
            Log(rr::lInfo)<<symbols;
            StringList fs       = symbols.GetStringList("Floating Species");
            StringList bs       = symbols.GetStringList("Boundary Species");
            StringList vols     = symbols.GetStringList("Volumes");
            StringList gp       = symbols.GetStringList("Global Parameters");
            StringList fluxes   = symbols.GetStringList("Fluxes");

            AddItemsToListBox(fs);
            AddItemsToListBox(bs);
            AddItemsToListBox(vols);
            AddItemsToListBox(gp);
            AddItemsToListBox(fluxes);
        }
        CheckUI();
    }
}

string TMForm::GetSettingsFile()
{
    string file =  FSF->GetSelectedFileInTree();
    string path =  rr::ExtractFilePath(file);
    vector<string> dirs = rr::SplitString(path,"\\");

    if(dirs.size())
    {
        string caseNr = dirs[dirs.size() -1];
        string setFile = rr::JoinPath(path, (caseNr + "-settings.txt"));
        return setFile;
    }
    return "";
}

string TMForm::GetCurrentModelPath()
{
    string file =  FSF->GetSelectedFileInTree();
    return  rr::ExtractFilePath(file);
}

void TMForm::AddItemsToListBox(const StringList& items)
{
    Log(rr::lInfo)<<items;

    for(int i = 0; i < items.Count(); i++)
    {
        SelList->Items->Add(items[i].c_str());
        SelList->Checked[i] = true;
    }
}

void __fastcall TMForm::PlotFromThread()
{
	if(mData)
    {
		Plot(*mData);
        delete mData;
        mData = NULL;
    }
    else
    {
    	Log(rr::lWarning)<<"Tried to plot NULL data";
    }
}
void TMForm::Plot(const rr::SimulationData& result)
{
    Chart1->RemoveAllSeries();

    //Fill out data for all series
    Log(rr::lDebug4)<<"Simulation Result"<<result;
    int nrOfSeries = result.cSize() -1; //First one is time
    StringList colNames = result.getColumnNames();
    vector<TLineSeries*> series;
    for(int i = 0; i < nrOfSeries; i++)
    {
        TLineSeries* aSeries = new TLineSeries(Chart1);
        aSeries->Title = colNames[i+1].c_str();
        aSeries->Color = GetColor(i);
        aSeries->LinePen->Width = 3;
        series.push_back(aSeries);
        Chart1->AddSeries(aSeries);
    }

    for(int j = 0; j < result.rSize(); j++)
    {
        double xVal = result(j,0);

        for(int i = 0; i < nrOfSeries; i++)
        {
            double yData = result(j, i+1);
            series[i]->AddXY(xVal, yData);
        }
    }
    Chart1->Update();
}

void __fastcall TMForm::PlotTestTestSuiteDataExecute(TObject *Sender)
{
    //Read the test suite data
    string file =  FSF->GetSelectedFileInTree();
    string path =  rr::ExtractFilePath(file);
    vector<string> dirs = rr::SplitString(path,"\\");
    string tsDataFile;
    if(dirs.size())
    {
        string caseNr = dirs[dirs.size() -1];
        tsDataFile = rr::JoinPath(path, (caseNr + "-results.csv"));
    }

    SimulationData result;
    if(!result.load(tsDataFile))
    {
        return;
    }

    //Fill out data for all series
    int nrOfSeries = result.cSize() -1; //First one is time
    StringList colNames = result.getColumnNames();
    vector<TLineSeries*> series;
    for(int i = 0; i < nrOfSeries; i++)
    {
        TLineSeries* aSeries = new TLineSeries(Chart1);
        aSeries->Title = colNames[i+1].c_str();
        aSeries->Color = clGray;
        aSeries->LinePen->Width = 1;
        aSeries->Pointer->Visible = true;
        series.push_back(aSeries);
        Chart1->AddSeries(aSeries);
    }

    for(int j = 0; j < result.rSize(); j++)
    {
        double xVal = result(j,0);

        for(int i = 0; i < nrOfSeries; i++)
        {
            double yData = result(j, i+1);
            series[i]->AddXY(xVal, yData);
        }
    }
    Chart1->Update();

}

void __fastcall TMForm::CheckUI()
{
    //Check if there is at least one checked species in the list box
    bool hasOneSelected = false;

    for(int i = 0; i < SelList->Count; i++)
    {
        if(SelList->Checked[i])
        {
            hasOneSelected = true;
            break;
        }
    }

    EnableDisableSimulation(hasOneSelected);
}

//---------------------------------------------------------------------------
void TMForm::EnableDisableSimulation(bool enableDisable)
{
    if(enableDisable)
    {
        Log(rr::lInfo)<<"Enabling simulation..";
    }
    else
    {
        Log(rr::lInfo)<<"Disabling simulation..";
    }
    mStartTimeE->Enabled            = enableDisable;
    mEndTimeE->Enabled              = enableDisable;
    mNrOfSimulationPointsE->Enabled = enableDisable;
    SimulateA->Enabled              = enableDisable;
}

void __fastcall TMForm::SelListClick(TObject *Sender)
{
    CheckUI();
}

void __fastcall TMForm::LoadModelAUpdate(TObject *Sender)
{
    if(mRR && mRR->isModelLoaded())
    {
        LoadModelA->Enabled = false;
        UnLoadModelA->Enabled = true;
        loadUnloadBtn->Action = UnLoadModelA;
    }
    else
    {
        UnLoadModelA->Enabled = false;
        loadUnloadBtn->Action = LoadModelA;
        //Check if there is a valid selection in the tree list
        //FSFTreeView1Click(NULL);
    }
}

void __fastcall TMForm::UnLoadModelAExecute(TObject *Sender)
{
    if(!mRR)
    {
        return;
    }

    mRR->unLoadModel();
    LoadModelA->Update();
}

void __fastcall TMForm::filterEditKeyDown(TObject *Sender, WORD &Key, TShiftState Shift)
{
    if(Key == VK_RETURN)
    {
        modelFoldersCBSelect(NULL);
    }
}

void __fastcall TMForm::Button4Click(TObject *Sender)
{
    FSF->TreeView1->ShowRoot = !FSF->TreeView1->ShowRoot;
}

void __fastcall TMForm::LogCurrentDataAExecute(TObject *Sender)
{
    if(mRR)
    {
        SimulationData data = mRR->getSimulationResult();
        Log(rr::lInfo)<<data;
    }
}

void __fastcall TMForm::LogLevelCBChange(TObject *Sender)
{
    gLog.SetCutOffLogLevel(rr::GetLogLevel(LogLevelCB->ItemIndex));
}

void __fastcall TMForm::UpdateTestSuiteInfo()
{
    string file =  FSF->GetSelectedFileInTree();
    string path =  rr::ExtractFilePath(file);
    vector<string> dirs = rr::SplitString(path,"\\");

    if(dirs.size())
    {
        string caseNr = dirs[dirs.size() -1];
        string htmlDoc = rr::JoinPath(path, (caseNr + "-model.html"));
        if(rr::FileExists(htmlDoc)) //This is a test suite folder
        {
            //If this is a testsuite folder.. show the http
            WebBrowser1->Navigate(htmlDoc.c_str());
            string aFile = rr::JoinPath(path, (caseNr + "-plot.jpg"));
            //Picture..
            if(rr::FileExists(aFile))
            {
                testSuitePic->Picture->LoadFromFile(aFile.c_str());
            }

            //Open and load settings
            aFile = rr::JoinPath(path, (caseNr + "-settings.txt"));
            if(rr::FileExists(aFile))
            {
                vector<string> fContent = rr::GetLinesInFile(aFile);
                Log(rr::lInfo)<<"Model Settings:\n"<<fContent;
            }
        }
        else
        {
            //Disable.....
        }
    }
}

void __fastcall TMForm::LogCCodeAExecute(TObject *Sender)
{
    string cCode;
    if(mRR)
    {
        cCode = mRR->getCSourceCode();
        vector<string> lines(rr::SplitString(cCode, "\n"));
        for(int i = 0; i < lines.size(); i++)
        {
            string aLine = lines[i];
            Log(rr::lInfo)<<i<<": "<<aLine;
        }
    }
}

void __fastcall TMForm::RemoveCurrentModelFolderItemAExecute(TObject *Sender)
{
    modelFoldersCB->Items->Delete(modelFoldersCB->ItemIndex);
}

//---------------------------------------------------------------------------
void __fastcall TMForm::modelFoldersCBContextPopup(TObject *Sender, TPoint &MousePos,
          bool &Handled)
{
    DropBoxPopup->Popup(0,0);
}

//---------------------------------------------------------------------------
void __fastcall TMForm::FormCloseQuery(TObject *Sender, bool &CanClose)
{
	if(mLogFileSniffer.IsAlive())
    {
    	CanClose = false;
    }

    if(!CanClose)
    {
		ShutDownTimer->Enabled = true;
    }
}

//---------------------------------------------------------------------------
void __fastcall TMForm::ShutDownTimerTimer(TObject *Sender)
{
	ShutDownTimer->Enabled = false;

	if(mLogFileSniffer.IsAlive())
    {
    	mLogFileSniffer.ShutDown();
    }

    Close();
}

//---------------------------------------------------------------------------
void __fastcall TMForm::FormClose(TObject *Sender, TCloseAction &Action)
{
	if(FSF)
    {
		FSF->ClearTree();
    }
}

//---------------------------------------------------------------------------
void __fastcall TMForm::Button5Click(TObject *Sender)
{
	int average = 0;

    mtkStopWatch sw;
    for(int i = 0; i < runCount->GetNumber(); i++)
    {
    	sw.Start();
		mRR->simulateEx(mStartTimeE->GetValue(), mEndTimeE->GetValue(), mNrOfSimulationPointsE->GetValue());
		int milliSecondsElapsed = sw.Stop();

        average += milliSecondsElapsed;
    	stringstream msg;
    	msg<<"Time for run "<<i<<": "<<fixed<<setprecision(15)<<milliSecondsElapsed<<" average: "<<(double) average/ (i + 1);
    	runCountMemo->Lines->Add(msg.str().c_str());
    }
}

void __fastcall TMForm::CheckThreadTimerTimer(TObject *Sender)
{
	if(mSimulateThread.isWorking())
    {
    	RunThreadBtn->Caption = "Stop";
    }
    else
    {
    	RunThreadBtn->Caption = "Run";
    }
}

//---------------------------------------------------------------------------
void __fastcall TMForm::RunThreadBtnClick(TObject *Sender)
{
	if(mSimulateThread.isWorking())
    {
    	mSimulateThread.exit();
    }
    else
    {
		mSimulateThread.start();
    }
}

//---------------------------------------------------------------------------
void __fastcall TMForm::ConservationAnalysisCBClick(TObject *Sender)
{
    LoadFromTreeViewAExecute(Sender);
    LoadModelA->Update();
}


