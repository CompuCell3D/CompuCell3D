//---------------------------------------------------------------------------
#include <string>
#include <vcl.h>
#pragma hdrstop
#include "Main.h"
#include "mtkVCLUtils.h"
#include "mtkFileUtils.h"
#include "mtkFileSystemTreeItems.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma link "TFileSelectionFrame"
#pragma link "mtkIniFileC"
#pragma resource "*.dfm"
TMainForm *MainForm;

using namespace mtk;
//---------------------------------------------------------------------------
__fastcall TMainForm::TMainForm(TComponent* Owner)
:
TForm(Owner),
mFormSaver(this, "MainForm", mIniFile->GetIniFile())
{
	mIniFile->Init();
	mIniFile->Load();
	mFormSaver.Read();
	mrrModelsRoot = "C:\\SBMLTestCases\\all";
	mOutputRoot  = "C:\\DataOutput";

	fsf->MonitorFolder(mrrModelsRoot,"*l2v4.xml");
	fsf->MonitorFolder(mOutputRoot,"*.c");
    fsf->ReScanDataFolderAExecute(NULL);

	fsf->TreeView1->OnClick    = fsfTreeView1Click;
    fsf->TreeView1->OnDblClick = fsfTreeView1DblClick;

    TMenuItem* loadXML = new TMenuItem(NULL);
    loadXML->Action = LoadModelA;
    fsf->TreeView1->PopupMenu->Items->Insert(0,loadXML);
	mRR.SetFileName("C:\\rrw\\installs\\xe\\bin\\simulate.exe");
    mRR.SetMessageHandling(CATCHMESSAGE);
}

__fastcall TMainForm::~TMainForm()
{
	mFormSaver.Write();
	mIniFile->Save();

}
string TMainForm::GetSelectedFileName()
{
	//Load XML file into XML memo
    string itemName;
    TTreeNode* aNode = fsf->GetSelected()->GetPrev();

    if(!aNode)
    {
    	return string("");
    }

    mtkFileSystemItem *info  = (mtkFileSystemItem*)(aNode->Data);    // cast data into mtkTreeItemBase pointer
    if(!info)
        return string("");
	mtkFileItem* 	fileItem 	= dynamic_cast<mtkFileItem*>(info);
    itemName = ToSTDString(fsf->GetSelected()->Text);

	string path = JoinPath(mrrModelsRoot, ToSTDString(aNode->Text));
	string fullName = JoinPath(path, itemName);
    return  mtk::FileExists(fullName) ? fullName : string("");

}
//---------------------------------------------------------------------------
void __fastcall TMainForm::LoadModelAExecute(TObject *Sender)
{
    string fName = GetSelectedFileName();

    if(fName.size())
    {
		Memo2->Lines->LoadFromFile(fName.c_str());
    }
}

void __fastcall TMainForm::fsfTreeView1DblClick(TObject *Sender)
{
	LoadModelAExecute(NULL);
}


void __fastcall TMainForm::TestModelAExecute(TObject *Sender)
{
	Log1->Clear();
    mtkProcess	rr(&mRR);

 	//Get test suite number from selected file
	string fName = ToSTDString(mModelFileName->Text);
    fName = ExtractFileName(fName);
    vector<string> parts = SplitString(fName,"-");

    //First part is the model number
    int nTheNumber = ToInt(parts[0]);

    stringstream paras;
    paras<<"-v"<<RGLogLevel->ItemIndex<<" -n"<<nTheNumber;


    if(compileOnlyCB->Checked)
    {
		paras<<" -c";
    }

    rr.Create(paras.str().c_str());
    rr.Run();
    vector<string> msg = rr.GetOutput();
    for(int i = 0; i < msg.size(); i++)
    {
    	Log1->Lines->Add(msg[i].c_str());
    }
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::FormKeyDown(TObject *Sender, WORD &Key, TShiftState Shift)
{
    if(Key == VK_ESCAPE)
    {
        Close();
    }
}

void __fastcall TMainForm::fsfTreeView1Click(TObject *Sender)
{
	//Fill out filename
    string name = GetSelectedFileName();
    if(name.size())
    {
    	mModelFileName->Text = name.c_str();
    }
}


