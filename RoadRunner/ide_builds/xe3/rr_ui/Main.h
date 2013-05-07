//---------------------------------------------------------------------------

#ifndef MainH
#define MainH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include <Controls.hpp>
#include <StdCtrls.hpp>
#include <Forms.hpp>
#include "TFileSelectionFrame.h"
#include <ExtCtrls.hpp>
#include <ActnList.hpp>
#include <ComCtrls.hpp>
#include "mtkIniFileC.h"
#include <string>
#include "mtkFormSaver.h"
#include "mtkExeFile.h"
#include "mtkProcess.h"
using namespace std;
//---------------------------------------------------------------------------
class TMainForm : public TForm
{
__published:	// IDE-managed Components
	TGroupBox *GroupBox1;
	TMemo *Memo1;
	TFileSelectionFrame *fsf;
	TPanel *Panel1;
	TPanel *Panel2;
	TSplitter *Splitter1;
	TPageControl *PageControl1;
	TTabSheet *TabSheet1;
	TTabSheet *TabSheet2;
	TMemo *Memo2;
	TButton *Button1;
	TActionList *ActionList1;
	TAction *LoadModelA;
	TAction *TestModelA;
	TMemo *Log1;
	TPanel *Panel3;
	TEdit *mModelFileName;
	TAction *updateFileName;
	TRadioGroup *RGLogLevel;
	TTabSheet *TabSheet3;
	TMemo *Memo3;
	TMemo *Memo4;
	TSplitter *Splitter2;
	TGroupBox *GroupBox2;
	TCheckBox *compileOnlyCB;
	TCheckBox *CheckBox2;
	TCheckBox *CheckBox3;
	mtkIniFileC *mIniFile;
	void __fastcall LoadModelAExecute(TObject *Sender);
	void __fastcall fsfTreeView1DblClick(TObject *Sender);
	void __fastcall TestModelAExecute(TObject *Sender);
	void __fastcall FormKeyDown(TObject *Sender, WORD &Key, TShiftState Shift);
	void __fastcall fsfTreeView1Click(TObject *Sender);

private:	// User declarations
	string 			mrrModelsRoot;
	string 			mOutputRoot;
    mtkExeFile		mRR;
	string 			GetSelectedFileName();

    mtkFormSaver	mFormSaver;

public:		// User declarations
	__fastcall 		TMainForm(TComponent* Owner);
	__fastcall 	   ~TMainForm();
};
//---------------------------------------------------------------------------
extern PACKAGE TMainForm *MainForm;
//---------------------------------------------------------------------------
#endif
