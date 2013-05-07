program rr_api_tester;

uses
///  madExcept,
//  madLinkDisAsm,
//  madListHardware,
//  madListProcesses,
//  madListModules,
  Forms,
  MainUnit in 'MainUnit.pas' {Form1};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
