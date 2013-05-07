object Form1: TForm1
  Left = 0
  Top = 0
  Caption = 'RoadRunner C API Tester'
  ClientHeight = 526
  ClientWidth = 849
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  KeyPreview = True
  OldCreateOrder = False
  OnCreate = FormCreate
  OnKeyDown = FormKeyDown
  PixelsPerInch = 96
  TextHeight = 13
  object Panel1: TPanel
    Left = 0
    Top = 0
    Width = 225
    Height = 526
    Align = alLeft
    TabOrder = 0
    object GroupBox1: TGroupBox
      Left = 1
      Top = 1
      Width = 223
      Height = 105
      Align = alTop
      Caption = 'General'
      TabOrder = 0
      object Button1: TButton
        Left = 183
        Top = 14
        Width = 27
        Height = 25
        Action = FileOpen1
        TabOrder = 0
      end
      object Button2: TButton
        Left = 8
        Top = 43
        Width = 75
        Height = 25
        Action = LoadDLL
        TabOrder = 1
      end
      object Button3: TButton
        Left = 110
        Top = 43
        Width = 75
        Height = 25
        Action = UnloadDLL
        TabOrder = 2
      end
      object Button4: TButton
        Left = 8
        Top = 74
        Width = 101
        Height = 25
        Action = LoadFunctionsA
        TabOrder = 3
      end
      object FileNameE: TEdit
        Left = 8
        Top = 16
        Width = 169
        Height = 21
        TabOrder = 4
        Text = '<select rr dll>'
      end
    end
    object GroupBox2: TGroupBox
      Left = 1
      Top = 106
      Width = 223
      Height = 419
      Align = alClient
      Caption = 'API Functions'
      TabOrder = 1
      object FunctionList: TListBox
        Left = 2
        Top = 15
        Width = 219
        Height = 402
        Align = alClient
        ItemHeight = 13
        Sorted = True
        TabOrder = 0
        OnClick = FunctionListClick
      end
    end
  end
  object Panel2: TPanel
    Left = 225
    Top = 0
    Width = 624
    Height = 526
    Align = alClient
    TabOrder = 1
    object Memo1: TMemo
      Left = 1
      Top = 408
      Width = 622
      Height = 117
      Align = alBottom
      TabOrder = 0
    end
    object Button5: TButton
      Left = 24
      Top = 44
      Width = 75
      Height = 25
      Action = CharStarVoidA
      TabOrder = 1
    end
    object btnLoadSBML: TButton
      Left = 24
      Top = 13
      Width = 75
      Height = 25
      Caption = 'Load SBML'
      TabOrder = 2
      OnClick = btnLoadSBMLClick
    end
    object grid: TStringGrid
      Left = 222
      Top = 1
      Width = 401
      Height = 407
      Align = alRight
      FixedCols = 0
      FixedRows = 0
      TabOrder = 3
      ExplicitLeft = 176
      ExplicitTop = 8
      ExplicitHeight = 385
    end
  end
  object ActionList1: TActionList
    OnUpdate = ActionList1Update
    Left = 136
    Top = 136
    object LoadDLL: TAction
      Caption = 'Load'
      OnExecute = LoadDLLExecute
    end
    object UnloadDLL: TAction
      Caption = 'Unload'
      OnExecute = UnloadDLLExecute
    end
    object SelectDLLA: TAction
      Caption = 'SelectDLLA'
    end
    object FileOpen1: TFileOpen
      Caption = '...'
      Dialog.Filter = '*.dll'
      Hint = 'Open|Opens an existing file'
      ImageIndex = 7
      ShortCut = 16463
      BeforeExecute = FileOpen1BeforeExecute
      OnAccept = FileOpen1Accept
    end
    object LoadFunctionsA: TAction
      Caption = 'Load Functions'
      OnExecute = LoadFunctionsAExecute
    end
  end
  object APIFuncs: TActionList
    OnUpdate = APIFuncsUpdate
    Left = 256
    Top = 72
    object CharStarVoidA: TAction
      Caption = 'char*_void'
      Enabled = False
      OnExecute = CharStarVoidAExecute
    end
  end
  object OpenDialog1: TOpenDialog
    Left = 256
    Top = 112
  end
end
