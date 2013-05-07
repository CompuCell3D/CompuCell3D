object MainForm: TMainForm
  Left = 0
  Top = 0
  Caption = 'MainForm'
  ClientHeight = 470
  ClientWidth = 781
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  KeyPreview = True
  OldCreateOrder = False
  OnKeyDown = FormKeyDown
  PixelsPerInch = 96
  TextHeight = 13
  object Splitter1: TSplitter
    Left = 169
    Top = 0
    Height = 470
    ExplicitLeft = 216
    ExplicitTop = 32
    ExplicitHeight = 657
  end
  object GroupBox1: TGroupBox
    Left = 296
    Top = 96
    Width = 193
    Height = 201
    Caption = 'Actions'
    TabOrder = 0
  end
  object Memo1: TMemo
    Left = 520
    Top = 112
    Width = 185
    Height = 89
    Lines.Strings = (
      'Memo1')
    TabOrder = 1
  end
  object Panel1: TPanel
    Left = 0
    Top = 0
    Width = 169
    Height = 470
    Align = alLeft
    Caption = 'Panel1'
    TabOrder = 2
    object fsf: TFileSelectionFrame
      Left = 1
      Top = 1
      Width = 167
      Height = 468
      Align = alClient
      TabOrder = 0
    end
  end
  object Panel2: TPanel
    Left = 172
    Top = 0
    Width = 609
    Height = 470
    Align = alClient
    Caption = 'Panel2'
    TabOrder = 3
  end
  object PageControl1: TPageControl
    Left = 172
    Top = 0
    Width = 609
    Height = 470
    ActivePage = TabSheet1
    Align = alClient
    TabOrder = 4
    object TabSheet1: TTabSheet
      Caption = 'TabSheet1'
      ExplicitLeft = 0
      ExplicitTop = 0
      ExplicitWidth = 0
      ExplicitHeight = 0
      object Log1: TMemo
        Left = 0
        Top = 297
        Width = 601
        Height = 145
        Align = alClient
        ReadOnly = True
        ScrollBars = ssBoth
        TabOrder = 0
      end
      object Panel3: TPanel
        Left = 0
        Top = 0
        Width = 601
        Height = 297
        Align = alTop
        TabOrder = 1
        DesignSize = (
          601
          297)
        object Button1: TButton
          Left = 472
          Top = 218
          Width = 129
          Height = 73
          Action = TestModelA
          TabOrder = 0
        end
        object mModelFileName: TEdit
          Left = 16
          Top = 32
          Width = 568
          Height = 21
          Anchors = [akLeft, akTop, akRight]
          ReadOnly = True
          TabOrder = 1
          Text = 'mModelFileName'
        end
        object GroupBox2: TGroupBox
          Left = 16
          Top = 86
          Width = 417
          Height = 205
          Caption = 'Program Options'
          TabOrder = 2
          object compileOnlyCB: TCheckBox
            Left = 32
            Top = 24
            Width = 97
            Height = 17
            Caption = 'Compile'
            TabOrder = 0
          end
          object CheckBox2: TCheckBox
            Left = 32
            Top = 47
            Width = 97
            Height = 17
            Caption = 'CheckBox2'
            TabOrder = 1
          end
          object CheckBox3: TCheckBox
            Left = 32
            Top = 70
            Width = 97
            Height = 17
            Caption = 'CheckBox3'
            TabOrder = 2
          end
          object RGLogLevel: TRadioGroup
            Left = 184
            Top = 24
            Width = 207
            Height = 169
            Caption = 'LogLevel'
            Columns = 3
            ItemIndex = 2
            Items.Strings = (
              'Errors'
              'Warnings'
              'Info'
              'Debug'
              'Debug1'
              'Debug2'
              'Debug3'
              'Debug4'
              'Debug5')
            TabOrder = 3
          end
        end
      end
    end
    object TabSheet2: TTabSheet
      Caption = 'XML'
      ImageIndex = 1
      ExplicitLeft = 0
      ExplicitTop = 0
      ExplicitWidth = 0
      ExplicitHeight = 0
      object Memo2: TMemo
        Left = 0
        Top = 0
        Width = 601
        Height = 442
        Align = alClient
        Lines.Strings = (
          'Memo2')
        ReadOnly = True
        ScrollBars = ssBoth
        TabOrder = 0
      end
    end
    object TabSheet3: TTabSheet
      Caption = 'TabSheet3'
      ImageIndex = 2
      ExplicitLeft = 0
      ExplicitTop = 0
      ExplicitWidth = 0
      ExplicitHeight = 0
      object Splitter2: TSplitter
        Left = 393
        Top = 0
        Height = 442
        ExplicitLeft = 440
        ExplicitTop = 216
        ExplicitHeight = 100
      end
      object Memo3: TMemo
        Left = 0
        Top = 0
        Width = 393
        Height = 442
        Align = alLeft
        Lines.Strings = (
          'Memo3')
        TabOrder = 0
      end
      object Memo4: TMemo
        Left = 396
        Top = 0
        Width = 205
        Height = 442
        Align = alClient
        Lines.Strings = (
          'Memo4')
        TabOrder = 1
      end
    end
  end
  object ActionList1: TActionList
    Left = 592
    Top = 64
    object LoadModelA: TAction
      Caption = 'Load'
      OnExecute = LoadModelAExecute
    end
    object TestModelA: TAction
      Caption = 'Test'
      OnExecute = TestModelAExecute
    end
    object updateFileName: TAction
      Caption = 'updateFileName'
    end
  end
  object mIniFile: mtkIniFileC
    IniFileName = 'SBMLTester.ini'
    RootFolder = '.'
    Left = 680
    Top = 136
  end
end
