object MainF: TMainF
  Left = 0
  Top = 0
  Caption = 'RoadRunner PluginTester'
  ClientHeight = 587
  ClientWidth = 946
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
    Left = 0
    Top = 339
    Width = 946
    Height = 3
    Cursor = crVSplit
    Align = alBottom
    ExplicitLeft = 402
    ExplicitTop = 0
    ExplicitWidth = 256
  end
  object Panel1: TPanel
    Left = 0
    Top = 342
    Width = 946
    Height = 245
    Align = alBottom
    Caption = 'Panel1'
    TabOrder = 0
    ExplicitTop = 184
    ExplicitWidth = 832
    object infoMemo: TMemo
      Left = 1
      Top = 1
      Width = 944
      Height = 243
      Align = alClient
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'Courier New'
      Font.Style = []
      ParentFont = False
      PopupMenu = PopupMenu1
      ScrollBars = ssBoth
      TabOrder = 0
      WordWrap = False
      ExplicitWidth = 830
    end
  end
  object Panel2: TPanel
    Left = 0
    Top = 0
    Width = 337
    Height = 339
    Align = alLeft
    TabOrder = 1
    ExplicitHeight = 181
    object Panel3: TPanel
      Left = 1
      Top = 1
      Width = 176
      Height = 337
      Align = alLeft
      TabOrder = 0
      ExplicitHeight = 179
      object GroupBox1: TGroupBox
        Left = 1
        Top = 1
        Width = 174
        Height = 81
        Align = alTop
        Caption = 'RoadRunner C-API'
        TabOrder = 0
        object Label1: TLabel
          Left = 16
          Top = 24
          Width = 39
          Height = 13
          Caption = 'Version:'
        end
        object Label2: TLabel
          Left = 16
          Top = 43
          Width = 52
          Height = 13
          Caption = 'Build Date:'
        end
        object apiVersionLBL: TLabel
          Left = 112
          Top = 24
          Width = 65
          Height = 13
          Caption = 'apiVersionLBL'
        end
        object buildDateLbl: TLabel
          Left = 112
          Top = 43
          Width = 58
          Height = 13
          Caption = 'buildDateLbl'
        end
        object Label3: TLabel
          Left = 16
          Top = 62
          Width = 51
          Height = 13
          Caption = 'Build Time:'
        end
        object buildTimeLbl: TLabel
          Left = 112
          Top = 62
          Width = 57
          Height = 13
          Caption = 'buildTimeLbl'
        end
      end
      object Button4: TButton
        Left = 17
        Top = 96
        Width = 75
        Height = 25
        Action = getLastErrorA
        TabOrder = 1
      end
    end
    object Panel4: TPanel
      Left = 177
      Top = 1
      Width = 159
      Height = 337
      Align = alClient
      TabOrder = 1
      ExplicitHeight = 179
      object GroupBox2: TGroupBox
        Left = 1
        Top = 1
        Width = 157
        Height = 56
        Align = alTop
        Caption = 'Plugins'
        TabOrder = 0
        object Button1: TButton
          Left = 6
          Top = 23
          Width = 75
          Height = 25
          Action = loadPluginsA
          TabOrder = 0
        end
      end
      object pluginList: TListBox
        Left = 1
        Top = 57
        Width = 157
        Height = 279
        Align = alClient
        ItemHeight = 13
        TabOrder = 1
        OnClick = pluginListClick
      end
    end
  end
  object GroupBox3: TGroupBox
    Left = 337
    Top = 0
    Width = 185
    Height = 339
    Align = alLeft
    Caption = 'Plugin Actions'
    TabOrder = 2
    ExplicitHeight = 181
    object Button2: TButton
      Left = 16
      Top = 21
      Width = 75
      Height = 25
      Action = getPluginInfoA
      TabOrder = 0
    end
    object GroupBox4: TGroupBox
      Left = 2
      Top = 192
      Width = 181
      Height = 145
      Align = alBottom
      Caption = 'Parameters'
      TabOrder = 1
      object pluginParasCB: TComboBox
        Left = 4
        Top = 17
        Width = 174
        Height = 21
        AutoDropDown = True
        Style = csDropDownList
        TabOrder = 0
        OnChange = pluginCBChange
      end
      object paraEdit: TEdit
        Left = 4
        Top = 56
        Width = 121
        Height = 21
        TabOrder = 1
        Text = 'paraEdit'
      end
      object SetParaBtn: TButton
        Left = 131
        Top = 54
        Width = 47
        Height = 25
        Caption = 'set'
        TabOrder = 2
        OnClick = SetParaBtnClick
      end
    end
    object Button3: TButton
      Left = 97
      Top = 21
      Width = 75
      Height = 25
      Action = executePluginA
      TabOrder = 2
    end
    object GroupBox5: TGroupBox
      Left = 2
      Top = 128
      Width = 181
      Height = 64
      Align = alBottom
      Caption = 'Capability'
      TabOrder = 3
      ExplicitLeft = 4
      ExplicitTop = 97
      ExplicitWidth = 185
      object pluginCapsCB: TComboBox
        Left = 4
        Top = 25
        Width = 174
        Height = 21
        AutoDropDown = True
        Style = csDropDownList
        TabOrder = 0
        OnChange = pluginCBChange
      end
    end
  end
  object startupTimer: TTimer
    Enabled = False
    Interval = 100
    OnTimer = startupTimerTimer
    Left = 728
    Top = 80
  end
  object ActionList1: TActionList
    Left = 728
    Top = 16
    object loadPluginsA: TAction
      Caption = 'Load'
      OnExecute = loadPluginsAExecute
    end
    object unloadPlugins: TAction
      Caption = 'Unload'
      OnExecute = unloadPluginsExecute
    end
    object clearMemo: TAction
      Caption = 'Clear'
      OnExecute = clearMemoExecute
    end
    object getPluginInfoA: TAction
      Category = 'PluginActions'
      Caption = 'Get Info'
      OnExecute = getPluginInfoAExecute
    end
    object executePluginA: TAction
      Category = 'PluginActions'
      Caption = 'Execute'
      OnExecute = executePluginAExecute
    end
    object getLastErrorA: TAction
      Category = 'C_API'
      Caption = 'Last Error'
      OnExecute = getLastErrorAExecute
    end
  end
  object ApplicationEvents1: TApplicationEvents
    OnException = ApplicationEvents1Exception
    Left = 560
    Top = 16
  end
  object PopupMenu1: TPopupMenu
    Left = 192
    Top = 304
    object Clear1: TMenuItem
      Action = clearMemo
    end
  end
end
