; installer/setup.iss
; =====================
; Inno Setup script for Offline Document Translator
;
; Prerequisites: Run `python build.py` first to generate dist/OfflineTranslator/
;
; To compile:
;   1. Install Inno Setup from https://jrsoftware.org/isdl.php
;   2. Open this file in Inno Setup Compiler
;   3. Click Build > Compile
;   Output: installer/Output/OfflineTranslatorSetup.exe

#define MyAppName "Offline Document Translator"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Offline Translation Project"
#define MyAppExeName "OfflineTranslator.exe"
#define MyAppURL "https://github.com/offline-doc-translator"

; Path to PyInstaller output (relative to this .iss file's parent directory)
#define DistDir "..\dist\OfflineTranslator"

[Setup]
AppId={{B5F3C7A2-4D1E-4F8A-9C2B-7E6D5A8F1C3D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=Output
OutputBaseFilename=OfflineTranslatorSetup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
; Minimum Windows 10
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application files from PyInstaller dist
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Installer helper scripts
Source: "install_argos_packages.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "check_python.bat"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Post-install: offer to launch the app
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  GemmaModelPage: TInputOptionWizardPage;
  PythonInstalled: Boolean;

// ─────────────────────────────────────────────────────────
//  CUSTOM WIZARD PAGES
// ─────────────────────────────────────────────────────────

procedure InitializeWizard;
begin
  // Gemma 4 Model Selection Page
  GemmaModelPage := CreateInputOptionPage(
    wpSelectTasks,
    'Gemma 4 Model Selection',
    'Choose which Gemma 4 model to download via Ollama.',
    'Select the model variant that matches your GPU. You can also download models later from within the application.',
    True,  // Exclusive (radio buttons)
    False  // Not list box
  );

  GemmaModelPage.Add('gemma4:e2b-it-qat (~5GB) - Best for 8GB VRAM GPUs');
  GemmaModelPage.Add('gemma4:12b-it-qat (~8GB) - Best for 12GB+ VRAM GPUs');
  GemmaModelPage.Add('gemma4:27b-it-qat (~17GB) - Best for 24GB+ VRAM GPUs');
  GemmaModelPage.Add('Skip model download (download later from the app)');
  GemmaModelPage.SelectedValueIndex := 0;  // Default: e2b
end;

// ─────────────────────────────────────────────────────────
//  PYTHON DETECTION
// ─────────────────────────────────────────────────────────

function IsPythonInstalled: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/C python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function GetPythonVersion: String;
var
  TmpFile: String;
  ResultCode: Integer;
  Lines: TArrayOfString;
begin
  Result := '';
  TmpFile := ExpandConstant('{tmp}\pyver.txt');
  if Exec('cmd.exe', '/C python --version > "' + TmpFile + '" 2>&1', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if LoadStringsFromFile(TmpFile, Lines) and (GetArrayLength(Lines) > 0) then
      Result := Lines[0];
    DeleteFile(TmpFile);
  end;
end;

// ─────────────────────────────────────────────────────────
//  OLLAMA DETECTION
// ─────────────────────────────────────────────────────────

function IsOllamaInstalled: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/C ollama --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

// ─────────────────────────────────────────────────────────
//  FILE DOWNLOAD HELPER (Native curl / PowerShell)
// ─────────────────────────────────────────────────────────

function DownloadFile(const Url, TargetFile: String): Boolean;
var
  ResultCode: Integer;
begin
  // Try Windows built-in curl.exe first
  Result := Exec('cmd.exe', '/C curl.exe -L -s -S -o "' + TargetFile + '" "' + Url + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) and FileExists(TargetFile);
  // Fallback to PowerShell if curl is unavailable
  if not Result then
    Result := Exec('powershell.exe', '-NoProfile -Command "(New-Object Net.WebClient).DownloadFile(''' + Url + ''', ''' + TargetFile + ''')"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) and FileExists(TargetFile);
end;

// ─────────────────────────────────────────────────────────
//  POST-INSTALL ACTIONS
// ─────────────────────────────────────────────────────────

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  SelectedModel: String;
  StatusText: String;
begin
  if CurStep = ssPostInstall then
  begin
    // ── Step 1: Check/Install Python ──
    WizardForm.StatusLabel.Caption := 'Checking Python installation...';
    WizardForm.StatusLabel.Update;

    PythonInstalled := IsPythonInstalled;

    if not PythonInstalled then
    begin
      if MsgBox('Python 3.14 is required but not found.' + #13#10 + #13#10 +
                'Would you like to download and install Python 3.14 now?' + #13#10 +
                '(This requires an internet connection)',
                mbConfirmation, MB_YESNO) = IDYES then
      begin
        WizardForm.StatusLabel.Caption := 'Downloading Python 3.14 installer...';
        WizardForm.StatusLabel.Update;

        // Download Python installer
        if not DownloadFile(
          'https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe',
          ExpandConstant('{tmp}\python_installer.exe')) then
        begin
          MsgBox('Failed to download Python installer. Please install Python 3.14+ manually from python.org.',
                 mbError, MB_OK);
        end
        else
        begin
          WizardForm.StatusLabel.Caption := 'Installing Python 3.14 (this may take a few minutes)...';
          WizardForm.StatusLabel.Update;

          Exec(ExpandConstant('{tmp}\python_installer.exe'),
               'InstallAllUsers=1 PrependPath=1 Include_pip=1 /quiet',
               '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

          if ResultCode = 0 then
            PythonInstalled := True
          else
            MsgBox('Python installation may have encountered issues. Please verify Python is installed.',
                   mbInformation, MB_OK);
        end;
      end;
    end
    else
    begin
      StatusText := GetPythonVersion;
      Log('Python found: ' + StatusText);
    end;

    // ── Step 2: Install pip dependencies ──
    if PythonInstalled then
    begin
      WizardForm.StatusLabel.Caption := 'Installing Python dependencies...';
      WizardForm.StatusLabel.Update;

      Exec('cmd.exe',
           '/C python -m pip install -r "' + ExpandConstant('{app}') + '\requirements.txt" --quiet',
           ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);

      if ResultCode <> 0 then
        Log('pip install returned code: ' + IntToStr(ResultCode));
    end;

    // ── Step 3: Install Argos language packages ──
    if PythonInstalled then
    begin
      WizardForm.StatusLabel.Caption := 'Installing Argos translation packages (~200MB)...';
      WizardForm.StatusLabel.Update;

      Exec('cmd.exe',
           '/C python "' + ExpandConstant('{app}') + '\installer\install_argos_packages.py"',
           ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);

      if ResultCode <> 0 then
        Log('Argos package install returned code: ' + IntToStr(ResultCode));
    end;

    // ── Step 4: Check/Install Ollama ──
    if not IsOllamaInstalled then
    begin
      if MsgBox('Ollama is not installed. Ollama is required for the Pure LLM translation mode.' + #13#10 + #13#10 +
                'Would you like to download and install Ollama now?' + #13#10 +
                '(Fast NMT mode works without Ollama)',
                mbConfirmation, MB_YESNO) = IDYES then
      begin
        WizardForm.StatusLabel.Caption := 'Downloading Ollama installer...';
        WizardForm.StatusLabel.Update;

        if not DownloadFile(
          'https://ollama.com/download/OllamaSetup.exe',
          ExpandConstant('{tmp}\OllamaSetup.exe')) then
        begin
          MsgBox('Failed to download Ollama. You can install it later from https://ollama.com',
                 mbInformation, MB_OK);
        end
        else
        begin
          WizardForm.StatusLabel.Caption := 'Installing Ollama...';
          WizardForm.StatusLabel.Update;

          Exec(ExpandConstant('{tmp}\OllamaSetup.exe'),
               '/VERYSILENT /NORESTART',
               '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
        end;
      end;
    end;

    // ── Step 5: Pull selected Gemma model ──
    if GemmaModelPage.SelectedValueIndex < 3 then  // Not "Skip"
    begin
      case GemmaModelPage.SelectedValueIndex of
        0: SelectedModel := 'gemma4:e2b-it-qat';
        1: SelectedModel := 'gemma4:12b-it-qat';
        2: SelectedModel := 'gemma4:27b-it-qat';
      end;

      if IsOllamaInstalled or (MsgBox('Ollama must be running to download the model. Continue?',
                                       mbConfirmation, MB_YESNO) = IDYES) then
      begin
        WizardForm.StatusLabel.Caption := 'Downloading ' + SelectedModel + ' (this may take a while)...';
        WizardForm.StatusLabel.Update;

        Exec('cmd.exe',
             '/C ollama pull ' + SelectedModel,
             '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

        if ResultCode <> 0 then
          MsgBox('Model download may have failed. You can pull the model later using:' + #13#10 +
                 'ollama pull ' + SelectedModel, mbInformation, MB_OK);
      end;
    end;

    WizardForm.StatusLabel.Caption := 'Installation complete!';
    WizardForm.StatusLabel.Update;
  end;
end;
