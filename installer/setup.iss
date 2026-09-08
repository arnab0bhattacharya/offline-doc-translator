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
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=Output
OutputBaseFilename=OfflineTranslatorSetup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
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
; Main application files from PyInstaller standalone distribution (bundled Python in _internal/)
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Standalone CLI helper script (optional headless use)
Source: "install_argos_packages.py"; DestDir: "{app}\installer"; Flags: ignoreversion

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

// ─────────────────────────────────────────────────────────
//  CUSTOM WIZARD PAGES
// ─────────────────────────────────────────────────────────

procedure InitializeWizard;
begin
  // Gemma 4 Model Selection Page
  GemmaModelPage := CreateInputOptionPage(
    wpSelectTasks,
    'Gemma 4 Model Selection',
    'Choose which Gemma 4 model to pull via Ollama.',
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
//  SHA-256 HASH VERIFICATION HELPER
// ─────────────────────────────────────────────────────────

function VerifyFileSHA256(const FilePath, ExpectedHash: String): Boolean;
var
  TmpFile: String;
  ResultCode: Integer;
  Lines: TArrayOfString;
  ActualHash: String;
begin
  Result := False;
  if not FileExists(FilePath) then Exit;
  TmpFile := ExpandConstant('{tmp}\hash_result.txt');
  if Exec('powershell.exe',
    '-NoProfile -Command "(Get-FileHash -Path ''' + FilePath + ''' -Algorithm SHA256).Hash.ToLower().Trim() | Out-File -Encoding ascii ''' + TmpFile + '''"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
  begin
    if LoadStringsFromFile(TmpFile, Lines) and (GetArrayLength(Lines) > 0) then
    begin
      ActualHash := Trim(Lowercase(Lines[0]));
      Result := (CompareText(ActualHash, Lowercase(Trim(ExpectedHash))) = 0);
    end;
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
//  POST-INSTALL ACTIONS
// ─────────────────────────────────────────────────────────

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  SelectedModel: String;
begin
  if CurStep = ssPostInstall then
  begin
    // ── Check Ollama for Pure LLM Mode ──
    if not IsOllamaInstalled then
    begin
      if MsgBox('Ollama was not detected on this system.' + #13#10 + #13#10 +
                'Ollama is required for Pure LLM mode (Fast NMT mode works fully offline without Ollama).' + #13#10 + #13#10 +
                'Would you like to open the official Ollama download page (https://ollama.com) now?',
                mbConfirmation, MB_YESNO) = IDYES then
      begin
        ShellExec('open', 'https://ollama.com/download', '', '', SW_SHOWNORMAL, ewNoWait, ResultCode);
      end;
    end
    else
    begin
      // ── Pull selected Gemma model if Ollama is available ──
      if GemmaModelPage.SelectedValueIndex < 3 then  // Not "Skip"
      begin
        case GemmaModelPage.SelectedValueIndex of
          0: SelectedModel := 'gemma4:e2b-it-qat';
          1: SelectedModel := 'gemma4:12b-it-qat';
          2: SelectedModel := 'gemma4:27b-it-qat';
        end;

        WizardForm.StatusLabel.Caption := 'Pulling ' + SelectedModel + ' via Ollama (this may take a few minutes)...';
        WizardForm.StatusLabel.Update;

        Exec('cmd.exe',
             '/C ollama pull ' + SelectedModel,
             '', SW_SHOW, ewWaitUntilTerminated, ResultCode);

        if ResultCode <> 0 then
          MsgBox('Model download may have encountered an issue. You can pull the model anytime using:' + #13#10 +
                 'ollama pull ' + SelectedModel + #13#10#13#10 +
                 'or from within the application System Settings.', mbInformation, MB_OK);
      end;
    end;

    WizardForm.StatusLabel.Caption := 'Installation complete!';
    WizardForm.StatusLabel.Update;
  end;
end;
