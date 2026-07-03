#ifndef MyAppVersion
#define MyAppVersion "0.1.0-rc1"
#endif
#ifndef MyOutputSuffix
#define MyOutputSuffix ""
#endif

[Setup]
AppId={{7C6F0A3D-3077-4BE3-8C2C-8DF89D3D6C8A}
AppName=Koemo
AppVersion={#MyAppVersion}
AppVerName=Koemo {#MyAppVersion}
AppPublisher=Koemo
DefaultDirName={localappdata}\Programs\Koemo
DefaultGroupName=Koemo
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=Koemo-{#MyAppVersion}{#MyOutputSuffix}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
SetupIconFile=..\assets\koemo.ico
UninstallDisplayIcon={app}\Koemo.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "..\dist\Koemo\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\packaging\release-notes-ja.md"; DestDir: "{app}"; DestName: "RELEASE-NOTES-ja.md"; Flags: ignoreversion

[Icons]
Name: "{group}\Koemo"; Filename: "{app}\Koemo.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Koemo"; Filename: "{app}\Koemo.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "デスクトップにKoemoショートカットを作成"; GroupDescription: "追加ショートカット:"; Flags: unchecked

[Run]
Filename: "{app}\Koemo.exe"; Description: "Koemoを起動"; Flags: nowait postinstall skipifsilent
