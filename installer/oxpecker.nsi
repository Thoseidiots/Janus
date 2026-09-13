; Universal Oxpecker Windows Installer
; ====================================
; NSIS script to create professional installer
; Install location: C:\Program Files\Oxpecker

!include "MUI2.nsh"
!include "x64.nsh"

Name "Universal Oxpecker"
OutFile "dist\oxpecker-installer.exe"
InstallDir "$PROGRAMFILES\Oxpecker"
LicenseText "License"
LicenseData "LICENSE"

; MUI Settings
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "dist\oxpecker\*"
  
  ; Create Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\Oxpecker"
  CreateShortCut "$SMPROGRAMS\Oxpecker\Oxpecker CLI.lnk" "$INSTDIR\oxpecker.exe"
  CreateShortCut "$SMPROGRAMS\Oxpecker\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  ; Add to PATH
  EnVar::SetHKCU
  EnVar::AddValue "PATH" "$INSTDIR"
  
  ; Write uninstall registry
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Oxpecker" "DisplayName" "Universal Oxpecker"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Oxpecker" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  MessageBox MB_OK "Oxpecker installed successfully!$\n$\nOpen Command Prompt and type: oxpecker --help"
SectionEnd

Section "Uninstall"
  RMDir /r "$INSTDIR"
  RMDir /r "$SMPROGRAMS\Oxpecker"
  
  ; Remove from PATH
  EnVar::SetHKCU
  EnVar::DeleteValue "PATH" "$INSTDIR"
  
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Oxpecker"
SectionEnd
