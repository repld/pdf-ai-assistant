Option Explicit

Dim fileSystem, shell, projectFolder, pythonPath, environmentFile, command
Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)
pythonPath = projectFolder & "\.venv\Scripts\python.exe"
environmentFile = projectFolder & "\.env"

If Not fileSystem.FileExists(pythonPath) Then
    MsgBox "The virtual environment was not found. Run the setup commands first.", 48, "AI Document Assistant"
    WScript.Quit 1
End If

If Not fileSystem.FileExists(environmentFile) Then
    MsgBox "The .env file was not found. Create it and add your API key.", 48, "AI Document Assistant"
    WScript.Quit 1
End If

command = "powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File " & _
          Chr(34) & projectFolder & "\start-assistant.ps1" & Chr(34)
shell.Run command, 0, False
