' Double-click this file to start the app with no terminal window.
' It opens your browser automatically once the server is ready (a few seconds).
' To stop the app, close the browser tab and end the "python.exe" process
' in Task Manager (there is no window to close, since this runs silently).
'
' If the app doesn't open, double-click "Launch (visible).bat" instead to see
' what's going wrong, or check launch.log in this folder.

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")
strPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = strPath
objShell.Run """" & strPath & "\Launch (silent helper).bat""", 0, False
