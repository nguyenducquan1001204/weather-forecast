Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "E:\MayAo\Crawdulieu"
WshShell.Run Chr(34) & "C:\Users\hoang\AppData\Local\Programs\Python\Python311\python.exe" & Chr(34) & " " & Chr(34) & "E:\MayAo\Crawdulieu\sync_all.py" & Chr(34), 0, False
Set WshShell = Nothing
