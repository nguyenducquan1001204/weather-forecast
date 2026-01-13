Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "E:\MayAo\Crawdulieu"
WshShell.Run Chr(34) & "C:\Users\hoang\AppData\Local\Programs\Python\Python311\python.exe" & Chr(34) & " " & Chr(34) & "E:\MayAo\Crawdulieu\auto_generate_forecast.py" & Chr(34), 0, False
Set WshShell = Nothing
