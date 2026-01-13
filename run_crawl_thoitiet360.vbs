Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "E:\MayAo\Crawdulieu"
WshShell.Run Chr(34) & "python" & Chr(34) & " " & Chr(34) & "E:\MayAo\Crawdulieu\crawl_thoitiet360.py" & Chr(34), 0, False
Set WshShell = Nothing
