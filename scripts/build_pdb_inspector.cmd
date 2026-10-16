@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /EHsc /I"C:\Program Files\Microsoft Visual Studio\2022\Community\DIA SDK\include" /Fo"artifacts\reports\diagnostics\inspect_carla_pdb.obj" /Fe"artifacts\reports\diagnostics\inspect_carla_pdb.exe" scripts\inspect_carla_pdb.cpp /link /LIBPATH:"C:\Program Files\Microsoft Visual Studio\2022\Community\DIA SDK\lib\amd64" diaguids.lib ole32.lib oleaut32.lib
