// Read-only inspection of the bundled CARLA PDB through the installed DIA SDK.
#include <windows.h>
#include <dia2.h>
#include <iostream>
#include <string>

int wmain(int argc, wchar_t** argv) {
  if (argc != 2) return 2;
  CoInitialize(nullptr);
  IDiaDataSource* source = nullptr;
  HRESULT hr = CoCreateInstance(__uuidof(DiaSource), nullptr, CLSCTX_INPROC_SERVER,
                               __uuidof(IDiaDataSource), reinterpret_cast<void**>(&source));
  if (FAILED(hr)) {
    HMODULE module = LoadLibraryW(L"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\DIA SDK\\bin\\amd64\\msdia140.dll");
    if (!module) return 3;
    auto getClass = reinterpret_cast<HRESULT(STDAPICALLTYPE*)(REFCLSID, REFIID, LPVOID*)>(GetProcAddress(module, "DllGetClassObject"));
    IClassFactory* factory = nullptr;
    hr = getClass(__uuidof(DiaSource), IID_IClassFactory, reinterpret_cast<void**>(&factory));
    if (FAILED(hr)) return 4;
    hr = factory->CreateInstance(nullptr, __uuidof(IDiaDataSource), reinterpret_cast<void**>(&source));
    factory->Release();
  }
  if (FAILED(hr) || FAILED(source->loadDataFromPdb(argv[1]))) return 5;
  IDiaSession* session = nullptr;
  IDiaSymbol* global = nullptr;
  source->openSession(&session);
  session->get_globalScope(&global);
  IDiaEnumSymbols* enums = nullptr;
  global->findChildren(SymTagEnum, L"ENamedThreads::Type", nsCaseSensitive, &enums);
  IDiaSymbol* type = nullptr;
  ULONG count;
  while (enums && enums->Next(1, &type, &count) == S_OK) {
    IDiaEnumSymbols* members = nullptr;
    type->findChildren(SymTagData, nullptr, nsNone, &members);
    IDiaSymbol* member = nullptr;
    while (members && members->Next(1, &member, &count) == S_OK) {
      BSTR name = nullptr;
      VARIANT value; VariantInit(&value);
      member->get_name(&name); member->get_value(&value);
      std::wcout << L"ENUM " << name << L"=" << value.ulVal << L"\n";
      VariantClear(&value); SysFreeString(name); member->Release();
    }
    if (members) members->Release();
    type->Release();
  }
  if (enums) enums->Release();
  IDiaEnumSymbols* functions = nullptr;
  global->findChildren(SymTagFunction, L"FPixelReader::WritePixelsToBuffer", nsCaseSensitive, &functions);
  IDiaSymbol* function = nullptr;
  while (functions && functions->Next(1, &function, &count) == S_OK) {
    DWORD rva = 0; ULONGLONG length = 0;
    function->get_relativeVirtualAddress(&rva); function->get_length(&length);
    std::cout << "FUNCTION rva=0x" << std::hex << rva << " length=0x" << length << "\n";
    function->Release();
  }
  if (functions) functions->Release();
  global->Release(); session->Release(); source->Release(); CoUninitialize();
  return 0;
}
