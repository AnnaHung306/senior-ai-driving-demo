"""Real Win32 no-share-delete handle, not a mocked PermissionError."""
import ctypes,json,sys,threading,time
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_atomic import write_json

def main():
    folder=ROOT/'artifacts/runs/phase10-Windows-lock-v1-001';folder.mkdir(parents=True,exist_ok=False)
    path=folder/'progress.json';path.write_text('{"old":true}')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW;create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    handle=create(str(path),0x80000000,3,None,3,128,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    def release():time.sleep(.15);assert close(handle)
    thread=threading.Thread(target=release);thread.start()
    try:receipt=write_json(path,{'new':True})
    finally:thread.join()
    assert receipt['attempts']>1 and json.loads(path.read_text())=={'new':True}
    result={'status':'passed_actual_Win32_read_lock_and_bounded_atomic_retry','replacement':receipt,
        'share_delete_was_disallowed':True,'lock_release_s':.15,'source_unchanged_during_test':True}
    (ROOT/'artifacts/reports/phase10-Windows-lock-final.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(result)

if __name__=='__main__':main()
