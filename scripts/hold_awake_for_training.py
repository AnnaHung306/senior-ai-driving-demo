"""Temporary Windows idle-sleep request bound to one verified process instance.

No power-plan settings, display request, scheduler, or service. The request is
cleared when that process exits, the time budget expires, or this helper exits.
Explicit user sleep remains under Windows/user control.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path


def kernel():
    if os.name!='nt':raise OSError('Windows only')
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;api.GetProcessTimes.restype=wintypes.BOOL
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];api.WaitForSingleObject.restype=wintypes.DWORD
    api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
    api.SetThreadExecutionState.argtypes=[wintypes.DWORD];api.SetThreadExecutionState.restype=wintypes.DWORD
    return api


def process_creation(api,handle):
    created,exited,kernel_time,user_time=[wintypes.FILETIME() for _ in range(4)]
    if not api.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel_time),ctypes.byref(user_time)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (created.dwHighDateTime<<32)|created.dwLowDateTime


def creation_filetime(pid):
    api=kernel();handle=api.OpenProcess(0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:return process_creation(api,handle)
    finally:api.CloseHandle(handle)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def hold(pid,expected_start,max_hours,output):
    if not 0<max_hours<=6:raise ValueError('Bound duration to 0..6 hours')
    api=kernel();handle=api.OpenProcess(0x1000|0x100000,False,pid)  # query limited information + synchronize
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    active=False
    receipt={'helper_pid':os.getpid(),'target_pid':pid,'target_creation_filetime':str(expected_start),
             'started_at_utc':datetime.now(timezone.utc).isoformat(),'max_hours':max_hours,
             'scope':'Thread-owned ES_CONTINUOUS|ES_SYSTEM_REQUIRED only while target process instance runs; no power-plan or display changes',
             'API_documentation':'https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate'}
    try:
        if process_creation(api,handle)!=expected_start:raise ValueError('PID instance changed; no idle-sleep request set')
        if api.WaitForSingleObject(handle,0)==0:raise ValueError('Target already exited; no request set')
        previous=api.SetThreadExecutionState(0x80000001)
        if not previous:raise ctypes.WinError(ctypes.get_last_error())
        active=True;receipt.update(status='active',request_cleared=False);write(output,receipt)
        print(json.dumps({'status':'active','target_pid':pid,'helper_pid':os.getpid()}),flush=True)
        result=api.WaitForSingleObject(handle,int(max_hours*3600*1000))
        if result==0:receipt['reason']='target_process_exited'
        elif result==258:receipt['reason']='bounded_duration_expired'
        else:raise ctypes.WinError(ctypes.get_last_error())
    except Exception as exc:
        receipt.update(status='failed',error=type(exc).__name__,message=str(exc));raise
    finally:
        if active:
            cleared=bool(api.SetThreadExecutionState(0x80000000))
            receipt.update(request_cleared=cleared,ended_at_utc=datetime.now(timezone.utc).isoformat())
            if receipt.get('status')!='failed':receipt['status']='released' if cleared else 'clear_failed'
        api.CloseHandle(handle);write(output,receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--expected-start-filetime',type=int,required=True);p.add_argument('--max-hours',type=float,default=6.)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    hold(args.pid,args.expected_start_filetime,args.max_hours,args.output)
