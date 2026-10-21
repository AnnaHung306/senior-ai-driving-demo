"""Bounded Windows read-lock recovery for navigation JSON publication."""
import json,time,uuid
from pathlib import Path

def write_json(path,value,timeout_s=.8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    started=time.monotonic();attempts=0
    try:
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
        while True:
            attempts+=1
            try:temporary.replace(path);break
            except PermissionError:
                if time.monotonic()-started>=timeout_s:raise
                time.sleep(.02)
        return {'attempts':attempts,'elapsed_s':time.monotonic()-started}
    finally:
        if temporary.exists():temporary.unlink()
