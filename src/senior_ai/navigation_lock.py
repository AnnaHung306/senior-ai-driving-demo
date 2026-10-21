"""OS-released lease prevents two project processes owning the simulator tick."""
import os
from pathlib import Path

class TickLease:
    def __init__(self,path):self.path=Path(path);self.stream=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.stream=self.path.open('a+b')
        self.stream.seek(0,2)
        if self.stream.tell()==0:self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close();self.stream=None;raise RuntimeError('Another project process owns CARLA tick')
    def release(self):
        if self.stream is None:return
        self.stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
        self.stream.close();self.stream=None
