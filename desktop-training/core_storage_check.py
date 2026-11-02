"""Installer preflight without Pillow or ML dependencies."""
import ctypes
import os
from pathlib import Path

def check(path=None):
    p=Path(path).absolute() if path is not None else Path(__file__).absolute().parent
    if os.name!='nt':
        if str(p).startswith(('/home/mount/','/workspace/mount/')):raise RuntimeError('Use local program storage, not mount aliases.')
        if any(a.is_symlink() for a in [p]+list(p.parents)):raise RuntimeError('Linked program folders are not supported.')
        mountinfo=Path('/proc/self/mountinfo')
        if mountinfo.exists():
            matches=[]
            for line in mountinfo.read_text().splitlines():
                left,right=line.split(' - ',1);mount=left.split()[4].replace('\\040',' ')
                if str(p)==mount or str(p).startswith(mount.rstrip('/')+'/'):matches.append((len(mount),right.split()[0]))
            if matches:
                fs=max(matches)[1]
                if fs in {'nfs','nfs4','cifs','smb3','9p'} or fs.startswith('fuse.'):raise RuntimeError('Network program storage is not allowed.')
        return
    if str(p).startswith('\\\\') or ctypes.windll.kernel32.GetDriveTypeW(p.anchor)!=3:
        raise RuntimeError('Use a local fixed disk. NAS and network drives are not allowed.')
    for a in [p]+list(p.parents):
        if a.is_symlink() or (a.exists() and getattr(a.lstat(),'st_file_attributes',0)&0x400):
            raise RuntimeError('Linked folders/junctions are not supported. Use a real local folder.')

if __name__=='__main__': check()
