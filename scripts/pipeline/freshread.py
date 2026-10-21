"""
freshread.py — Read a file's bytes straight from the device, bypassing the OS cache
==================================================================================
Some storage (e.g. USB disk enclosures) occasionally delivers damaged bytes for a
read without any OS-level error; netCDF/HDF5 then fails with "NetCDF: HDF error" (the
zlib checksum catches it). The damaged pages stay in the Windows file cache for a
while, so an ordinary re-read fails the same way. Reading with
FILE_FLAG_NO_BUFFERING fetches the bytes from the device again; the result can be
opened with ``netCDF4.Dataset(path, memory=bytes)``.
"""

from __future__ import annotations
import os

_CHUNK = 8 * 1024 * 1024          # multiple of any sector size


def read_unbuffered(path: str) -> bytes:
    if os.name != 'nt':
        flags = os.O_RDONLY | getattr(os, 'O_DIRECT', 0)
        try:
            fd = os.open(path, flags)
        except OSError:
            fd = os.open(path, os.O_RDONLY)
        try:
            with os.fdopen(fd, 'rb', buffering=0) as fh:
                return fh.read()
        except OSError:
            with open(path, 'rb') as fh:
                return fh.read()

    import ctypes
    from ctypes import wintypes as wt
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
    k32.CreateFileW.restype = wt.HANDLE
    k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    k32.ReadFile.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]

    GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN = 0x20000000, 0x08000000
    h = k32.CreateFileW(os.path.abspath(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    if h in (None, wt.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f'CreateFile failed: {path}')
    buf = k32.VirtualAlloc(None, _CHUNK, 0x3000, 0x04)      # MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE
    out = bytearray()
    n = wt.DWORD()
    try:
        while True:
            if not k32.ReadFile(h, buf, _CHUNK, ctypes.byref(n), None):
                raise OSError(ctypes.get_last_error(), f'ReadFile failed: {path}')
            if n.value == 0:
                break
            out += ctypes.string_at(buf, n.value)
            if n.value < _CHUNK:
                break
    finally:
        k32.VirtualFree(buf, 0, 0x8000)                       # MEM_RELEASE
        k32.CloseHandle(h)
    return bytes(out[:os.path.getsize(path)])
