"""Opt this process out of Windows 11 power throttling (EcoQoS).

Processes Windows considers "background" (e.g. launched from an IDE terminal) get parked on
efficiency cores at low clocks. Batch-1 decoding is bound by CPU-side kernel launches, so this
alone took launch latency from 50-90 us to ~16 us and made every timing ~4-10x slower.
Affects only the current process; no admin rights needed. No-op off Windows.
"""
import sys


def disable_power_throttling() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class ProcessPowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # ProcessPowerThrottling = 4; control EXECUTION_SPEED (1) with state 0 = never throttle
    state = ProcessPowerThrottlingState(1, 1, 0)
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))
