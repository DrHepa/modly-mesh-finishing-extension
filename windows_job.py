# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows-only native ownership; imported lazily by production_runner.

The PROCESS owns the sole unnamed, non-inheritable job handle. A CPython
bootstrap cannot start Blender until its retained Popen process HANDLE has
been assigned to the job. No PID lookup, breakaway, or external dependency.
"""
import ctypes
from pathlib import Path
import subprocess


# Dedicated stdin is admission only, never the PROCESS protocol. EOF or any
# other byte refuses launch. Blender inherits the log, not the admission pipe.
_WINDOWS_GATE = """import ctypes,subprocess,sys
if sys.stdin.buffer.read(1)!=b'\\x01':
    raise SystemExit(125)
exit_process=ctypes.WinDLL('kernel32',use_last_error=True).ExitProcess
exit_process.argtypes=[ctypes.c_uint32]
exit_process.restype=None
code=subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL,close_fds=True,shell=False).wait()
exit_process(code)
"""


def windows_job_command(command):
    command=list(map(str,command))
    interpreter=_JobAPI().running_image()
    if not command or not Path(command[0]).is_absolute():
        raise ValueError('Windows job launcher requires the validated absolute native executable')
    return [str(interpreter),'-I','-S','-B','-c',_WINDOWS_GATE,*command]


# Fixed-width Windows types, including when these layouts are checked on a
# non-Windows test host. SIZE_T/ULONG_PTR retain the native pointer width.
class _BasicLimits(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
              ('LimitFlags',ctypes.c_uint32),('MinimumWorkingSetSize',ctypes.c_size_t),
              ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',ctypes.c_uint32),
              ('Affinity',ctypes.c_size_t),('PriorityClass',ctypes.c_uint32),('SchedulingClass',ctypes.c_uint32)]


class _IOCounters(ctypes.Structure):
    _fields_=[(name,ctypes.c_uint64) for name in
              ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
               'ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_=[('BasicLimitInformation',_BasicLimits),('IoInfo',_IOCounters),
              ('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),
              ('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]


class _JobAPI:
    def __init__(self):
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        for name,args,result in (
            ('GetModuleFileNameW',[ctypes.c_void_p,ctypes.POINTER(ctypes.c_wchar),ctypes.c_uint32],ctypes.c_uint32),
            ('CreateJobObjectW',[ctypes.c_void_p,ctypes.c_wchar_p],ctypes.c_void_p),
            ('SetInformationJobObject',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32],ctypes.c_int),
            ('AssignProcessToJobObject',[ctypes.c_void_p,ctypes.c_void_p],ctypes.c_int),
            ('TerminateJobObject',[ctypes.c_void_p,ctypes.c_uint32],ctypes.c_int),
            ('CloseHandle',[ctypes.c_void_p],ctypes.c_int)):
            function=getattr(self.kernel,name);function.argtypes=args;function.restype=result

    @staticmethod
    def checked(value):
        if not value:raise ctypes.WinError(ctypes.get_last_error())
        return value

    def running_image(self):
        # A Windows venv sys.executable may be a redirector that creates the
        # real interpreter before assignment. Launch the current loaded image
        # itself, not that redirector; -S also excludes startup site hooks.
        buffer=ctypes.create_unicode_buffer(32768)
        length=self.kernel.GetModuleFileNameW(None,buffer,len(buffer))
        if not 0<length<len(buffer):
            raise RuntimeError('Cannot determine an untruncated Windows Python image')
        image=Path(buffer.value)
        if not image.is_absolute() or not image.is_file():
            raise RuntimeError('Windows Python image must be an existing absolute executable')
        return image

    def create(self):
        # NULL security attributes => non-inheritable; NULL name => new unnamed
        # job. Only KILL_ON_JOB_CLOSE, never either BREAKAWAY flag.
        handle=self.checked(self.kernel.CreateJobObjectW(None,None))
        try:
            limits=_ExtendedLimits();limits.BasicLimitInformation.LimitFlags=0x00002000
            self.checked(self.kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)))
        except BaseException:
            self.close(handle)
            raise
        return handle

    def assign(self,handle,process_handle):
        self.checked(self.kernel.AssignProcessToJobObject(handle,process_handle))

    def terminate(self,handle):
        self.checked(self.kernel.TerminateJobObject(handle,1))

    def close(self,handle):
        self.checked(self.kernel.CloseHandle(handle))


class WindowsJobOwner:
    """Caller must close in finally, even when start fails or the shim exits.

    CPython Popen._handle is retained by Popen; it is never reopened by PID,
    detached, closed by this class, or duplicated into the shim. An unsupported
    Popen implementation / incompatible enclosing job fails before admission.
    """
    def __init__(self):
        self.child=None
        self._api=_JobAPI()
        self._handle=self._api.create()

    def start(self,launch_command,log):
        self.child=subprocess.Popen(launch_command,stdin=subprocess.PIPE,stdout=log,
                                    stderr=subprocess.STDOUT,close_fds=True,shell=False,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        retained=getattr(self.child,'_handle',None)
        if retained is None or getattr(retained,'closed',False) or int(retained)<=0:
            raise RuntimeError('Windows job requires a retained CPython process handle')
        self._api.assign(self._handle,int(retained))
        if self.child.stdin.write(b'\x01')!=1:
            raise OSError('Windows job admission byte was not written')
        self.child.stdin.flush()
        self.child.stdin.close()
        return self.child

    def close(self):
        failure=None
        # Before admission, closing this pipe makes the unassigned shim exit
        # without native work. After admission, the job owns all descendants.
        if self.child is not None and self.child.stdin is not None:
            try:self.child.stdin.close()
            except Exception as exc:failure=exc
        if self._handle is not None:
            try:self._api.terminate(self._handle)
            except Exception as exc:failure=failure or exc
            finally:
                try:self._api.close(self._handle)
                except Exception as exc:failure=failure or exc
                else:self._handle=None
        if self.child is not None:
            try:self.child.wait(timeout=10)
            except Exception as exc:failure=failure or exc
        if failure:raise failure
