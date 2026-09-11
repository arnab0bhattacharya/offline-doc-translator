"""
gui/dnd_helper.py
=================
Windows ctypes and cross-platform drag-and-drop integration for CustomTkinter/Tkinter.
Enables receiving dropped files without external C extensions or binary wheels.
"""

import os
import sys
import ctypes
from typing import Callable, List, Optional

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4


def is_point_in_widget(widget, screen_x: int, screen_y: int) -> bool:
    """Returns True if the given absolute screen coordinates fall inside the widget."""
    try:
        if not widget or not widget.winfo_exists() or not widget.winfo_ismapped():
            return False
        rx = widget.winfo_rootx()
        ry = widget.winfo_rooty()
        w = widget.winfo_width()
        h = widget.winfo_height()
        return (rx <= screen_x <= rx + w) and (ry <= screen_y <= ry + h)
    except Exception:
        return False


class WindowsDropHook:
    """Subclasses HWND window procedure to intercept WM_DROPFILES on Windows."""

    def __init__(self, root_widget, on_drop_callback: Callable[[List[str], int, int], None]):
        self.root_widget = root_widget
        self.on_drop_callback = on_drop_callback
        self.hwnd: Optional[int] = None
        self._orig_wndproc = None
        self._wndproc_cb = None
        self._is_hooked = False

    def hook(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            shell32 = ctypes.windll.shell32

            # Determine 64-bit vs 32-bit types
            if ctypes.sizeof(ctypes.c_void_p) == 8:
                WNDPROC_TYPE = ctypes.WINFUNCTYPE(
                    ctypes.c_int64, wintypes.HWND, wintypes.UINT, ctypes.c_uint64, ctypes.c_int64
                )
                SetWindowLongPtr = user32.SetWindowLongPtrW
                SetWindowLongPtr.restype = ctypes.c_void_p
                SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
                CallWindowProc = user32.CallWindowProcW
                CallWindowProc.restype = ctypes.c_int64
                CallWindowProc.argtypes = [
                    ctypes.c_void_p,
                    wintypes.HWND,
                    wintypes.UINT,
                    ctypes.c_uint64,
                    ctypes.c_int64,
                ]
            else:
                WNDPROC_TYPE = ctypes.WINFUNCTYPE(
                    ctypes.c_long, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
                )
                SetWindowLongPtr = user32.SetWindowLongW
                SetWindowLongPtr.restype = ctypes.c_void_p
                SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
                CallWindowProc = user32.CallWindowProcW
                CallWindowProc.restype = ctypes.c_long
                CallWindowProc.argtypes = [
                    ctypes.c_void_p,
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                ]

            self.root_widget.update_idletasks()
            wid = self.root_widget.winfo_id()
            self.hwnd = user32.GetAncestor(wid, 2) or wid

            shell32.DragAcceptFiles(self.hwnd, True)

            def _py_wndproc(hwnd, msg, wparam, lparam):
                if msg == WM_DROPFILES:
                    hdrop = wparam
                    pt = wintypes.POINT()
                    shell32.DragQueryPoint(hdrop, ctypes.byref(pt))
                    user32.ClientToScreen(hwnd, ctypes.byref(pt))
                    screen_x, screen_y = pt.x, pt.y

                    num_files = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
                    files: List[str] = []
                    for i in range(num_files):
                        length = shell32.DragQueryFileW(hdrop, i, None, 0)
                        buf = ctypes.create_unicode_buffer(length + 1)
                        shell32.DragQueryFileW(hdrop, i, buf, length + 1)
                        if buf.value:
                            files.append(buf.value)
                    shell32.DragFinish(hdrop)

                    if files and self.on_drop_callback:
                        # Schedule on Tk event loop safely
                        try:
                            self.root_widget.after(
                                0,
                                lambda f=files, sx=screen_x, sy=screen_y: self.on_drop_callback(f, sx, sy),
                            )
                        except Exception:
                            pass
                    return 0

                return CallWindowProc(self._orig_wndproc, hwnd, msg, wparam, lparam)

            self._wndproc_cb = WNDPROC_TYPE(_py_wndproc)
            self._orig_wndproc = SetWindowLongPtr(
                self.hwnd, GWLP_WNDPROC, ctypes.cast(self._wndproc_cb, ctypes.c_void_p)
            )
            self._is_hooked = True
            return True
        except Exception:
            return False

    def unhook(self):
        if not self._is_hooked or not self.hwnd or not self._orig_wndproc:
            return
        try:
            user32 = ctypes.windll.user32
            shell32 = ctypes.windll.shell32
            shell32.DragAcceptFiles(self.hwnd, False)
            if ctypes.sizeof(ctypes.c_void_p) == 8:
                user32.SetWindowLongPtrW(self.hwnd, GWLP_WNDPROC, self._orig_wndproc)
            else:
                user32.SetWindowLongW(self.hwnd, GWLP_WNDPROC, self._orig_wndproc)
        except Exception:
            pass
        finally:
            self._is_hooked = False
