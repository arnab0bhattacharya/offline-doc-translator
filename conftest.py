"""
conftest.py
===========
Pytest root configuration.
Ensures TCL/TK library environment variables are initialized for GUI tests on Windows.
"""

import os
import sys

if sys.platform == "win32":
    tcl_dir = os.path.join(sys.prefix, "tcl", "tcl8.6")
    tk_dir = os.path.join(sys.prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_dir) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_dir
    if os.path.isdir(tk_dir) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_dir
