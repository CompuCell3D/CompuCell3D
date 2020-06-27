# the call has to be showTweditWindowInForeground has to be done from the twedit
# (foregraound process) and not from EditorWindow (not a foreground process)

# because otherwise windows will not allow this call to succees and will return error code 

# PERHAPS A BETER SOLUTION IS TO STORE HANDLE TO TWEDIT ONCE
# IT IS STARTED INSTEAD OF SEARCHING EACH TIME WE OPEN NEW DOCUMENT


def win_enum_callback_twedit(hwnd, results):
    import win32gui
    import re

    expr_checker = re.compile(".*Twedit\+\+$")

    if expr_checker.match(win32gui.GetWindowText(hwnd)):
        # dbgMsg("GOT TWEDIT ++ ",win32gui.GetWindowText(hwnd))

        results.append(hwnd)


def showTweditWindowInForeground():
    # documentaiton of win32con window flags and win32gui showWindow functions

    # style=win32con.SW_SHOWNORMAL : int

    # Specifies how the window is to be shown. It must be one of win32con.SW_HIDE,
    # win32con.SW_MINIMIZE, win32con.SW_RESTORE, win32con.SW_SHOW, win32con.SW_SHOWMAXIMIZED
    # win32con.SW_SHOWMINIMIZED, win32con.SW_SHOWMINNOACTIVE, win32con.SW_SHOWNA, win32con.SW_SHOWNOACTIVATE,
    # or win32con.SW_SHOWNORMAL

    import win32gui
    import win32con

    handle = []

    win32gui.EnumWindows(win_enum_callback_twedit, handle)

    if len(handle):
        win32gui.SetActiveWindow(handle[0])

        win32gui.BringWindowToTop(handle[0])

        win32gui.SetFocus(handle[0])

        win32gui.SetForegroundWindow(handle[0])

        win32gui.ShowWindow(handle[0], win32con.SW_RESTORE)


def showTweditWindowInForeground1():
    # documentaiton of win32con window flags and win32gui showWindow functions

    # style=win32con.SW_SHOWNORMAL : int

    # Specifies how the window is to be shown. It must be one of win32con.SW_HIDE, win32con.SW_MINIMIZE,
    # win32con.SW_RESTORE, win32con.SW_SHOW, win32con.SW_SHOWMAXIMIZED win32con.SW_SHOWMINIMIZED, win32con.
    # SW_SHOWMINNOACTIVE, win32con.SW_SHOWNA, win32con.SW_SHOWNOACTIVATE, or win32con.SW_SHOWNORMAL

    import win32gui
    import win32con

    handle = []

    win32gui.EnumWindows(win_enum_callback_twedit, handle)

    if len(handle):
        # win32gui.SetForegroundWindow(handle[0])

        win32gui.SetActiveWindow(handle[0])

        win32gui.BringWindowToTop(handle[0])

        win32gui.SetFocus(handle[0])

        win32gui.ShowWindow(handle[0], win32con.SW_RESTORE)


def showTweditWindowWithPIDInForeground(_pid):
    # documentaiton of win32con window flags and win32gui showWindow functions

    # style=win32con.SW_SHOWNORMAL : int

    # Specifies how the window is to be shown. It must be one of win32con.SW_HIDE, win32con.SW_MINIMIZE,
    # win32con.SW_RESTORE, win32con.SW_SHOW, win32con.SW_SHOWMAXIMIZED win32con.SW_SHOWMINIMIZED,
    # win32con.SW_SHOWMINNOACTIVE, win32con.SW_SHOWNA, win32con.SW_SHOWNOACTIVATE, or win32con.SW_SHOWNORMAL

    import win32gui
    import win32con
    import win32process

    handle = []

    win32gui.EnumWindows(win_enum_callback_twedit, handle)

    # try:

    # dbgMsg("this is twedit window handle,",handle[0])

    # except IndexError,e:

    # print "problem with getting window handle"

    print("HANDLE ELELMENTS:", len(handle))

    # pid=win32process.GetCurrentProcessId()

    for hwnd in handle:

        t, p = win32process.GetWindowThreadProcessId(hwnd)

        print("t=", t, " p=", p)

        print("pid=", _pid)

        # pid = win32process.GetProcessId()

        # print "pid=",pid

        if _pid == p:
            win32gui.SetActiveWindow(hwnd)

            win32gui.BringWindowToTop(hwnd)

            # this does not work on windows Vista dna windows 7

            # win32gui.SetFocus(hwnd) 

            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            win32gui.SetForegroundWindow(hwnd)

    # if len(handle):

    # win32gui.SetActiveWindow(handle[0])

    # win32gui.BringWindowToTop(handle[0])

    # win32gui.SetFocus(handle[0])

    # win32gui.SetForegroundWindow(handle[0])

    # win32gui.ShowWindow(handle[0],win32con.SW_RESTORE)
