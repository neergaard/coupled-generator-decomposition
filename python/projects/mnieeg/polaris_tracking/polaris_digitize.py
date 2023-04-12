import argparse
import os
from pathlib import Path
import sys
import time
import winsound

import keyboard
import numpy as np
from sksurgerynditracker.nditracker import NDITracker

# Configuration

# L* = left from back to front
# R* = right from back to front
LABELS_LANDMARKS = (
    "lpa",
    "rpa",
    "nasion",
    "L1",
    "R1",
    "L2",
    "R2",
    "L3",
    "R3",
    "L4",
    "R4",
    "L5",
    "R5",
)
LABELS_ELECTRODES = np.arange(1, 64).astype(str)

NDI_SETTINGS = {
    "tracker type": "polaris",
    "verbose": True,
    "serial port": -1,  #'COM5',
    "ports to probe": -1,
    "romfiles": [
        "polaris/LOCALITE-Pointer.rom",
        # "polaris/LOCALITE-TMS-Coil-1.rom"
        "polaris/LOCALITE-TMS-Reference.rom"
        # "polaris/LOCALITE-FUSION-Reference.rom"
    ],
}

KEY_ACQ = "a"
KEY_DEL = "d"
KEY_END = "q"

msg_acq = "Press to acquire {:10} ({} of {} acquired)"
msg_del = "Cancelled        {:10}"
msg_acq_free = "Acquired number of points {:4d}"


def polaris_digitize(args):
    args = parse_args(args)
    mode = args.mode
    session = args.session

    # outpath = Path(__file__).parent / "session_{}".format(session)
    outpath = os.getcwd() / "session_{}".format(session)

    if mode == "landmarks":
        labels = LABELS_LANDMARKS
    elif mode == "electrodes":
        labels = LABELS_ELECTRODES
    else:
        labels = None

    # NDI_SETTINGS = {"tracker type": "dummy"}
    tracker = NDITracker(NDI_SETTINGS)
    print("Device description")
    print(tracker.get_tool_descriptions())
    print()
    tracker.start_tracking()

    if mode in ("landmarks", "electrodes"):
        points = acquire_points(tracker, labels)
    elif mode == "free":
        points = acquire_points_continuous(tracker)
    to_npy(outpath, mode, points, labels)

    tracker.stop_tracking()
    tracker.close()


def acquire_points(tracker, labels):
    """Acquire a list of points corresponding to labels."""

    print(
        "Acquiring INDIVIDUAL POINTS. Usage:\n"
        "[{:s}]   Acquire point\n"
        "[{:s}]   Discard last point\n"
        "[{:s}]   End acquisition and save\n".format(KEY_ACQ, KEY_DEL, KEY_END)
    )

    points = []

    print(msg_acq.format(labels[0], 0, len(labels)))

    _ = keyboard.add_hotkey(KEY_ACQ, acquire_point, args=[tracker, points, labels])
    _ = keyboard.add_hotkey(KEY_DEL, cancel_point, args=[points, labels])

    keyboard.wait(KEY_END)

    return points


def acquire_point(tracker, points, labels):
    """Acquire a point."""

    n = len(labels)
    # tracking.shape = (2, 4, 4) where
    # tracking[0] is the pointer
    # tracking[1] is the reference
    _, _, _, tracking, _ = tracker.get_frame()

    ### DUMMY TEST
    # tracking = np.random.random((2, 4, 4))
    # if np.random.random() < 0.3:
    #     tracking[0, 0, 0] = np.nan
    ###

    visible = ~np.isnan(tracking).any((1, 2))
    if visible.all():
        points.append(tracking)
        m = len(points)
        winsound.Beep(2000, 100)
        if m == n:
            print(f"{m} of {n} points acquired\n")
            print(f"Press [{KEY_END}] to save and exit")
            keyboard.remove_hotkey(KEY_ACQ)
            keyboard.remove_hotkey(KEY_DEL)
        else:
            print(msg_acq.format(labels[m], m, n))
    else:
        winsound.Beep(500, 100)
        if not any(visible):
            print("  Pointer and reference not visible!")
        elif not visible[0]:
            print("  Pointer not visible!")
        elif not visible[1]:
            print("  Reference not visible!")


def cancel_point(points, labels):
    """Cancel last digitized point."""

    n = len(labels)
    try:
        points.pop()
        m = len(points)
        print(msg_del.format(labels[m]))
        print(msg_acq.format(labels[m], m, n))
    except IndexError:
        pass


def acquire_points_continuous(tracker):
    """Acquire a free set of points."""

    print(
        "Acquiring FREE POINTS. Usage:\n"
        "[{:s}]   Acquire points (hold to acquire continuously)\n"
        "[{:s}]   End acquisition and save\n".format(KEY_ACQ, KEY_END)
    )
    points = []

    _ = keyboard.add_hotkey(KEY_ACQ, sample_points, args=[tracker, points])

    keyboard.wait(KEY_END)

    return points


def sample_points(tracker, points):
    """Sample points freely while holding down KEY_ACQ."""
    sfreq = 5
    t_target = 1 / sfreq

    while keyboard.is_pressed(KEY_ACQ):
        t = time.time()
        _, _, _, tracking, _ = tracker.get_frame()

        ###
        tracking = np.random.random((2, 4, 4))
        if np.random.random() < 0.3:
            tracking[0, 0, 0] = np.nan
        ###

        visible = ~np.isnan(tracking).any((1, 2))
        if visible.all():
            points.append(tracking)
            winsound.Beep(2000, 50)
        dt = t_target - (time.time() - t)
        time.sleep(dt)

    print(msg_acq_free.format(len(points)))


def to_npy(outpath, mode, points, labels=None):
    if not outpath.exists():
        outpath.mkdir()

    # data.shape == (n, 2, 4, 4) where n is the number of acquired points
    data = np.array(points)

    filename = outpath / f"{mode}_data.npy"
    np.save(filename, data)
    print(f"Wrote {filename}")

    if labels is not None:
        filename = outpath / f"{mode}_labels.npy"
        np.save(filename, labels)
        print(f"Wrote {filename}")


def parse_args(argv):

    program = dict(
        prog="polaris_digitize",
        description="Digitize positions using an NDI Polaris tracker.",
    )
    arg_mode = dict(help="Mode to run.", choices=("landmarks", "electrodes", "free"))
    arg_session = dict(help="Name of session. Results will be saved in ./session.")

    parser = argparse.ArgumentParser(**program)
    parser.add_argument("mode", **arg_mode)
    parser.add_argument("session", **arg_session)
    return parser.parse_args(argv)


if __name__ == "__main__":
    polaris_digitize(sys.argv[1:])
