from pyopenms_viewer import cli


def test_sanitize_pyinstaller_args_removes_known_flags():
    argv = [
        "pyopenms-viewer",
        "-B",
        "--multiprocessing-fork",
        "--multiprocessing-spawn",
        "-psn_0_123456",
        "tracker_fd=8",
        "pipe_handle=17",
        "-S",
        "-I",
        "--known",
    ]

    sanitized = cli._sanitize_pyinstaller_args(argv)

    assert sanitized == ["pyopenms-viewer", "--known"]


def test_sanitize_pyinstaller_args_noop_without_flags():
    argv = ["pyopenms-viewer", "--help"]

    assert cli._sanitize_pyinstaller_args(argv) == argv
