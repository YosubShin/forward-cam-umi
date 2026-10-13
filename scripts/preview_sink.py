"""mpv/ffplay raw-video window shared by the command-line camera tools."""
import subprocess


def open_sink(title, w, h):
    """mpv first (XVideo, survives the broken-GLX state after NVIDIA driver
    updates), ffplay second — same choice as uvc_record's previews."""
    cmds = [
        ["mpv", "--no-config", "--really-quiet", f"--title={title}",
         "--profile=low-latency", "--no-cache", "--untimed",
         "--demuxer-readahead-secs=0", "--demuxer=rawvideo",
         "--demuxer-rawvideo-mp-format=bgr24", f"--demuxer-rawvideo-w={w}",
         f"--demuxer-rawvideo-h={h}", "--demuxer-rawvideo-fps=60", "-"],
        ["ffplay", "-loglevel", "error", "-window_title", title,
         "-fflags", "nobuffer", "-flags", "low_delay", "-f", "rawvideo",
         "-pixel_format", "bgr24", "-video_size", f"{w}x{h}",
         "-framerate", "60", "-i", "pipe:0"],
    ]
    for cmd in cmds:
        try:
            return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError:
            continue
    print("(no mpv/ffplay — numbers only, no window)", flush=True)
    return None

def close_sink(sink):
    """Release the external player when a preview finishes or capture fails."""
    if sink is None:
        return
    if sink.poll() is None:
        sink.terminate()
    try:
        sink.wait(timeout=2)
    except subprocess.TimeoutExpired:
        sink.kill()
        sink.wait(timeout=2)
    if sink.stdin is not None:
        try:
            sink.stdin.close()
        except (BrokenPipeError, OSError):
            pass
