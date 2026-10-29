"""Stills, figures, videos and GIFs of a replay bundle, drawn by the viewer itself.

Every file this script writes is the viewer's own drawing: it opens
``viewer/render.html`` (the same ``js/scene.js`` the interactive viewer runs) in
headless Chrome and prints it (PDF) or screenshots it (PNG frames), and ffmpeg
encodes the frames. There is no second renderer.

    # one figure: panels at chosen steps, with headings and the key (vector PDF)
    python3 -m scripts.replay_media figure --bundle b.replay.js --steps 97,146,226,275 \\
        --headings "First pursuits|One pursuer per intruder|First intruder destroyed|All four destroyed" \\
        --out fig.pdf
    # a still of one frame (PNG at any width, or the SVG itself)
    python3 -m scripts.replay_media still --bundle b.replay.js --frame 226 --out s.png --width-px 1600
    # a video (H.264 MP4) or a GIF, any speed and frame range
    python3 -m scripts.replay_media video --bundle b.replay.js --speed 4 --out v.mp4
    python3 -m scripts.replay_media gif --bundle b.replay.js --speed 8 --fps 10 --width-px 640 --out v.gif

Chrome is driven over its DevTools pipe (``--remote-debugging-pipe``), so only
the standard library is needed. Free: nothing here touches the network.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RENDER = os.path.join(ROOT, "viewer", "render.html")
CSS_PX_PER_MM = 96 / 25.4

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
)


def find_chrome():
    env = os.environ.get("CHROME")
    for c in ((env,) if env else ()) + CHROME_CANDIDATES:
        path = c if os.path.isabs(c) else shutil.which(c)
        if path and os.path.exists(path):
            return path
    raise SystemExit("Google Chrome or Chromium is needed (set CHROME=/path/to/chrome)")


class Chrome:
    """Headless Chrome over the DevTools pipe: JSON messages, NUL-terminated, on
    file descriptors 3 (commands in) and 4 (replies out)."""

    def __init__(self):
        cmd_r, self._cmd_w = os.pipe()
        self._out_r, out_w = os.pipe()
        self._profile = tempfile.mkdtemp(prefix="pev-chrome-")

        def child():
            os.dup2(cmd_r, 3)
            os.dup2(out_w, 4)

        self.proc = subprocess.Popen(
            [find_chrome(), "--headless=new", "--remote-debugging-pipe", "--disable-gpu",
             "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
             "--allow-file-access-from-files", "--force-color-profile=srgb",
             "--font-render-hinting=none", f"--user-data-dir={self._profile}", "about:blank"],
            preexec_fn=child, pass_fds=(3, 4), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.close(cmd_r)
        os.close(out_w)
        self._buf = b""
        self._id = 0
        target = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        self.session = self.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        self.call("Page.enable", session=True)

    def call(self, method, params=None, session=False, timeout=120):
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = self.session
        os.write(self._cmd_w, json.dumps(msg).encode() + b"\0")
        deadline = time.time() + timeout
        while True:
            while b"\0" not in self._buf:
                if time.time() > deadline:
                    raise TimeoutError(method)
                chunk = os.read(self._out_r, 1 << 20)
                if not chunk:
                    raise RuntimeError("Chrome closed the DevTools pipe")
                self._buf += chunk
            raw, self._buf = self._buf.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def evaluate(self, expr):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True},
                      session=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page error: {r['exceptionDetails']}")
        return r["result"].get("value")

    def open(self, url, timeout=60):
        self.call("Page.navigate", {"url": url}, session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = self.evaluate("document.body ? document.body.dataset.ready || '' : ''")
            if ready == "1":
                return json.loads(self.evaluate("document.getElementById('probe-out').textContent"))
            if ready == "error":
                raise RuntimeError(self.evaluate("document.getElementById('probe-out').textContent"))
            time.sleep(0.05)
        raise TimeoutError(f"the render page never became ready: {url}")

    def viewport(self, width_mm, height_mm, width_px=None):
        # The viewport is whole CSS pixels; the scale is taken from that whole
        # width so the capture is exactly width_px wide.
        css_w = max(1, int(round(width_mm * CSS_PX_PER_MM)))
        scale = (width_px / css_w) if width_px else 1.0
        self.call("Emulation.setDeviceMetricsOverride",
                  {"width": css_w, "height": int(round(height_mm * CSS_PX_PER_MM)),
                   "deviceScaleFactor": scale, "mobile": False}, session=True)

    def screenshot(self, path):
        data = self.call("Page.captureScreenshot", {"format": "png"}, session=True)["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def pdf(self, path):
        data = self.call("Page.printToPDF", {"preferCSSPageSize": True, "printBackground": True,
                                             "marginTop": 0, "marginBottom": 0, "marginLeft": 0,
                                             "marginRight": 0}, session=True)["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def close(self):
        try:
            self.call("Browser.close", timeout=5)
        except Exception:
            pass
        self.proc.wait(timeout=10)
        os.close(self._cmd_w)
        os.close(self._out_r)
        shutil.rmtree(self._profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_url(bundle, kind, spec, frame=0, episode=0):
    """file:// URL of render.html for one bundle. The bundle is referenced
    relative to the viewer folder when it can be, else by absolute path."""
    b = os.path.abspath(bundle)
    q = urllib.parse.urlencode({"bundle": "file://" + b, "kind": kind, "spec": json.dumps(spec),
                                "f": frame, "episode": episode})
    return "file://" + RENDER + "?" + q


# ---------------------------------------------------------------------------
# The four outputs
# ---------------------------------------------------------------------------


def figure(bundle, spec, out_pdf, png_dpi=300, chrome=None):
    """A multi-panel figure (scene.figure) as a one-page vector PDF, plus a PNG
    preview. Returns the page's size and layout info."""
    own = chrome is None
    chrome = chrome or Chrome()
    try:
        info = chrome.open(render_url(bundle, "figure", spec))
        os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
        chrome.pdf(out_pdf)
        if png_dpi:
            chrome.viewport(info["width_mm"], info["height_mm"], width_px=round(info["width_mm"] / 25.4 * png_dpi))
            chrome.screenshot(os.path.splitext(out_pdf)[0] + ".png")
        return info
    finally:
        if own:
            chrome.close()


def svg_of(bundle, kind, spec, frame=0, chrome=None):
    own = chrome is None
    chrome = chrome or Chrome()
    try:
        chrome.open(render_url(bundle, kind, spec, frame))
        return chrome.evaluate("document.getElementById('out').innerHTML")
    finally:
        if own:
            chrome.close()


def still(bundle, frame, out, spec=None, width_px=1600, chrome=None):
    """One frame (scene.frameDoc) as PNG, or as SVG when ``out`` ends in .svg."""
    spec = spec or {}
    if out.endswith(".svg"):
        with open(out, "w") as f:
            f.write(svg_of(bundle, "frame", spec, frame, chrome))
        return out
    own = chrome is None
    chrome = chrome or Chrome()
    try:
        info = chrome.open(render_url(bundle, "frame", spec, frame))
        chrome.viewport(info["width_mm"], info["height_mm"], width_px=width_px)
        chrome.screenshot(out)
        return out
    finally:
        if own:
            chrome.close()


def frame_times(last, start=0, end=None, speed=4.0, fps=20, dt=0.1):
    """Fractional frames from start to end at ``speed`` times real time."""
    end = last if end is None else min(end, last)
    per = speed / fps / dt
    out, f = [], float(start)
    while f < end - 1e-9:
        out.append(round(f, 6))
        f += per
    out.append(float(end))
    return out


def frames(bundle, out_dir, spec=None, start=0, end=None, speed=4.0, fps=20, hold_s=2.0, width_px=1080):
    """Render every frame of a clip to out_dir/f00000.png ... Returns the count."""
    spec = spec or {}
    with Chrome() as chrome:
        info = chrome.open(render_url(bundle, "frame", spec, start))
        dt = chrome.evaluate("PEV.render.episode().dims.dt")
        times = frame_times(info["lastStep"], start, end, speed, fps, dt)
        times += [times[-1]] * int(round(hold_s * fps))
        chrome.viewport(info["width_mm"], info["height_mm"], width_px=width_px)
        for n, f in enumerate(times):
            chrome.evaluate(f"PEV.render.frame({f}) && 0")
            chrome.screenshot(os.path.join(out_dir, f"f{n:05d}.png"))
    return len(times)


def video(bundle, out, spec=None, start=0, end=None, speed=4.0, fps=20, hold_s=2.0, width_px=1080, crf=18):
    """H.264 MP4 (yuv420p, faststart), every frame the viewer's drawing."""
    with tempfile.TemporaryDirectory() as d:
        n = frames(bundle, d, spec, start, end, speed, fps, hold_s, width_px)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{d}/f%05d.png",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", str(crf), "-preset", "slow", "-map_metadata", "-1", "-movflags", "+faststart", out],
                       check=True)
    return n


def gif(bundle, out, spec=None, start=0, end=None, speed=8.0, fps=10, hold_s=2.0, width_px=640):
    """GIF through ffmpeg's palettegen/paletteuse (one palette for the clip)."""
    with tempfile.TemporaryDirectory() as d:
        n = frames(bundle, d, spec, start, end, speed, fps, hold_s, width_px)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{d}/f%05d.png",
                        "-vf", "split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", out], check=True)
    return n


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _spec(args):
    spec = json.loads(args.spec) if args.spec else {}
    if getattr(args, "steps", None):
        spec["steps"] = [int(x) for x in args.steps.split(",")]
    if getattr(args, "headings", None):
        spec["headings"] = args.headings.split("|")
    if getattr(args, "title", None):
        spec["title"] = args.title
    if getattr(args, "subtitle", None):
        spec["subtitle"] = args.subtitle
    if getattr(args, "width_mm", None):
        spec["width_mm"] = args.width_mm
    if getattr(args, "layers", None):
        spec["layers"] = {k: v != "0" for k, v in (kv.split(":") for kv in args.layers.split(","))}
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bundle", required=True, help=".replay.js bundle (scripts/export_replay.py)")
    common.add_argument("--out", required=True)
    common.add_argument("--spec", help="JSON spec for scene.js (title, layers, window, ...)")
    common.add_argument("--layers", help="e.g. links:1,orders:1,labels:1 (see scene.FIGURE_LAYERS)")
    common.add_argument("--width-mm", type=float)
    clip = argparse.ArgumentParser(add_help=False)
    clip.add_argument("--start", type=int, default=0)
    clip.add_argument("--end", type=int)
    clip.add_argument("--speed", type=float, help="times real time")
    clip.add_argument("--fps", type=int)
    clip.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    clip.add_argument("--width-px", type=int)
    clip.add_argument("--title")
    clip.add_argument("--subtitle")

    f = sub.add_parser("figure", parents=[common], help="multi-panel figure, vector PDF + PNG preview")
    f.add_argument("--steps", required=True, help="comma-separated steps, one panel each")
    f.add_argument("--headings", help="'|'-separated panel headings")
    s = sub.add_parser("still", parents=[common], help="one frame as PNG or SVG")
    s.add_argument("--frame", type=float, required=True)
    s.add_argument("--width-px", type=int, default=1600)
    s.add_argument("--title")
    s.add_argument("--subtitle")
    sub.add_parser("video", parents=[common, clip], help="H.264 MP4")
    sub.add_parser("gif", parents=[common, clip], help="GIF")
    args = ap.parse_args(argv)
    spec = _spec(args)

    if args.cmd == "figure":
        info = figure(args.bundle, spec, args.out)
        print(f"wrote {args.out} ({info['width_mm']:.1f} x {info['height_mm']:.1f} mm)")
    elif args.cmd == "still":
        still(args.bundle, args.frame, args.out, spec, args.width_px)
        print(f"wrote {args.out}")
    else:
        kw = dict(spec=spec, start=args.start, end=args.end, hold_s=args.hold)
        for k, v in (("speed", args.speed), ("fps", args.fps), ("width_px", args.width_px)):
            if v is not None:
                kw[k] = v
        n = (video if args.cmd == "video" else gif)(args.bundle, args.out, **kw)
        print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    sys.exit(main())
