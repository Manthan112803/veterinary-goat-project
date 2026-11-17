"""
Goat AI Training Model - 4-Panel Dashboard
-------------------------------------------
 ┌──────────────────────┬──────────────────────┐
 │ TOP LEFT             │ TOP RIGHT            │
 │ Demo / training video│ Main camera (TPU     │
 │                      │ part, rod goes in)   │
 ├──────────────────────┼──────────────────────┤
 │ BOTTOM LEFT          │ BOTTOM RIGHT         │
 │ Second camera        │ OpenCV output: rod   │
 │ (CAM_SECOND)         │ shadow outlined      │
 └──────────────────────┴──────────────────────┘

Install:  pip install PyQt5 opencv-python numpy
Run:      python infinal.py        (no arguments needed)

Fixed setup (change the constants under "Fixed settings" below if needed):
  training video = demo.mp4 next to this file
  top right   = external USB webcam (CAM_MAIN)  <- rod shadow detection runs on this one
  bottom left = second camera       (CAM_SECOND)
  Camera numbers differ per PC - press T / B while running to find the right ones, then set them below.
  college logo   = ycce_logo.png next to this file (shown top right; skipped if missing)

Keys:
  C          capture reference (rod OUT, LEDs ON)  <- do this first
  Mouse drag on the top-right camera = select the TPU part (ROI)
  R          reset ROI to full frame
  T          switch top-right camera to the next camera number (0,1,2...) - use this to find the USB cam
  B          switch bottom-left camera to the next camera number
  E          cycle the side the rod enters from (left/right/top/bottom)
  Up / Down  shadow sensitivity (Darkness %)
  Space      pause / play the demo video
  P          restart the demo video from the beginning
  S          save snapshot of the OpenCV output
  H          show / hide the bottom hint bar (the scrolling credit line stays always on)
  F11        fullscreen on/off      Esc  leave fullscreen      Q  quit
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# ----------------------------- Theme (matches the PPT) -----------------------------
BG = "#060d18"          # window background (near-black navy)
PANEL = "#0b1628"       # card background
BORDER = "#16314f"      # hairline borders
TEAL = "#2dd4bf"        # single accent colour (also used for the OpenCV overlays)
TEXT = "#e8eef7"
TEXT_SOFT = "#b6c4d8"
MUTED = "#6f86a3"
AMBER = "#fbbf24"
VIEW_BG = "#050b14"     # behind the video frames
VIEW_STYLE = f"color:{MUTED}; background:{VIEW_BG}; border:1px solid #0f2238; border-radius:8px;"


def rgba(hex_color, alpha):
    """'#rrggbb' + alpha 0..1  ->  'rgba(r,g,b,a)' for style sheets."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


ENTRY_SIDES = ["left", "right", "top", "bottom"]
PROJECT_TITLE = "GOAT AI TRAINING MODEL  ·  ROD SHADOW TRACKER"

# ----------------------------- Fixed settings (edit here) -----------------------------
# Folder of this script (or of the .exe if you package it later)
BASE = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
VIDEO_FILE = os.path.join(BASE, "demo.mp4")   # training video, keep it next to this file
CAM_MAIN = 1              # camera number: external USB webcam (top right + OpenCV). Laptop cam is usually 0. Press T to cycle live
                          # (the earliest version of this project used CAM_MAIN = 2, CAM_SECOND = 4)
CAM_SECOND = 2            # camera number: second camera (bottom left). Press B to cycle live
MAX_CAM_INDEX = 6        # T / B keys cycle through camera numbers 0 .. MAX_CAM_INDEX
HEADING = "GOAT ARTIFICIAL INSEMINATION"      # title shown at the top of the screen
LOGO_FILES = ["ycce_logo.png", "ycce_logo.jpg", "ycce_logo.jpeg", "ycce_logo.webp"]  # first one found is used
LOGO_HEIGHT = 64          # logo height in pixels (top right)
START_FULLSCREEN = False  # set True for the final monitor setup (Esc / F11 leaves it)
LOCK_CAMERA = False       # True = freeze auto exposure/focus (only if brightness pumps)

# Scrolling credit line at the very bottom of the window
TICKER_LINE_1 = "This simulation is made by "
TICKER_LINE_2 = "This simulation setup is developed by "
TICKER_COLLEGE = "Yeshwantrao Chavan College of Engineering, Nagpur"   # shown in CAPITALS + bold
TICKER_GAP = "        \u25c6        "   # separator between each repeat of the line
TICKER_COLLEGE_PT = 11    # college name font size (bold, capitals)
TICKER_TEXT_PT = 10       # rest of the line: one size smaller
TICKER_SPEED = 70         # pixels per second, higher = faster
TICKER_HEIGHT = 36        # height of the bar in pixels


# ----------------------------- Video sources (background threads) -----------------------------
class Grabber(threading.Thread):
    """Reads a camera or video file in the background and keeps the newest frame."""

    def __init__(self, source, is_file=False, lock_exposure=False):
        super().__init__(daemon=True)
        self.source = source
        self.is_file = is_file
        self.lock_exposure = lock_exposure
        self.frame = None
        self.ok = False
        self.paused = False
        self.restart = False   # set True to jump back to the first frame
        self.running = True
        self._lock = threading.Lock()

    def run(self):
        cap = None
        for _ in range(8):   # retry: the previous owner of this camera may still be releasing it
            if not self.running:
                return
            if not self.is_file and os.name == "nt":
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # more reliable on Windows
            else:
                cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.3)
        if cap is None or not cap.isOpened():
            self.ok = False
            return
        self.ok = True

        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Webcams on Windows REMEMBER the last settings they were given, so we always set
            # white balance back to AUTO (the old manual setting caused the colour cast).
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            if self.lock_exposure:  # optional: keep brightness stable for shadow detection
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            else:                   # normal: put exposure and focus back to automatic
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # auto
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30

        while self.running:
            t0 = time.time()
            if self.restart:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.paused = False
                self.restart = False
            if self.paused:
                time.sleep(0.05)
                continue
            ok, f = cap.read()
            if not ok:
                if self.is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                else:
                    time.sleep(0.05)
                continue
            with self._lock:
                self.frame = f
            if self.is_file:
                time.sleep(max(0.0, delay - (time.time() - t0)))
        cap.release()

    def get(self):
        with self._lock:
            return self.frame

    def stop(self):
        self.running = False


# ----------------------------- Shadow detection (same idea as the tracker script) -----------------------------
def preprocess(frame, roi, blur_k):
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = max(3, blur_k) | 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return crop, gray.astype(np.float32)


def find_shadow(gray, ref, drop_pct, min_area):
    ratio = gray / ref                       # 1.0 = same as the rod-out reference
    mask = (ratio < 1.0 - drop_pct / 100.0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return max(cnts, key=cv2.contourArea) if cnts else None


def tip_point(contour, side):
    pts = contour.reshape(-1, 2)
    if side == "left":
        return tuple(int(v) for v in pts[pts[:, 0].argmax()])
    if side == "right":
        return tuple(int(v) for v in pts[pts[:, 0].argmin()])
    if side == "top":
        return tuple(int(v) for v in pts[pts[:, 1].argmax()])
    return tuple(int(v) for v in pts[pts[:, 1].argmin()])


# ----------------------------- UI pieces -----------------------------
class ViewLabel(QtWidgets.QLabel):
    """Shows a frame scaled to fit (keeps aspect ratio). Optionally lets you drag a box."""

    selected = QtCore.pyqtSignal(int, int, int, int)  # x, y, w, h in FRAME coordinates

    def __init__(self, selectable=False):
        super().__init__()
        self.selectable = selectable
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(160, 90)
        self.setStyleSheet(VIEW_STYLE)
        self._frame_wh = None
        self._origin = None
        self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def show_text(self, text):
        self._frame_wh = None
        self.setPixmap(QtGui.QPixmap())
        self.setText(text)

    def show_frame(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._frame_wh = (w, h)
        self.setPixmap(pix)

    # --- mouse -> frame coordinates ---
    def _to_frame(self, pos):
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._frame_wh is None:
            return None
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        fx = (pos.x() - ox) / pm.width() * self._frame_wh[0]
        fy = (pos.y() - oy) / pm.height() * self._frame_wh[1]
        return fx, fy

    def mousePressEvent(self, e):
        if self.selectable and e.button() == QtCore.Qt.LeftButton:
            self._origin = e.pos()
            self._band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._band.show()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._band.setGeometry(QtCore.QRect(self._origin, e.pos()).normalized())

    def mouseReleaseEvent(self, e):
        if self._origin is None:
            return
        a = self._to_frame(self._origin)
        b = self._to_frame(e.pos())
        self._origin = None
        self._band.hide()
        if a is None or b is None:
            return
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        w, h = self._frame_wh
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 > 20 and y1 - y0 > 20:
            self.selected.emit(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


class VideoPanel(QtWidgets.QFrame):
    """A card with a title, a status pill and HUD-style corner brackets."""

    def __init__(self, title, selectable=False):
        super().__init__()
        self.setObjectName("panel")
        self._last_status = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(10)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        dot = QtWidgets.QLabel("\u25cf")
        dot.setFont(QtGui.QFont("Segoe UI", 6))
        dot.setStyleSheet(f"color:{TEAL};")
        self.title = QtWidgets.QLabel(title.upper())
        f = QtGui.QFont("Segoe UI", 9, QtGui.QFont.DemiBold)
        f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 2.5)
        self.title.setFont(f)
        self.title.setStyleSheet(f"color:{TEXT_SOFT};")
        self.status = QtWidgets.QLabel("")
        self.status.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Bold))
        self.status.hide()
        head.addWidget(dot)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.status)
        lay.addLayout(head)

        self.view = ViewLabel(selectable)
        lay.addWidget(self.view, 1)

    def set_status(self, text, color=MUTED):
        key = (text, color)
        if key == self._last_status:      # called every frame, only touch the widget on change
            return
        self._last_status = key
        if not text:
            self.status.hide()
            return
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{color}; background:{rgba(color, 0.12)}; "
            f"border:1px solid {rgba(color, 0.45)}; border-radius:9px; padding:2px 10px;")
        self.status.show()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(TEAL))
        pen.setWidthF(1.6)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        w, h, m, L = self.width(), self.height(), 7, 14
        for cx, cy, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            p.drawLine(cx, cy, cx + dx * L, cy)
            p.drawLine(cx, cy, cx, cy + dy * L)
        p.end()


def _text_width(font, text):
    fm = QtGui.QFontMetrics(font)
    return fm.horizontalAdvance(text) if hasattr(fm, "horizontalAdvance") else fm.width(text)


class Ticker(QtWidgets.QWidget):
    """Slim, continuously scrolling credit line (right to left) with soft faded edges."""

    FADE = 90

    def __init__(self):
        super().__init__()
        self.setFixedHeight(TICKER_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        f_text = QtGui.QFont("Segoe UI", TICKER_TEXT_PT)
        f_text.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1.0)
        f_college = QtGui.QFont("Segoe UI", TICKER_COLLEGE_PT, QtGui.QFont.Bold)
        f_college.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 1.8)
        f_sep = QtGui.QFont("Segoe UI", TICKER_TEXT_PT - 3)

        # (text, font, colour)
        self.parts = [
            (TICKER_LINE_1, f_text, QtGui.QColor(TEXT_SOFT)),
            (TICKER_COLLEGE.upper(), f_college, QtGui.QColor(TEXT)),
            (".      ", f_text, QtGui.QColor(TEXT_SOFT)),
            (TICKER_LINE_2, f_text, QtGui.QColor(TEXT_SOFT)),
            (TICKER_COLLEGE.upper(), f_college, QtGui.QColor(TEXT)),
            (".", f_text, QtGui.QColor(TEXT_SOFT)),
            (TICKER_GAP, f_sep, QtGui.QColor(TEAL)),
        ]
        self.widths = [_text_width(f, t) for t, f, _ in self.parts]
        self.baselines = []
        for _, f, _ in self.parts:
            fm = QtGui.QFontMetrics(f)
            self.baselines.append((TICKER_HEIGHT + fm.ascent() - fm.descent()) // 2)
        self.total = max(1, sum(self.widths))

        self.offset = 0.0
        self._last = time.monotonic()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(16)

    def _step(self):
        now = time.monotonic()
        self.offset = (self.offset + TICKER_SPEED * (now - self._last)) % self.total
        self._last = now
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor("#040a13"))

        # hairline accent on top that fades out to both sides
        g = QtGui.QLinearGradient(0, 0, w, 0)
        clear = QtGui.QColor(TEAL)
        clear.setAlpha(0)
        mid = QtGui.QColor(TEAL)
        mid.setAlpha(150)
        g.setColorAt(0.0, clear)
        g.setColorAt(0.5, mid)
        g.setColorAt(1.0, clear)
        p.fillRect(0, 0, w, 1, QtGui.QBrush(g))

        x = int(-self.offset)
        while x < w:                       # repeat the line so there is never an empty gap
            for (text, font, color), width, base in zip(self.parts, self.widths, self.baselines):
                if x + width > 0 and x < w:
                    p.setFont(font)
                    p.setPen(color)
                    p.drawText(x, base, text)
                x += width

        # soft fade at the left and right edge
        bg = QtGui.QColor("#040a13")
        bg0 = QtGui.QColor("#040a13")
        bg0.setAlpha(0)
        gl = QtGui.QLinearGradient(0, 0, self.FADE, 0)
        gl.setColorAt(0, bg)
        gl.setColorAt(1, bg0)
        p.fillRect(0, 0, self.FADE, h, QtGui.QBrush(gl))
        gr = QtGui.QLinearGradient(w - self.FADE, 0, w, 0)
        gr.setColorAt(0, bg0)
        gr.setColorAt(1, bg)
        p.fillRect(w - self.FADE, 0, self.FADE, h, QtGui.QBrush(gr))
        p.end()


# ----------------------------- Main window -----------------------------
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goat Artificial Insemination")
        self.resize(1400, 860)

        # video sources
        self.cam_main = CAM_MAIN
        self.cam_second = CAM_SECOND
        self.g_video = Grabber(VIDEO_FILE, is_file=True)
        self.g_main = Grabber(self.cam_main, lock_exposure=LOCK_CAMERA)
        self.g_second = Grabber(self.cam_second)
        for g in (self.g_video, self.g_main, self.g_second):
            g.start()

        # detection state
        self.roi = None            # (x, y, w, h) in main-camera frame coords; None = full frame
        self.ref = None
        self.calib_left = 0
        self.calib_acc = None
        self.calib_n = 30
        self.side_idx = 0
        self.drop = 35             # % darker than reference counts as shadow
        self.min_area = 400
        self.blur = 7
        self.tip_hist = []
        self.last_output = None

        # layout
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(14, 10, 14, 8)
        outer.setSpacing(8)

        # header: centred heading + subtitle, college logo on the right
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(6, 0, 6, 6)
        logo_pix = self._load_logo()
        title = QtWidgets.QLabel(HEADING)
        title.setAlignment(QtCore.Qt.AlignCenter)
        tf = QtGui.QFont("Segoe UI", 22, QtGui.QFont.DemiBold)
        tf.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 7)
        title.setFont(tf)
        title.setStyleSheet(f"color:{TEXT};")
        subtitle = QtWidgets.QLabel(PROJECT_TITLE)
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        sf = QtGui.QFont("Segoe UI", 9)
        sf.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 4)
        subtitle.setFont(sf)
        subtitle.setStyleSheet(f"color:{TEAL};")
        centre = QtWidgets.QVBoxLayout()
        centre.setSpacing(2)
        centre.addWidget(title)
        centre.addWidget(subtitle)
        left_pad = QtWidgets.QWidget()          # balances the logo so the heading stays centred
        hl.addWidget(left_pad)
        hl.addLayout(centre, 1)
        if logo_pix is not None:
            logo = QtWidgets.QLabel()
            logo.setPixmap(logo_pix)
            # soft white badge so any logo looks right on the dark background
            logo.setStyleSheet("background:#f4f7fb; border-radius:12px; padding:6px;")
            hl.addWidget(logo)
            left_pad.setFixedWidth(logo_pix.width() + 26)
        else:
            left_pad.setFixedWidth(0)
        outer.addWidget(header)

        # thin accent line under the header (fades out at both ends)
        accent = QtWidgets.QFrame()
        accent.setFixedHeight(2)
        accent.setStyleSheet(
            f"border:none; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, "
            f"stop:0 {rgba(TEAL, 0.0)}, stop:0.5 {TEAL}, stop:1 {rgba(TEAL, 0.0)});")
        outer.addWidget(accent)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(14)
        self.p_video = VideoPanel("Training video")
        self.p_main = VideoPanel("Main camera  ·  drag to select part", selectable=True)
        self.p_second = VideoPanel("Second camera")
        self.p_output = VideoPanel("OpenCV output  ·  rod shadow")
        grid.addWidget(self.p_video, 0, 0)    # top left
        grid.addWidget(self.p_main, 0, 1)     # top right
        grid.addWidget(self.p_second, 1, 0)   # bottom left
        grid.addWidget(self.p_output, 1, 1)   # bottom right
        for i in (0, 1):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        outer.addLayout(grid, 1)

        self.footer = QtWidgets.QLabel(
            f"{PROJECT_TITLE}      C calibrate · drag = select part · R reset · T/B switch camera · E entry side · "
            "↑/↓ sensitivity · Space video · P replay video · S snapshot · H hide · F11 fullscreen · Q quit")
        self.footer.setFont(QtGui.QFont("Segoe UI", 8))
        self.footer.setStyleSheet(f"color:{MUTED};")
        outer.addWidget(self.footer)

        # scrolling credit line at the very bottom of the whole window
        self.ticker = Ticker()
        outer.addWidget(self.ticker)

        self.setStyleSheet(f"""
            QMainWindow, QWidget#root {{ background:{BG}; }}
            QFrame#panel {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                              stop:0 #0d1b30, stop:1 {PANEL});
                            border:1px solid {BORDER}; border-radius:12px; }}
            QFrame#panel QLabel {{ border:none; background:transparent; }}
            QFrame#header {{ background:transparent; border:none; }}
        """)
        # the video labels keep their own darker background
        for p in (self.p_video, self.p_main, self.p_second, self.p_output):
            p.view.setStyleSheet(VIEW_STYLE)

        self.p_main.view.selected.connect(self.set_roi)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---------- logo ----------
    def _load_logo(self):
        """Returns the scaled logo pixmap, or None if no logo file is next to this script."""
        for name in LOGO_FILES:
            path = os.path.join(BASE, name)
            if os.path.exists(path):
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    return pix.scaledToHeight(LOGO_HEIGHT, QtCore.Qt.SmoothTransformation)
        return None

    # ---------- actions ----------
    def set_roi(self, x, y, w, h):
        self.roi = (x, y, w, h)
        self.ref = None
        self.tip_hist.clear()

    def next_camera(self, which):
        """Switch the top-right ('main') or bottom-left ('second') panel to the next camera number."""
        if which == "main":
            self.g_main.stop()
            self.cam_main = (self.cam_main + 1) % (MAX_CAM_INDEX + 1)
            self.g_main = Grabber(self.cam_main, lock_exposure=LOCK_CAMERA)
            self.g_main.start()
            self.ref = None
            self.calib_left = 0
            self.tip_hist.clear()
        else:
            self.g_second.stop()
            self.cam_second = (self.cam_second + 1) % (MAX_CAM_INDEX + 1)
            self.g_second = Grabber(self.cam_second)
            self.g_second.start()

    def start_calibration(self):
        self.calib_left = self.calib_n
        self.calib_acc = None
        self.tip_hist.clear()

    # ---------- main loop ----------
    def show_source(self, panel, grabber, label):
        f = grabber.get()
        if f is None:
            panel.view.show_text("NO SIGNAL" if not grabber.ok else "LOADING…")
            panel.set_status("NO SIGNAL" if not grabber.ok else "", MUTED)
            return None
        panel.view.show_frame(f)
        return f

    def tick(self):
        self.show_source(self.p_video, self.g_video, "video")
        if self.show_source(self.p_second, self.g_second, "cam2") is not None:
            self.p_second.set_status(f"CAM {self.cam_second}  ·  B = next camera", MUTED)
        else:
            self.p_second.set_status(f"CAM {self.cam_second}  ·  NO SIGNAL  ·  B = next camera", MUTED)

        frame = self.g_main.get()
        if frame is None:
            self.p_main.view.show_text("NO SIGNAL" if not self.g_main.ok else "LOADING…")
            self.p_main.set_status(f"CAM {self.cam_main}  ·  NO SIGNAL  ·  T = next camera", AMBER)
            self.p_output.view.show_text("WAITING FOR CAMERA")
            return

        H, W = frame.shape[:2]
        x, y, w, h = self.roi if self.roi else (0, 0, W, H)
        x, y = min(x, W - 2), min(y, H - 2)
        w, h = min(w, W - x), min(h, H - y)
        roi = (x, y, w, h)

        # main camera panel: draw ROI box so you can see what's being analysed
        shown = frame.copy()
        if self.roi:
            cv2.rectangle(shown, (x, y), (x + w, y + h), (191, 212, 45), 2)  # teal (BGR)
        self.p_main.view.show_frame(shown)
        self.p_main.set_status(f"CAM {self.cam_main}  ·  T = next camera", MUTED)

        crop, gray = preprocess(frame, roi, self.blur)
        view = crop.copy()

        # --- calibration in progress ---
        if self.calib_left > 0:
            self.calib_acc = gray.copy() if self.calib_acc is None else self.calib_acc + gray
            self.calib_left -= 1
            if self.calib_left == 0:
                self.ref = np.maximum(self.calib_acc / self.calib_n, 10.0)
            self.p_output.set_status(f"CALIBRATING… {self.calib_n - self.calib_left}/{self.calib_n}", AMBER)
        elif self.ref is None or self.ref.shape != gray.shape:
            self.ref = None
            self.p_output.set_status("PRESS C TO CALIBRATE (ROD OUT)", AMBER)
        else:
            cnt = find_shadow(gray, self.ref, self.drop, self.min_area)
            if cnt is not None:
                overlay = view.copy()
                cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), -1)
                view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)
                cv2.drawContours(view, [cnt], -1, (191, 212, 45), 2)  # teal outline

                self.tip_hist.append(tip_point(cnt, ENTRY_SIDES[self.side_idx]))
                self.tip_hist = self.tip_hist[-5:]
                sx = int(np.mean([p[0] for p in self.tip_hist]))
                sy = int(np.mean([p[1] for p in self.tip_hist]))
                cv2.circle(view, (sx, sy), 10, (36, 191, 251), 2)
                cv2.drawMarker(view, (sx, sy), (36, 191, 251), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(view, "ROD TIP", (sx + 14, sy - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 191, 251), 2)
                self.p_output.set_status(
                    f"ROD DETECTED  ·  enters {ENTRY_SIDES[self.side_idx]}  ·  sens {self.drop}%", TEAL)
            else:
                self.tip_hist.clear()
                self.p_output.set_status(
                    f"NO ROD  ·  enters {ENTRY_SIDES[self.side_idx]}  ·  sens {self.drop}%", MUTED)

        self.last_output = view
        self.p_output.view.show_frame(view)

    # ---------- keys ----------
    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Q:
            self.close()
        elif k == QtCore.Qt.Key_C:
            self.start_calibration()
        elif k == QtCore.Qt.Key_R:
            self.roi = None
            self.ref = None
            self.tip_hist.clear()
        elif k == QtCore.Qt.Key_T:
            self.next_camera("main")
        elif k == QtCore.Qt.Key_B:
            self.next_camera("second")
        elif k == QtCore.Qt.Key_E:
            self.side_idx = (self.side_idx + 1) % len(ENTRY_SIDES)
            self.tip_hist.clear()
        elif k == QtCore.Qt.Key_Up:
            self.drop = min(90, self.drop + 5)
        elif k == QtCore.Qt.Key_Down:
            self.drop = max(5, self.drop - 5)
        elif k == QtCore.Qt.Key_Space:
            self.g_video.paused = not self.g_video.paused
        elif k == QtCore.Qt.Key_P:
            self.g_video.restart = True
        elif k == QtCore.Qt.Key_S and self.last_output is not None:
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, self.last_output)
        elif k == QtCore.Qt.Key_H:
            self.footer.setVisible(not self.footer.isVisible())
        elif k == QtCore.Qt.Key_F11:
            self.setWindowState(self.windowState() ^ QtCore.Qt.WindowFullScreen)
        elif k == QtCore.Qt.Key_Escape and self.isFullScreen():
            self.showNormal()

    def closeEvent(self, e):
        for g in (self.g_video, self.g_main, self.g_second):
            g.stop()
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard()
    if START_FULLSCREEN:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()