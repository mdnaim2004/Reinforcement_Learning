#!/usr/bin/env python3
"""
WiFi Animated Radar Sweep Visualizer
=====================================
Real-time rotating radar that "discovers" WiFi networks as the sweep passes.

Requirements:
    pip install matplotlib numpy

Run:
    python wifi_radar_sweep.py
    sudo python wifi_radar_sweep.py   ← for real scan on Linux
"""

import subprocess, platform, re, math, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe


# ──────────────────────────────────────────────────────────────
#  SCANNER  (same cross-platform logic as before)
# ──────────────────────────────────────────────────────────────

def scan_linux():
    try:
        out = subprocess.check_output(
            ["nmcli","-t","-f","SSID,BSSID,SIGNAL,CHAN","dev","wifi","list"],
            encoding="utf-8", errors="ignore")
        nets = []
        for line in out.splitlines():
            p = line.split(":")
            if len(p) >= 4:
                try:
                    nets.append({"ssid": p[0].strip() or "<Hidden>",
                                 "signal": int(p[2])/2 - 100,
                                 "channel": int(p[3]) if p[3].isdigit() else 0,
                                 "bssid": p[1].strip()})
                except ValueError: pass
        return nets
    except: return []

def scan_windows():
    try:
        out = subprocess.check_output(
            ["netsh","wlan","show","networks","mode=bssid"],
            encoding="utf-8", errors="ignore")
        nets = []
        for block in out.split("\n\n"):
            sm = re.search(r"SSID\s+\d+\s*:\s*(.+)", block)
            pm = re.search(r"Signal\s*:\s*(\d+)%", block)
            cm = re.search(r"Channel\s*:\s*(\d+)", block)
            bm = re.search(r"BSSID\s+\d+\s*:\s*([\da-fA-F:]+)", block)
            if sm and pm:
                nets.append({"ssid": sm.group(1).strip() or "<Hidden>",
                             "signal": int(pm.group(1))/2 - 100,
                             "channel": int(cm.group(1)) if cm else 0,
                             "bssid": bm.group(1) if bm else "??"})
        return nets
    except: return []

def scan_macos():
    airport = ("/System/Library/PrivateFrameworks/Apple80211.framework"
               "/Versions/Current/Resources/airport")
    try:
        out = subprocess.check_output([airport,"-s"], encoding="utf-8", errors="ignore")
        nets = []
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    sig = int(parts[-5])
                    ch  = int(re.sub(r"\D","", parts[-4]) or 0)
                    nets.append({"ssid":" ".join(parts[:-6]).strip() or "<Hidden>",
                                 "signal": sig, "channel": ch, "bssid": parts[-6]})
                except: pass
        return nets
    except: return []

def demo_networks(n=16):
    names = ["HomeNet","NETGEAR_5G","xfinitywifi","TP-Link_7A3F",
             "AndroidHotspot","Linksys","ATT-WiFi","CoffeeShop_Guest",
             "FBI_Surveillance_Van","Pretty_Fly_For_A_WiFi",
             "Loading...","404_Network_Not_Found","TellMyWiFiLoveHer",
             "HideYoKidsHideYoWifi","The_LAN_Before_Time","BillsWifi"]
    rng = random.Random(7)
    result = []
    for i in range(min(n, len(names))):
        ch = rng.choice([1,6,11,36,40,44,48,149])
        result.append({"ssid": names[i],
                       "signal": rng.randint(-85,-35),
                       "channel": ch,
                       "bssid": ":".join(f"{rng.randint(0,255):02X}" for _ in range(6))})
    return result

def get_networks():
    sys = platform.system()
    if sys == "Linux":   nets = scan_linux()
    elif sys == "Darwin": nets = scan_macos()
    elif sys == "Windows": nets = scan_windows()
    else: nets = []
    if not nets:
        print("[!] Real scan failed → using demo data")
        nets = demo_networks()
    else:
        print(f"[+] Found {len(nets)} real networks")
    return nets


# ──────────────────────────────────────────────────────────────
#  LAYOUT HELPERS
# ──────────────────────────────────────────────────────────────

def dbm_to_r(dbm, lo=-90, hi=-30, max_r=0.88):
    """Stronger signal → smaller radius (closer to center)."""
    t = (max(lo, min(hi, dbm)) - lo) / (hi - lo)   # 0=weak, 1=strong
    return max_r * (1 - t * 0.85)

def assign_positions(networks, seed=42):
    rng = random.Random(seed)
    positions = []
    used = []
    for i, net in enumerate(networks):
        base = (i / len(networks)) * 360
        angle = base + rng.uniform(-10, 10)
        for ua in used:
            while abs((angle - ua) % 360) < 7:
                angle += 8
        used.append(angle)
        r = dbm_to_r(net["signal"])
        rad = math.radians(angle)
        positions.append((r * math.cos(rad), r * math.sin(rad), angle))
    return positions

def ch_color(ch):
    palette_24 = ["#00FF41","#39FF14","#7FFF00","#ADFF2F",
                  "#CCFF00","#80FF80","#00FF80","#00FFAA",
                  "#00FFD1","#00FFE5","#00FFFF","#00F5FF"]
    palette_5  = ["#FF6B35","#FF9F1C","#FFBF69","#FFD166","#F4A261"]
    if 1 <= ch <= 11:  return palette_24[(ch-1) % len(palette_24)]
    if ch > 11:        return palette_5[(ch % len(palette_5))]
    return "#AAAAAA"

def sig_label(dbm):
    if dbm >= -50: return "EXCELLENT"
    if dbm >= -60: return "GOOD"
    if dbm >= -70: return "FAIR"
    return "WEAK"


# ──────────────────────────────────────────────────────────────
#  ANIMATION
# ──────────────────────────────────────────────────────────────

class RadarViz:
    SWEEP_SPEED = 1.8          # degrees per frame
    TRAIL_DEG   = 90           # length of fading trail
    FRAMES      = int(360 / SWEEP_SPEED) + 20
    FADE_FRAMES = 60           # how many frames a blip stays bright

    def __init__(self, networks):
        self.networks  = networks
        self.positions = assign_positions(networks)
        # angle in degrees for each network dot
        self.net_angles = [p[2] for p in self.positions]

        self.fig = plt.figure(figsize=(10,10), facecolor="#000A00")
        self.ax  = self.fig.add_subplot(111, facecolor="#000A00")
        self.ax.set_xlim(-1.15, 1.15)
        self.ax.set_ylim(-1.15, 1.15)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.sweep_angle = 0.0
        # last_seen[i] = frame number when network i was last hit by sweep
        self.last_seen = [-9999] * len(networks)
        self.frame_no  = 0

        self._build_static()
        self._build_dynamic()

    # ── Static elements ──────────────────────────────────────

    def _build_static(self):
        ax = self.ax
        # Rings
        for dbm, lbl in [(-40,"−40"),(-55,"−55"),(-70,"−70"),(-85,"−85")]:
            r = dbm_to_r(dbm)
            c = plt.Circle((0,0), r, fill=False,
                            color="#003300", linewidth=0.8, linestyle="--")
            ax.add_patch(c)
            ax.text(r*0.72, r*0.72, f"{lbl} dBm",
                    color="#005500", fontsize=6, fontfamily="monospace")

        # Outer border ring
        outer = plt.Circle((0,0), 0.93, fill=False, color="#00AA00",
                            linewidth=1.5)
        ax.add_patch(outer)

        # Cross-hairs
        for a in [0,45,90,135]:
            rad = math.radians(a)
            ax.plot([-math.cos(rad), math.cos(rad)],
                    [-math.sin(rad), math.sin(rad)],
                    color="#002200", linewidth=0.5, alpha=0.8)

        # Cardinal labels
        for label, angle in [("N",90),("E",0),("S",270),("W",180)]:
            rad = math.radians(angle)
            ax.text(0.98*math.cos(rad), 0.98*math.sin(rad), label,
                    color="#00AA00", fontsize=8, ha="center", va="center",
                    fontfamily="monospace", fontweight="bold")

        # Scanline tick marks on outer ring
        for deg in range(0, 360, 10):
            r1, r2 = (0.90, 0.93) if deg % 30 == 0 else (0.915, 0.93)
            rad = math.radians(deg)
            ax.plot([r1*math.cos(rad), r2*math.cos(rad)],
                    [r1*math.sin(rad), r2*math.sin(rad)],
                    color="#007700", linewidth=0.8)

        # Center dot
        ax.plot(0,0,"o", color="#00FF41", markersize=4, zorder=20)

        # Title
        ax.text(0, 1.10, "◈  W I F I  R A D A R  ◈",
                color="#00FF41", fontsize=15, fontweight="bold",
                ha="center", fontfamily="monospace",
                path_effects=[pe.withStroke(linewidth=3, foreground="#000A00")])
        ax.text(0, 1.045, f"{len(self.networks)} networks  •  sweeping 360°",
                color="#007700", fontsize=8, ha="center", fontfamily="monospace")

    # ── Dynamic elements (rebuilt each frame) ────────────────

    def _build_dynamic(self):
        self.sweep_artists  = []
        self.blip_artists   = []

    def _clear_dynamic(self):
        for a in self.sweep_artists + self.blip_artists:
            try: a.remove()
            except: pass
        self.sweep_artists = []
        self.blip_artists  = []

    # ── Per-frame update ─────────────────────────────────────

    def update(self, frame):
        self.frame_no = frame
        self._clear_dynamic()
        ax = self.ax

        sa = self.sweep_angle   # current sweep angle in degrees

        # ── Check which networks got hit this frame ──────────
        for i, net_angle in enumerate(self.net_angles):
            # Normalise difference into [0, 360)
            diff = (sa - net_angle) % 360
            if diff < self.SWEEP_SPEED * 2:
                self.last_seen[i] = frame

        # ── Draw sweep trail (gradient of alpha wedges) ──────
        n_wedge = 60
        for k in range(n_wedge):
            frac   = k / n_wedge               # 0=leading edge, 1=trailing end
            alpha  = (1 - frac) * 0.55
            start  = sa - frac * self.TRAIL_DEG
            end    = sa - (frac + 1/n_wedge) * self.TRAIL_DEG
            wedge  = mpatches.Wedge(
                (0,0), 0.92, end, start,
                facecolor=(0, 1.0, 0.25, alpha),
                edgecolor="none", zorder=3)
            ax.add_patch(wedge)
            self.sweep_artists.append(wedge)

        # ── Bright sweep leading edge ─────────────────────────
        rad = math.radians(sa)
        line, = ax.plot([0, 0.93*math.cos(rad)],
                        [0, 0.93*math.sin(rad)],
                        color="#00FF41", linewidth=1.6, alpha=0.95, zorder=10)
        self.sweep_artists.append(line)

        # ── Draw network blips ────────────────────────────────
        for i, (x, y, ang) in enumerate(self.positions):
            net = self.networks[i]
            frames_since = frame - self.last_seen[i]

            if frames_since > self.FADE_FRAMES:
                # Draw faint ghost dot
                dot, = ax.plot(x, y, "o", color="#003300",
                               markersize=4, zorder=5, alpha=0.3)
                self.blip_artists.append(dot)
                continue

            # Blip brightness fades over FADE_FRAMES frames
            t     = 1 - frames_since / self.FADE_FRAMES
            col   = ch_color(net["channel"])
            glow  = t * 0.7
            size  = 5 + t * 8

            # Glow halos
            for hr, ha in [(size*2.6, glow*0.18), (size*1.6, glow*0.35)]:
                h, = ax.plot(x, y, "o", color=col,
                             markersize=hr, alpha=ha, zorder=6)
                self.blip_artists.append(h)

            # Main blip
            dot, = ax.plot(x, y, "o", color=col, markersize=size,
                           alpha=min(1.0, 0.5 + t*0.5), zorder=8,
                           markeredgecolor="#000A00", markeredgewidth=0.5)
            self.blip_artists.append(dot)

            # Label — only show when freshly hit
            if frames_since < 25:
                label_alpha = t * 0.95
                offset = 0.06
                rad_n  = math.radians(ang)
                lx = x + offset * math.cos(rad_n)
                ly = y + offset * math.sin(rad_n)

                txt = ax.text(
                    lx, ly,
                    net["ssid"][:16] + ("…" if len(net["ssid"])>16 else ""),
                    color=col, fontsize=6.5, fontfamily="monospace",
                    ha="left" if x >= 0 else "right",
                    va="bottom" if y >= 0 else "top",
                    alpha=label_alpha, zorder=9,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#000A00")])
                self.blip_artists.append(txt)

                dbm_txt = ax.text(
                    x, y - 0.055,
                    f"{net['signal']}dBm",
                    color="#00FF41", fontsize=5, fontfamily="monospace",
                    ha="center", alpha=label_alpha*0.8, zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="#000A00")])
                self.blip_artists.append(dbm_txt)

        # ── Advance sweep angle (clockwise = subtract) ────────
        self.sweep_angle = (self.sweep_angle - self.SWEEP_SPEED) % 360

        return self.sweep_artists + self.blip_artists

    # ── Run ──────────────────────────────────────────────────

    def run(self):
        total_frames = 0        # 0 = loop forever
        anim = FuncAnimation(
            self.fig, self.update,
            frames=None,         # infinite loop
            interval=30,         # ~33 fps
            blit=False,
            cache_frame_data=False,
            repeat=True
        )
        plt.tight_layout(pad=0)
        plt.show()


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nets = get_networks()
    nets.sort(key=lambda x: x["signal"], reverse=True)

    print(f"\n{'#':<4} {'SSID':<22} {'Signal':>9}  {'Quality':<10}  Ch")
    print("─" * 55)
    for i, n in enumerate(nets, 1):
        print(f"{i:<4} {n['ssid'][:21]:<22} {n['signal']:>7} dBm"
              f"  {sig_label(n['signal']):<10}  {n['channel']}")

    print("\n[*] Opening animated radar... (close window to quit)\n")
    RadarViz(nets).run()