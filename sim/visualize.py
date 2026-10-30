"""
Replay a mission recorded by runMission as a matplotlib animation.

Generated code (presentation only): reads AuctionResult.state_history and draws
it. Nothing here feeds back into the simulation.

    anim = animateMission(result, agents, mission_data)
    plt.show()                                  # or
    saveAnimation(anim, "mission.mp4")          # needs ffmpeg; ".gif" uses Pillow

Encoding
  - one colour per capability (Okabe-Ito, colour-blind safe), shown in the legend
  - drones: icon per platform class, rotated by heading; dark = assigned, faded = idle;
    coloured pips beside the icon = the drone's capabilities; dashed line to its slot
  - targets: dark X; under it one square per slot, coloured by the role it needs
    (a split square needs ONE drone with ALL of its colours), solid = held,
    hollow = open, grey dashed with a cross = no drone in the swarm can fill it;
    "held/total" count beside the squares
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, offset_copy

from swarm.agent import Agent
from swarm.models import AuctionResult, Capability, MissionData, PlatformClass, Role

# ---------- colours ----------

CAPABILITY_COLOURS = {
    Capability.SURVEILLANCE: "#0072B2",        # blue
    Capability.MAPPING: "#E69F00",             # orange
    Capability.COMMS_RELAY: "#009E73",         # bluish green
    Capability.SIGNALS_COLLECTION: "#CC79A7",  # reddish purple
    Capability.PAYLOAD_DELIVERY: "#D55E00",    # vermillion
    Capability.SEARCH_AND_RESCUE: "#56B4E9",   # sky blue
    Capability.STRIKE: "#F0E442",              # yellow
}
ASSIGNED, IDLE, TARGET = "#3d3d3a", "#B4B2A9", "#2C2C2A"

# ---------- icons: closed polygons pointing along +x, roughly unit size ----------

_QUAD = Path([(0.7, 0.15), (0.15, 0.15), (0.15, 0.7), (-0.15, 0.7), (-0.15, 0.15),
              (-0.7, 0.15), (-0.7, -0.15), (-0.15, -0.15), (-0.15, -0.7), (0.15, -0.7),
              (0.15, -0.15), (0.7, -0.15), (0.7, 0.15)], closed=True)


def _plane(span: float, length: float) -> Path:
    """Generic fixed-wing silhouette; longer span/length for larger groups."""
    s, l = span / 2, length / 2
    return Path([(l, 0.0), (0.25 * l, 0.08), (0.1, s), (-0.15, s), (-0.1 * l, 0.08),
                 (-0.8 * l, 0.06), (-l, 0.35 * s), (-l, -0.35 * s), (-0.8 * l, -0.06),
                 (-0.1 * l, -0.08), (-0.15, -s), (0.1, -s), (0.25 * l, -0.08), (l, 0.0)],
                closed=True)


ICONS = {
    PlatformClass.GROUP_1_MULTIROTOR: (_QUAD, 13),
    PlatformClass.GROUP_1_FIXED_WING: (_plane(1.4, 1.0), 14),
    PlatformClass.GROUP_2: (_plane(1.8, 1.1), 16),
    PlatformClass.GROUP_3: (_plane(2.0, 1.3), 18),
    PlatformClass.GROUP_4: (_plane(2.4, 1.5), 20),
    PlatformClass.GROUP_5: (_plane(2.6, 1.6), 22),
}

PIP_SIZE, PIP_STEP = 5, 5.5          # points
SLOT_SIZE, SLOT_STEP = 9, 11         # points
SLOT_DROP = -15                      # points below the target marker


def _marker(platform: PlatformClass, heading_deg: float) -> MarkerStyle:
    path, _ = ICONS[platform]
    return MarkerStyle(path, transform=Affine2D().rotate_deg(heading_deg))


def _stripe(i: int, n: int) -> Path:
    """i-th of n vertical stripes of a unit square centred on the origin."""
    x0, x1 = -0.5 + i / n, -0.5 + (i + 1) / n
    return Path([(x0, -0.5), (x1, -0.5), (x1, 0.5), (x0, 0.5), (x0, -0.5)], closed=True)


def _sorted_caps(caps) -> list[Capability]:
    return sorted(caps, key=lambda c: c.value)


def animateMission(result: AuctionResult, agents: list[Agent], mission: MissionData,
                   every: int = 5, interval_ms: int = 40, trail_s: float = 8.0,
                   title: str = "") -> FuncAnimation:
    """
    every: draw every n-th recorded snapshot (dt=0.02, every=5 -> 10 frames per sim second)
    interval_ms: delay between frames on screen (40 ms -> 25 fps)
    trail_s: seconds of path drawn behind each drone
    """
    if result.state_history is None:
        raise ValueError("result has no state_history; run it with runMission")
    history = result.state_history
    ids = [a.config.agent_id for a in agents]
    platform = {a.config.agent_id: a.config.platform for a in agents}
    capabilities = {a.config.agent_id: _sorted_caps(a.config.capabilities) for a in agents}
    n_snaps = len(history[ids[0]])
    frames = list(range(0, n_snaps, every)) + [n_snaps - 1]
    snap_dt = history[ids[0]][1].time - history[ids[0]][0].time if n_snaps > 1 else 1.0
    trail_n = max(1, round(trail_s / snap_dt)) if snap_dt > 0 else 1

    xy = {i: np.array([s.state.estimate.position[:2] for s in history[i]]) for i in ids}
    vel = {i: np.array([s.state.estimate.velocity[:2] for s in history[i]]) for i in ids}
    heading = {}
    for i in ids:                                    # hold the last heading while ~stationary
        h, last = np.zeros(n_snaps), 90.0
        for k, (vx, vy) in enumerate(vel[i]):
            if math.hypot(vx, vy) > 0.3:
                last = math.degrees(math.atan2(vy, vx))
            h[k] = last
        heading[i] = h

    targets = list(mission.targets_by_id.values())
    slots_by_target: dict[int, list] = {t.target_id: [] for t in targets}
    for slots in mission.slots_by_role.values():
        for s in slots:
            slots_by_target[s.target_id].append(s)
    for tid in slots_by_target:
        slots_by_target[tid].sort(key=lambda s: (sorted(c.value for c in s.required), s.rank))

    tx = np.array([t.target_position.position[0] for t in targets])
    ty = np.array([t.target_position.position[1] for t in targets])
    all_xy = np.vstack([*xy.values(), np.column_stack([tx, ty])])
    lo, hi = all_xy.min(axis=0), all_xy.max(axis=0)
    pad = 0.1 * float(max(hi - lo)) + 30

    fig, ax = plt.subplots(figsize=(11.5, 9))
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)

    def at_offset(dx_pt: float, dy_pt: float):
        return offset_copy(ax.transData, fig=fig, x=dx_pt, y=dy_pt, units="points")

    # ---- targets: marker, id, slot squares, held/total count ----
    ax.scatter(tx, ty, marker="X", s=120, color=TARGET, zorder=3)
    slot_artists: dict = {}                          # slot -> list of stripe Line2Ds
    counts = {}
    for t in targets:
        x, y = t.target_position.position[:2]
        ax.annotate(f"T{t.target_id}", (x, y), textcoords="offset points",
                    xytext=(8, 6), color=TARGET, fontsize=9)
        row = slots_by_target[t.target_id]
        for j, s in enumerate(row):
            dx = (j - (len(row) - 1) / 2) * SLOT_STEP
            caps = _sorted_caps(s.required)
            if not any(s.required <= a.config.capabilities for a in agents):
                ax.plot([x], [y], ls="none", marker="s", ms=SLOT_SIZE, mfc="none",
                        mec=IDLE, mew=1.0, transform=at_offset(dx, SLOT_DROP), zorder=5)
                ax.plot([x], [y], ls="none", marker="x", ms=SLOT_SIZE * 0.7, color=IDLE,
                        mew=1.0, transform=at_offset(dx, SLOT_DROP), zorder=5)
                for c_i, c in enumerate(caps):     # still show what it would need, small
                    ax.plot([x], [y], ls="none", marker="o", ms=3.5,
                            color=CAPABILITY_COLOURS[c],
                            transform=at_offset(dx + (c_i - (len(caps) - 1) / 2) * 4,
                                                SLOT_DROP - 9), zorder=5)
                continue
            stripes = []
            for c_i, c in enumerate(caps):
                ln, = ax.plot([x], [y], ls="none", marker=_stripe(c_i, len(caps)),
                              ms=SLOT_SIZE, mec=CAPABILITY_COLOURS[c], mew=1.2,
                              mfc="white", transform=at_offset(dx, SLOT_DROP), zorder=5)
                stripes.append((ln, CAPABILITY_COLOURS[c]))
            slot_artists[s] = stripes
        counts[t.target_id] = ax.text(x, y, "", transform=at_offset(
            (len(row) + 1) / 2 * SLOT_STEP + 2, SLOT_DROP - 4), fontsize=8, color=TARGET)

    # ---- drones: trail, link, icon, capability pips ----
    trails, links, icons, pips = {}, {}, {}, {}
    for i in ids:
        trails[i], = ax.plot([], [], lw=1, alpha=0.3, color=IDLE)
        links[i], = ax.plot([], [], lw=0.8, ls="--", alpha=0.55, color=ASSIGNED)
        icons[i], = ax.plot([], [], ls="none", ms=ICONS[platform[i]][1], zorder=6,
                            mec="black", mew=0.4)
        pips[i] = [ax.plot([], [], ls="none", marker="o", ms=PIP_SIZE,
                           color=CAPABILITY_COLOURS[c], mec="white", mew=0.5, zorder=7,
                           transform=at_offset(9 + n * PIP_STEP, 8))[0]
                   for n, c in enumerate(capabilities[i])]
    clock = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    # ---- legend: capabilities present, plus state ----
    present = _sorted_caps({c for i in ids for c in capabilities[i]} |
                           {c for r in mission.slots_by_role for c in r})
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=CAPABILITY_COLOURS[c],
                      label=c.name.replace("_", " ").lower()) for c in present]
    handles += [Line2D([], [], ls="none", marker="s", ms=8, mfc=IDLE, mec=IDLE,
                       label="slot held (solid) / open (hollow)"),
                Line2D([], [], ls="none", marker=_stripe(0, 2), ms=9, mfc="white",
                       mec=CAPABILITY_COLOURS[Capability.COMMS_RELAY], mew=1.2,
                       label="split slot: one drone, all colours"),
                Line2D([], [], ls="none", marker="x", ms=7, color=IDLE,
                       label="no drone in swarm can fill"),
                Line2D([], [], ls="none", marker=_marker(PlatformClass.GROUP_1_FIXED_WING, 0),
                       ms=12, color=ASSIGNED, label="assigned drone"),
                Line2D([], [], ls="none", marker=_marker(PlatformClass.GROUP_1_FIXED_WING, 0),
                       ms=12, color=IDLE, alpha=0.6, label="idle drone")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.76, top=0.93, bottom=0.07)

    def draw(k):
        held = set()
        for i in ids:
            snap = history[i][k]
            x, y = xy[i][k]
            assigned = snap.slot is not None
            if assigned:
                held.add(snap.slot)
            icons[i].set_data([x], [y])
            icons[i].set_marker(_marker(platform[i], heading[i][k]))
            icons[i].set_color(ASSIGNED if assigned else IDLE)
            icons[i].set_markeredgecolor("black")
            icons[i].set_alpha(1.0 if assigned else 0.6)
            for p in pips[i]:
                p.set_data([x], [y])
            seg = xy[i][max(0, k - trail_n):k + 1]
            trails[i].set_data(seg[:, 0], seg[:, 1])
            if assigned:
                gx, gy = mission.slotPosition(snap.slot)[:2]
                links[i].set_data([x, gx], [y, gy])
            else:
                links[i].set_data([], [])
        for s, stripes in slot_artists.items():
            for ln, colour in stripes:
                ln.set_markerfacecolor(colour if s in held else "white")
        for t in targets:
            row = slots_by_target[t.target_id]
            counts[t.target_id].set_text(f"{sum(s in held for s in row)}/{len(row)}")
        n_assigned = sum(history[i][k].slot is not None for i in ids)
        clock.set_text(f"t = {history[ids[0]][k].time:5.1f} s    "
                       f"drones assigned {n_assigned}/{len(ids)}    slots held {len(held)}/{len(slot_artists)}")
        return []

    return FuncAnimation(fig, draw, frames=frames, interval=interval_ms, blit=False)


def saveAnimation(anim: FuncAnimation, path: str, fps: int = 25):
    """Save as .mp4 (needs ffmpeg installed) or .gif (uses Pillow)."""
    writer = "pillow" if path.endswith(".gif") else "ffmpeg"
    anim.save(path, writer=writer, fps=fps, dpi=110)
