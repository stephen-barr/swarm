import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# get robot 5 odometry
DATA = Path.cwd() / "data" / "MRCLAM_Dataset9"
odom = np.loadtxt(DATA / "Robot5_Odometry.dat", comments="#")

# get robot 5 ground truth
gt = np.loadtxt(DATA / "Robot5_Groundtruth.dat", comments="#")

t = odom[:, 0] - odom[0, 0]     # seconds from start
v = odom[:, 1]                  # forward velocity, m/s
w = odom[:, 2]                  # angular velocity, rad/s

x = np.zeros(len(t))
y = np.zeros(len(t))
th = np.zeros(len(t))
th[0] = gt[0, 3]                # set first theta to global position
for k in range(len(t)-1):
    dt = t[k+1] - t[k]
    x[k+1] = x[k]+v[k]*dt*np.cos(th[k])
    y[k+1] = y[k]+v[k]*dt*np.sin(th[k])
    th[k+1] = th[k]+dt*w[k]

tg = gt[:, 0] - odom[0, 0]
xg, yg = gt[:, 1], gt[:, 2]

thg = np.interp(t, tg, np.unwrap(gt[:, 3]))
err = np.arctan2(np.sin(th - thg), np.cos(th - thg))

xgi = np.interp(t, tg, xg)
ygi = np.interp(t, tg, yg)
pos_err = np.hypot(x + xg[0] - xgi, y + yg[0] - ygi)

# heading error
d = np.degrees(err)
d[np.abs(np.diff(d, prepend=d[0])) > 180] = np.nan

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, d, lw=0.8)
ax.set_xlabel("time [s]")
ax.set_ylabel("heading error [deg]")
ax.set_title("Robot 5 — odometry heading drift")
fig.tight_layout()
fig.savefig("heading_error.png", dpi=150)

# position error
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, pos_err, lw=0.8)
ax.set_xlabel("time [s]")
ax.set_ylabel("position error [m]")
ax.set_title("Robot 5 — odometry position drift")
fig.tight_layout()
fig.savefig("position_error.png", dpi=150)
'''
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(xg, yg, lw=1, color="0.6", label="ground truth")
ax.plot(x + xg[0], y + yg[0], lw=1, color="C3", label="dead reckoned")
ax.plot(xg[0], yg[0], "ko", ms=5, label="start")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Robot 5, dataset 9 — odometry only")
ax.legend()
fig.tight_layout()
plt.show()
'''