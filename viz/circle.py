"""
Experiment with drawing circles
"""

# <codecell>

import matplotlib.pyplot as plt
import numpy as np

def rot_mat(angle):
    return np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]
    )

def make_traj(rot, inc, n_runs=10):
    point = np.zeros(2).reshape(-1, 1)
    traj = [point]
    traj_orig = [point]

    for _ in range(n_runs):
        # point = np.tanh(rot @ point + inc)
        # point = rot @ point + inc
        point = rot @ point
        traj.append(point[:])
        point_old = point[:]
        point = point + inc
        traj.append(point[:])
        point = np.tanh(point)
        traj.append(point[:])
        traj_orig.append(point[:])

    return traj, traj_orig

def estimate_rot(rot):
    circ = _draw_sphere(1000, 2)
    rotated_circ = circ @ rot.T
    cos_ang = np.diag(circ @ rotated_circ.T) / np.linalg.norm(rotated_circ, axis=-1)
    ang = np.arccos(cos_ang)
    return ang

def _draw_sphere(n, dims):
    samp = np.random.randn(n, dims)
    norms = np.linalg.norm(samp, axis=-1).reshape(-1, 1)
    norms = np.repeat(norms, dims, axis=1)
    return samp / norms


# <codecell>
# rot = rot_mat(np.pi / 8)
rot = 1 * np.random.randn(2, 2)

inc = np.array([0, -0.1]).reshape(-1, 1)

plt.gcf().set_size_inches(6, 6)
traj, traj_orig = make_traj(rot, inc, n_runs=100)
traj = np.concatenate(traj, axis=-1)
traj_orig = np.concatenate(traj_orig, axis=-1)
plt.plot(traj[0], traj[1], '--o')
plt.plot(traj_orig[0], traj_orig[1], '--o', alpha=0.5)

vals, vecs = np.linalg.eig(rot)

print('VALS', vals)
print('VECS', vecs)

vecs[:,0] *= vals[0]
vecs[:,1] *= vals[1]

plt.arrow(0, 0, vecs[0,0], vecs[1,0])
plt.arrow(0, 0, vecs[0,1], vecs[1,1])


# %%
angs = estimate_rot(np.random.randn(2, 2))
plt.hist(angs)

# %%
