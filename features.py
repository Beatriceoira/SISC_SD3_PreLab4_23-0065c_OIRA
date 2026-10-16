"""Feature engineering: normalization, distances, angles, velocity, acceleration.

Inputs are raw landmarks `raw` (T, 2, 21, 3) and presence mask `mask` (T, 2); slot 0 = Left, 1 = Right.

Formulas (all implemented below, demo with:  python features.py --demo)
    palm scale      s = || P9 - P0 ||_xy            (wrist -> middle-finger MCP)
    normalization   p' = (p - P0) / s               (wrist at origin, palm length = 1)
    distance        d = sqrt((x2-x1)^2 + (y2-y1)^2)                    (xy plane, normalized coords)
    joint angle     theta = arccos( ((a-b).(c-b)) / (||a-b|| ||c-b||) ) (3D, at vertex b)
    velocity        v_t = (p_t - p_{t-1}) / dt          v_0 = 0
    acceleration    a_t = (v_t - v_{t-1}) / dt          a_0 = a_1 = 0
Missing hands: coordinates/features are zero and the mask is part of the feature vector;
velocity/acceleration are zero whenever any of the frames they depend on is missing.

Per-frame feature vector (FRAME_DIM = 151):
    126  normalized coordinates (2 hands x 21 x 3)
     10  distances: 5 per hand (thumb-index, index-middle, thumb-pinky, wrist-middle tip, index-pinky)
     10  angles:    5 per hand (thumb IP, index/middle/ring/pinky PIP), divided by pi
      3  inter-hand: wrist-wrist distance, dx, dy (in palm units; 0 unless both hands present)
      2  presence mask (Left, Right)
Sequence vector (SEQ_DIM = 205) = frame features + velocity(27) + acceleration(27) of the
"base" signals: wrist image position (4) + distances (10) + angles (10) + inter-hand (3).
"""
import argparse

import numpy as np

EPS = 1e-6
WRIST, MIDDLE_MCP = 0, 9
DIST_PAIRS = [("thumb_index", 4, 8), ("index_middle", 8, 12), ("thumb_pinky", 4, 20),
              ("wrist_middle_tip", 0, 12), ("index_pinky", 8, 20)]
ANGLE_TRIPLETS = [("thumb_IP", 2, 3, 4), ("index_PIP", 5, 6, 7), ("middle_PIP", 9, 10, 11),
                  ("ring_PIP", 13, 14, 15), ("pinky_PIP", 17, 18, 19)]
N_BASE = 4 + 10 + 10 + 3
FRAME_DIM = 126 + 10 + 10 + 3 + 2
SEQ_DIM = FRAME_DIM + 2 * N_BASE


# ----------------------------------------------------------------------- primitives
def euclidean(p, q):
    """d = sqrt((x2-x1)^2 + (y2-y1)^2) using the first two coordinates."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.sqrt(((q[..., :2] - p[..., :2]) ** 2).sum(-1))


def joint_angle(a, b, c):
    """theta = arccos(((a-b).(c-b)) / (||a-b|| ||c-b||)), radians, vertex at b. 0 if degenerate."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    v1, v2 = a - b, c - b
    den = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.clip((v1 * v2).sum(-1) / np.maximum(den, EPS), -1.0, 1.0)
    return np.where(den > EPS, np.arccos(cos), 0.0)


def finite_diff(p, valid, dt):
    """Backward difference (p_t - p_{t-1}) / dt, zero where either frame is invalid."""
    out = np.zeros_like(p, dtype=np.float32)
    ok = np.zeros(valid.shape, bool)
    out[1:] = (p[1:] - p[:-1]) / dt
    ok[1:] = valid[1:] & valid[:-1]
    return out * ok, ok


# ----------------------------------------------------------------------- normalization
def palm_scale(raw):
    return euclidean(raw[..., WRIST, :], raw[..., MIDDLE_MCP, :])


def normalize_hands(raw, mask):
    """Wrist-centered, palm-scaled coordinates. -> norm (T,2,21,3), scale (T,2)."""
    scale = palm_scale(raw)
    safe = np.where(mask & (scale > EPS), scale, 1.0)
    norm = (raw - raw[:, :, WRIST:WRIST + 1, :]) / safe[..., None, None]
    return (norm * mask[..., None, None]).astype(np.float32), safe


# ----------------------------------------------------------------------- features
def frame_features(raw, mask):
    """-> feats (T, FRAME_DIM), base (T, N_BASE), base_valid (T, N_BASE)."""
    raw = np.asarray(raw, np.float32)
    mask = np.asarray(mask, bool)
    T = raw.shape[0]
    norm, scale = normalize_hands(raw, mask)
    coords = norm.reshape(T, -1)
    dist = np.stack([euclidean(norm[:, :, i], norm[:, :, j]) for _, i, j in DIST_PAIRS], -1)
    ang = np.stack([joint_angle(norm[:, :, a], norm[:, :, b], norm[:, :, c])
                    for _, a, b, c in ANGLE_TRIPLETS], -1) / np.pi
    dist = (dist * mask[..., None]).astype(np.float32)
    ang = (ang * mask[..., None]).astype(np.float32)
    both = mask.all(axis=1)
    wrist = raw[:, :, WRIST, :]
    dxy = (wrist[:, 1, :2] - wrist[:, 0, :2]) / scale.mean(axis=1)[:, None]
    dd = np.sqrt((dxy ** 2).sum(-1))
    inter = (np.concatenate([dd[:, None], dxy], 1) * both[:, None]).astype(np.float32)
    feats = np.concatenate([coords, dist.reshape(T, -1), ang.reshape(T, -1), inter,
                            mask.astype(np.float32)], 1)
    wrist_xy = (wrist[:, :, :2] * mask[..., None]).reshape(T, 4)
    base = np.concatenate([wrist_xy, dist.reshape(T, -1), ang.reshape(T, -1), inter], 1)
    valid = np.concatenate([np.repeat(mask, 2, axis=1), np.repeat(mask, 5, axis=1),
                            np.repeat(mask, 5, axis=1), np.repeat(both[:, None], 3, axis=1)], 1)
    return feats.astype(np.float32), base.astype(np.float32), valid


def sequence_features(raw, mask, dt):
    """-> (T, SEQ_DIM): frame features + velocity + acceleration of the base signals."""
    feats, base, valid = frame_features(raw, mask)
    v, vok = finite_diff(base, valid, dt)
    a, _ = finite_diff(v, vok, dt)
    return np.concatenate([feats, v, a], 1).astype(np.float32)


def motion_energy(raw, mask, dt):
    """Mean wrist speed (image-height units / s) over valid frames; 0 if undefined."""
    _, base, valid = frame_features(raw, mask)
    v, ok = finite_diff(base[:, :4], valid[:, :4], dt)
    speeds, oks = [], []
    for h in range(2):
        speeds.append(np.sqrt(v[:, 2 * h] ** 2 + v[:, 2 * h + 1] ** 2))
        oks.append(ok[:, 2 * h])
    s, o = np.concatenate(speeds), np.concatenate(oks)
    return float(s[o].mean()) if o.any() else 0.0


# ----------------------------------------------------------------------- worked examples
def demo():
    print("== Worked numeric examples (for Calculations.pdf) ==")
    p1, p2 = np.array([0.30, 0.40]), np.array([0.60, 0.80])
    print(f"Distance: p1={p1}, p2={p2}\n  d = sqrt((0.60-0.30)^2 + (0.80-0.40)^2) = sqrt(0.09+0.16) = {euclidean(p1, p2):.4f}")
    a, b, c = np.array([1., 0, 0]), np.array([0., 0, 0]), np.array([1., 1, 0])
    th = joint_angle(a, b, c)
    print(f"Angle: a={a}, b={b}, c={c}\n  (a-b).(c-b)=1, ||a-b||=1, ||c-b||=sqrt(2)\n"
          f"  theta = arccos(1/sqrt(2)) = {th:.4f} rad = {np.degrees(th):.2f} deg")
    pos = np.array([0.40, 0.46, 0.55]); dt = 1 / 30
    v, ok = finite_diff(pos, np.ones(3, bool), dt)
    acc, _ = finite_diff(v, ok, dt)
    print(f"Velocity/acceleration: p={pos}, dt=1/30 s\n  v1=(0.46-0.40)/dt={v[1]:.2f}, v2=(0.55-0.46)/dt={v[2]:.2f}\n"
          f"  a2=(v2-v1)/dt={acc[2]:.2f}")
    hand = np.zeros((21, 3)); hand[0] = [0.5, 0.8, 0]; hand[9] = [0.5, 0.6, 0]; hand[12] = [0.52, 0.45, 0]
    raw = hand[None, None].repeat(2, 1); m = np.array([[True, False]])
    n, s = normalize_hands(raw, m)
    print(f"Normalization: wrist=(0.5,0.8), middle MCP=(0.5,0.6) -> palm scale s={s[0, 0]:.3f}\n"
          f"  middle tip (0.52,0.45) -> ({n[0, 0, 12, 0]:.3f}, {n[0, 0, 12, 1]:.3f})  [= (0.02/0.2, -0.35/0.2)]")
    print(f"Feature sizes: FRAME_DIM={FRAME_DIM}, SEQ_DIM={SEQ_DIM}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        demo()
