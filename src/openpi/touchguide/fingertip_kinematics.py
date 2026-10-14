"""Static, finger-local IK for contact-anchor augmentation (no physics steps).

27D states retain right arm 7 + right hand 20 ordering. Passive thumb joint6
tracks the change of joint5 while retaining the recorded equality residual.
Unselected joints, including the whole arm, stay exactly at their source values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 8)) + tuple(
    f"hand_r_{finger}_joint{i}" for finger in FINGERS for i in ((1, 2, 3, 5) if finger == "thumb" else (1, 2, 3, 4))
)
SITES = tuple(f"hand_r_{f}_link{6 if f == 'thumb' else 4}_site" for f in FINGERS)


def qpos_labels(model):
    labels = []
    for j in range(model.njnt):
        name = model.joint(j).name
        kind = int(model.jnt_type[j])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            labels.extend(f"{name}/{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz"))
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            labels.extend(f"{name}/{c}" for c in ("qw", "qx", "qy", "qz"))
        else:
            labels.append(name)
    if len(labels) != model.nq or len(set(labels)) != model.nq:
        raise ValueError("Model requires unique, complete named qpos coordinates")
    return labels


@dataclass(frozen=True)
class IKConfig:
    position_tolerance_m: float = 5e-5
    max_joint_delta_rad: float = 0.15
    posture_scale_m_per_rad: float = 1e-4
    max_evaluations: int = 80

    def __post_init__(self):
        for value in (self.position_tolerance_m, self.max_joint_delta_rad, self.posture_scale_m_per_rad):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("IK tolerances/scales must be positive and finite")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")


class FingertipKinematics:
    """Keep one independent scratch MjData; never drive simulation actuators."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.labels = qpos_labels(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINTS])
        self.state_qpos = self.model.jnt_qposadr[self.joint_ids].copy()
        self.state_dofs = self.model.jnt_dofadr[self.joint_ids].copy()
        self.site_ids = np.array([self.model.site(n).id for n in SITES])
        self.thumb6 = self.model.joint("hand_r_thumb_joint6").id
        self.thumb6_qpos = int(self.model.jnt_qposadr[self.thumb6])
        self.thumb6_dof = int(self.model.jnt_dofadr[self.thumb6])
        # This reduced Jacobian is valid only for the recorded 1:1 coupling.
        pairs = [
            (int(self.model.eq_obj1id[e]), int(self.model.eq_obj2id[e]), self.model.eq_data[e])
            for e in range(self.model.neq)
            if int(self.model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)
        ]
        thumb5 = self.model.joint("hand_r_thumb_joint5").id
        if not any({a, b} == {thumb5, self.thumb6} and np.allclose(c[:5], [0, 1, 0, 0, 0]) for a, b, c in pairs):
            raise ValueError("Expected thumb joint5/joint6 one-to-one model equality")
        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.jac_p = np.zeros((3, self.model.nv))

    def source_columns(self, names):
        lookup = {n: i for i, n in enumerate(names)}
        if len(lookup) != len(names):
            raise ValueError("Duplicated source qpos names")
        return np.array([lookup[n] for n in self.labels], dtype=np.int64)

    def fk(self, full_qpos):
        q = np.asarray(full_qpos, dtype=np.float64)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError("FK requires a finite, complete model qpos")
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return (self.data.site_xpos[self.site_ids].copy(), self.data.site_xmat[self.site_ids].reshape(5, 3, 3).copy())

    def finger_jacobian(self, full_qpos, finger):
        self.fk(full_qpos)
        self.jac_p.fill(0)
        mujoco.mj_jacSite(self.model, self.data, self.jac_p, None, int(self.site_ids[finger]))
        state_slice = slice(7 + 4 * finger, 11 + 4 * finger)
        jac = self.jac_p[:, self.state_dofs[state_slice]].copy()
        if finger == 0:
            jac[:, 3] += self.jac_p[:, self.thumb6_dof]
        return jac

    def _solve_finger(self, baseline, seed, target, finger, lower, upper, config):
        s = slice(7 + 4 * finger, 11 + 4 * finger)
        addresses = self.state_qpos[s]

        scratch = baseline.copy()
        last = None
        current_pos = None
        current_jac = None

        def evaluate(q):
            nonlocal last, current_pos, current_jac
            if last is None or not np.array_equal(q, last):
                scratch[addresses] = q
                if finger == 0:
                    scratch[self.thumb6_qpos] = baseline[self.thumb6_qpos] + q[3] - seed[3]
                pos, _ = self.fk(scratch)
                current_pos = pos[finger]
                self.jac_p.fill(0)
                mujoco.mj_jacSite(self.model, self.data, self.jac_p, None, int(self.site_ids[finger]))
                current_jac = self.jac_p[:, self.state_dofs[s]].copy()
                if finger == 0:
                    current_jac[:, 3] += self.jac_p[:, self.thumb6_dof]
                last = q.copy()

        def residual(q):
            evaluate(q)
            return np.r_[current_pos - target, config.posture_scale_m_per_rad * (q - seed)]

        def jacobian(q):
            evaluate(q)
            return np.vstack([current_jac, config.posture_scale_m_per_rad * np.eye(4)])

        return least_squares(
            residual,
            np.clip(seed, lower, upper),
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=config.max_evaluations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-12,
        )

    def solve(self, state, full_qpos, displacement_world_m, config=None):
        if config is None:
            config = IKConfig()
        state = np.asarray(state, dtype=np.float64)
        baseline = np.array(full_qpos, dtype=np.float64, copy=True)
        delta = np.asarray(displacement_world_m, dtype=np.float64)
        if state.shape != (27,) or delta.shape != (5, 3) or not np.isfinite(np.r_[state, delta.ravel()]).all():
            raise ValueError("Expected finite state[27] and displacement[5,3]")
        if baseline.shape != (self.model.nq,) or not np.isfinite(baseline).all():
            raise ValueError("Invalid full qpos")
        baseline[self.state_qpos] = state
        original_pos, original_rot = self.fk(baseline)
        targets = original_pos + delta
        mask = np.any(delta != 0, axis=1)
        result = state.copy()
        solved_full = baseline.copy()
        evaluations = np.zeros(5, dtype=np.int64)
        statuses = ["not_perturbed"] * 5
        for finger in np.flatnonzero(mask):
            s = slice(7 + 4 * finger, 11 + 4 * finger)
            addresses = self.state_qpos[s]
            seed = state[s].copy()
            lower = np.maximum(self.lower[s], seed - config.max_joint_delta_rad)
            upper = np.minimum(self.upper[s], seed + config.max_joint_delta_rad)
            if finger == 0:
                # The passive joint must also remain within its model range.
                offset = baseline[self.thumb6_qpos] - seed[3]
                lower[3] = max(lower[3], self.model.jnt_range[self.thumb6, 0] - offset)
                upper[3] = min(upper[3], self.model.jnt_range[self.thumb6, 1] - offset)
            if np.any(lower >= upper):
                statuses[finger] = "empty_joint_bounds"
                continue
            solution = self._solve_finger(baseline, seed, targets[finger], finger, lower, upper, config)
            result[s] = solution.x
            solved_full[addresses] = solution.x
            if finger == 0:
                solved_full[self.thumb6_qpos] = baseline[self.thumb6_qpos] + solution.x[3] - seed[3]
            evaluations[finger] = solution.nfev
            statuses[finger] = f"optimizer_status_{solution.status}"
        actual_pos, actual_rot = self.fk(solved_full)
        errors = np.linalg.norm(actual_pos - targets, axis=1)
        actual_delta = actual_pos - original_pos
        bound_ok = True
        for finger in np.flatnonzero(mask):
            s = slice(7 + 4 * finger, 11 + 4 * finger)
            bound_ok &= bool(np.all(result[s] >= self.lower[s] - 1e-10) and np.all(result[s] <= self.upper[s] + 1e-10))
        if mask[0]:
            bound_ok &= bool(
                self.model.jnt_range[self.thumb6, 0] - 1e-10
                <= solved_full[self.thumb6_qpos]
                <= self.model.jnt_range[self.thumb6, 1] + 1e-10
            )
        success = bool(
            np.all(errors[mask] <= config.position_tolerance_m)
            and bound_ok
            and not any(v == "empty_joint_bounds" for v in statuses)
        )
        trace = np.einsum("fij,fij->f", original_rot, actual_rot)
        angles = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return {
            "state": result,
            "full_qpos": solved_full,
            "perturbed_fingers": mask,
            "target_position_world_m": targets,
            "actual_position_world_m": actual_pos,
            "actual_rotation_world": actual_rot,
            "requested_displacement_world_m": delta.copy(),
            "actual_displacement_world_m": actual_delta,
            "error_m": errors,
            "rotation_change_rad": angles,
            "evaluations": evaluations,
            "status": statuses,
            "joint_bounds_ok": bound_ok,
            "success": success,
        }
