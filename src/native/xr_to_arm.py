from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from humanola import robo

from .controller import ArmController


def quat_pos_to_pose(
    quat: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x, y, z, w = quat / np.linalg.norm(quat)  # normalize to guard against drift
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = pos
    return pose


@dataclass
class XrConfig:
    ef_act: robo.Query
    ef_pos: robo.Query
    ef_rot: robo.Query
    arm_gripper: robo.Query
    arm_reset: robo.Query
    arm_init: robo.Query
    arm_restart: robo.Query
    arm_lr: robo.Query

    @staticmethod
    def left() -> XrConfig:
        return XrConfig(
            ef_act=robo.Query(kind=robo.SensorKind.Btn, name="xr.left.squeeze"),
            ef_pos=robo.Query(kind=robo.SensorKind.Pos, name="xr.left.bod"),
            ef_rot=robo.Query(kind=robo.SensorKind.Rot, name="xr.left.bod"),
            arm_gripper=robo.Query(kind=robo.SensorKind.Btn, name="xr.left.aim"),
            arm_reset=robo.Query(kind=robo.SensorKind.Btn, name="xr.left.secondary"),
            arm_init=robo.Query(kind=robo.SensorKind.Btn, name="xr.left.primary"),
            arm_restart=robo.Query(kind=robo.SensorKind.Btn, name="xr.left.joy"),
            arm_lr=robo.Query(kind=robo.SensorKind.Joy, name="xr.left.joy"),
        )

    @staticmethod
    def right() -> XrConfig:
        return XrConfig(
            ef_act=robo.Query(kind=robo.SensorKind.Btn, name="xr.right.squeeze"),
            ef_pos=robo.Query(kind=robo.SensorKind.Pos, name="xr.right.bod"),
            ef_rot=robo.Query(kind=robo.SensorKind.Rot, name="xr.right.bod"),
            arm_gripper=robo.Query(kind=robo.SensorKind.Btn, name="xr.right.aim"),
            arm_reset=robo.Query(kind=robo.SensorKind.Btn, name="xr.right.secondary"),
            arm_init=robo.Query(kind=robo.SensorKind.Btn, name="xr.right.primary"),
            arm_restart=robo.Query(kind=robo.SensorKind.Btn, name="xr.right.joy"),
            arm_lr=robo.Query(kind=robo.SensorKind.Joy, name="xr.right.joy"),
        )


class Xr2Arm:
    XR_POSE_TRANSFORM = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )

    def __init__(
        self,
        arm: ArmController,
        config: XrConfig,
        init_joints: npt.NDArray[np.float64] | None = None,
    ):
        self.arm = arm
        self.config = config
        if init_joints is None:
            self.init_joints = np.array([0, 45, 90, -45, 0, 0, 0]) * math.pi / 180

    def reset_init(self):
        self.arm.update_over_time(self.init_joints, 1)

    def reset_flat(self):
        self.arm.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)

    def open(self):
        self.reset_init()

    def close_stream(self):
        self.reset_flat()

    def recv_delta(
        self, prev: robo.Device, cur: robo.Device
    ) -> npt.NDArray[np.float64] | None:
        # position and rotation
        cur_act_btn = cur.get(self.config.ef_act)
        if cur_act_btn is not None and cur_act_btn.as_btn().value > 0.5:
            prev_position = prev.get(self.config.ef_pos)
            prev_rotation = prev.get(self.config.ef_rot)
            cur_position = cur.get(self.config.ef_pos)
            cur_rotation = cur.get(self.config.ef_rot)
            if (
                prev_position is not None
                and prev_rotation is not None
                and cur_position is not None
                and cur_rotation is not None
            ):
                prev_position = prev_position.as_pos()
                prev_rotation = prev_rotation.as_rot()
                cur_position = cur_position.as_pos()
                cur_rotation = cur_rotation.as_rot()
                prev_rotation = np.array(
                    [prev_rotation.x, prev_rotation.y, prev_rotation.z, prev_rotation.w]
                )
                prev_position = np.array(
                    [prev_position.x, prev_position.y, prev_position.z]
                )
                cur_rotation = np.array(
                    [cur_rotation.x, cur_rotation.y, cur_rotation.z, cur_rotation.w]
                )
                cur_position = np.array(
                    [cur_position.x, cur_position.y, cur_position.z]
                )
                prev_pose = (
                    quat_pos_to_pose(prev_rotation, prev_position)
                    @ Xr2Arm.XR_POSE_TRANSFORM
                )
                cur_pose = (
                    quat_pos_to_pose(cur_rotation, cur_position)
                    @ Xr2Arm.XR_POSE_TRANSFORM
                )
                delta_pose = np.linalg.inv(prev_pose) @ cur_pose
                self.arm.state.set_with_ef_delta(delta_pose)
        # gripper pressed
        cur_gripper_btn = cur.get(self.config.arm_gripper)
        if cur_gripper_btn is not None and cur_gripper_btn.as_btn().pressed:
            self.arm.state.open_gripper()
        else:
            self.arm.state.close_gripper()
        prev_reset_btn = prev.get(self.config.arm_reset)
        cur_reset_btn = cur.get(self.config.arm_reset)
        # reset
        if (
            prev_reset_btn is not None
            and cur_reset_btn is not None
            and prev_reset_btn.as_btn().pressed
            and not cur_reset_btn.as_btn().pressed
        ):
            self.reset_init()
            return
        prev_init_btn = prev.get(self.config.arm_init)
        cur_init_btn = cur.get(self.config.arm_init)
        # reset
        if (
            prev_init_btn is not None
            and cur_init_btn is not None
            and prev_init_btn.as_btn().pressed
            and not cur_init_btn.as_btn().pressed
        ):
            self.reset_flat()
            return
        prev_restart_btn = prev.get(self.config.arm_restart)
        cur_restart_btn = cur.get(self.config.arm_restart)
        if (
            prev_restart_btn is not None
            and cur_restart_btn is not None
            and prev_restart_btn.as_btn().pressed
            and not cur_restart_btn.as_btn().pressed
        ):
            self.arm.restart()
            return
        cur_lr = cur.get(self.config.arm_lr)
        if cur_lr is not None:
            cur_x_joy = cur_lr.as_joy().x
            if abs(cur_x_joy) > 0.2:
                joints = self.arm.state.joints
                if cur_x_joy > 0:
                    joints[0] += 1 / 180 * math.pi
                else:
                    joints[0] -= 1 / 180 * math.pi
                self.arm.state.set_with_joints(joints)
        self.arm.update()
