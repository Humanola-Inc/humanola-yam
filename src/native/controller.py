import math
import multiprocessing as mp
import subprocess
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt

from .state import ArmState


class Arm(Protocol):
    def __init__(self, channel: str):
        pass

    def start(self):
        pass

    def restart(self):
        pass

    def update(self, joints: npt.NDArray[np.float64]):
        pass

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        pass

    def get_joints(self) -> npt.NDArray[np.float64]:
        pass

    def close(self):
        pass


class YamSim:
    def __init__(self, channel: str):
        self.channel = channel
        self.joints = np.zeros(7)

    def start(self):
        pass

    def restart(self):
        pass

    def update(self, joints: npt.NDArray[np.float64]):
        self.joints = joints

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        start_time = 0
        delta_joints = (target_joints - self.joints) / duration
        while start_time < duration:
            self.joints = self.joints + delta_joints * 0.1
            start_time += 0.1
            time.sleep(0.1)
        self.joints = target_joints

    def get_joints(self) -> npt.NDArray[np.float64]:
        return self.joints

    def close(self):
        pass


class YamNotSim:
    def __init__(self, channel: str):
        self.channel = channel

    def start(self):
        # imported here so --sim runs without i2rt installed
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.arm = get_yam_robot(
            channel=self.channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=True,
            sim=False,
            enable_auto_recovery=True,
        )

    def restart(self):
        subprocess.run(["ip", "link", "set", self.channel, "down"], check=True)
        subprocess.run(
            [
                "ip",
                "link",
                "set",
                self.channel,
                "up",
                "type",
                "can",
                "bitrate",
                "1000000",
            ],
            check=True,
        )
        self.start()

    def update(self, joints: npt.NDArray[np.float64]):
        self.arm.command_joint_pos(joints)

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        self.arm.move_joints(target_joints, duration)

    def get_joints(self) -> npt.NDArray[np.float64]:
        joints = np.zeros(7)
        obs = self.arm.get_observations()
        joints[:6] = obs["joint_pos"]
        joints[6] = obs["gripper_pos"]
        return joints

    def close(self):
        self.arm.close()


class YamLeaderNotSim(YamNotSim):
    def start(self):
        # imported here so --sim runs without i2rt installed
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.arm = get_yam_robot(
            channel=self.channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
            sim=False,
            enable_auto_recovery=True,
        )

    def get_joints(self) -> npt.NDArray[np.float64]:
        joints = np.zeros(7)
        joints[:6] = self.arm.get_observations()["joint_pos"]
        # the teaching handle trigger is an encoder on the same bus, empty
        # until the first CAN read lands
        encoders = self.arm.motor_chain.get_same_bus_device_states()
        if encoders:
            joints[6] = 1 - encoders[0].position
        return joints


@dataclass
class JointCtrl:
    joints: npt.NDArray[np.float64]
    duration: Optional[float]


class GetJoint:
    pass


class Close:
    pass


def offshore_controller(
    channel: str,
    sim: bool,
    restart: bool,
    is_ready: Event,
    ctrl_rx: Connection,
    joints_rx: Connection,
) -> None:
    arm: Arm = YamSim(channel) if sim else YamNotSim(channel)
    if restart:
        arm.restart()
    else:
        arm.start()

    def joints_loop():
        while True:
            ev = joints_rx.recv()
            if isinstance(ev, Close):
                break
            elif isinstance(ev, GetJoint):
                joints_rx.send(arm.get_joints())

    def ctrl_loop():
        while True:
            ev = ctrl_rx.recv()
            if isinstance(ev, Close):
                break
            elif isinstance(ev, JointCtrl):
                if ev.duration is None:
                    arm.update(ev.joints)
                else:
                    arm.update_over_time(ev.joints, ev.duration)

    joints_thread = threading.Thread(target=joints_loop, daemon=True)
    ctrl_thread = threading.Thread(target=ctrl_loop, daemon=True)
    joints_thread.start()
    ctrl_thread.start()
    is_ready.set()
    try:
        joints_thread.join()
        ctrl_thread.join()
    finally:
        arm.close()


class ArmController:
    def __init__(
        self,
        channel: str,
        state: ArmState,
        init_state: bool = True,
        sim: bool = False,
    ):
        self.channel = channel
        self.sim = sim
        self.state = state
        self.__start_process()
        if init_state:
            joints = self.get_joints()
            self.state.set_with_joints(joints)
        else:
            self.update()

    def __start_process(self, restart: bool = False):
        self.is_ready = mp.Event()
        ctrl_rx, self.ctrl_tx = mp.Pipe(duplex=False)
        self.joints_tx, joints_rx = mp.Pipe()
        self.process = mp.Process(
            target=offshore_controller,
            args=(
                self.channel,
                self.sim,
                restart,
                self.is_ready,
                ctrl_rx,
                joints_rx,
            ),
        )
        self.process.start()
        self.is_ready.wait()

    def update(self):
        self.ctrl_tx.send(JointCtrl(self.state.joints, None))

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        self.state.set_with_joints(target_joints)
        self.ctrl_tx.send(JointCtrl(target_joints, duration))

    def get_joints(self) -> npt.NDArray[np.float64]:
        self.joints_tx.send(GetJoint())
        return self.joints_tx.recv()

    def restart(self):
        self.close()
        self.__start_process(restart=True)

    def close(self):
        self.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)
        self.joints_tx.send(Close())
        self.ctrl_tx.send(Close())
        self.process.join()
