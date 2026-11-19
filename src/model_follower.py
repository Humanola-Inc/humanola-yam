from __future__ import annotations

import argparse
import math
import threading
from typing import Callable

import numpy as np
import numpy.typing as npt
from humanola import constants, dataset, robo
from native import Arm, YamNotSim, YamSim

# Shutdown position (6 joints + gripper) both arms move to on "reset" and on exit.
REST_JOINTS = np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180


def arm_fields(prefix: str) -> list[dataset.Field]:
    return [
        *[
            dataset.Field(
                name=f"{prefix}.joint{i + 1}",
                dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
            )
            for i in range(6)
        ],
        dataset.Field(
            name=f"{prefix}.joint7",
            dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
        ),
        dataset.Field(
            name=f"{prefix}.joint8",
            dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
        ),
        dataset.Field(
            name=f"{prefix}.gripper",
            dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
        ),
    ]


def joints_frame(
    left_joints: npt.NDArray[np.float64], right_joints: npt.NDArray[np.float64]
) -> robo.DataFrame:
    frame = robo.DataFrame()
    for joints in [left_joints, right_joints]:
        for joint in joints[:6]:
            frame.attach(dataset.Num(joint))
        frame.attach(dataset.Num(joints[6] * -0.04695))
        frame.attach(dataset.Num(joints[6] * -0.04695))
        frame.attach(dataset.Num(joints[6]))
    return frame


class PolicyCtrl:
    """Applies joint targets and commands ("cmd.reset", "cmd.restart") sent on ctl/yam-policy."""

    def __init__(self, left_arm: Arm, right_arm: Arm) -> None:
        self.arms: dict[str, Arm] = {"left_arm": left_arm, "right_arm": right_arm}
        self.queries: dict[str, list[robo.Query]] = {
            prefix: [
                robo.Query(kind=robo.SensorKind.RawDouble, name=f"{prefix}.{joint}")
                for joint in [f"joint{i + 1}" for i in range(6)] + ["gripper"]
            ]
            for prefix in self.arms
        }
        self.commands: dict[str, Callable[[], None]] = {
            "reset": self.reset,
            "restart": self.restart,
        }
        self.busy = threading.Event()
        self.last_tp_us = 0

    def open(self) -> PolicyCtrl:
        return self

    def recv_delta(self, prev: robo.Device, cur: robo.Device) -> None:
        # Handle each message once, so a re-delivered old message can't repeat a command or undo a reset.
        if cur.tp_us <= self.last_tp_us:
            return
        self.last_tp_us = cur.tp_us
        # Drop everything that arrives while a command is still running.
        if self.busy.is_set():
            return

        for name, command in self.commands.items():
            query = robo.Query(kind=robo.SensorKind.RawDouble, name=f"cmd.{name}")
            if cur.get(query) is not None:
                # Run off the subscriber thread; busy is set here so no action slips in first.
                self.busy.set()
                threading.Thread(target=self.run_command, args=(name, command)).start()
                return

        targets: dict[str, npt.NDArray[np.float64]] = {}
        for prefix in self.arms:
            sensors = [cur.get(q) for q in self.queries[prefix]]
            if any(s is None for s in sensors):
                return
            targets[prefix] = np.array([s.as_raw().value for s in sensors])
        for prefix, joints in targets.items():
            self.arms[prefix].update(joints)

    def run_command(self, name: str, command: Callable[[], None]) -> None:
        print(f"{name}: both arms...")
        try:
            command()
            print(f"{name} done")
        except Exception as e:
            print(f"{name} failed: {e!r}")
        finally:
            self.busy.clear()

    def on_both_arms(self, fn: Callable[[Arm], None]) -> None:
        """Run fn on both arms in parallel and wait for both, re-raising the first error."""
        errors: list[BaseException] = []

        def run(arm: Arm) -> None:
            try:
                fn(arm)
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run, args=(arm,)) for arm in self.arms.values()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    def reset(self, duration: float = 1.0) -> None:
        """Move both arms to the shutdown position, blocking until they arrive."""
        self.on_both_arms(lambda arm: arm.update_over_time(REST_JOINTS, duration))

    def restart(self) -> None:
        """Bring both arms' CAN links down and up and reconnect to them."""
        self.on_both_arms(lambda arm: arm.restart())

    def close_stream(self) -> None:
        pass


class ArmData:
    def __init__(self, left_arm: Arm, right_arm: Arm) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm

    def open(self) -> ArmData:
        return self

    def get_data(self) -> robo.DataFrame:
        return joints_frame(self.left_arm.get_joints(), self.right_arm.get_joints())

    def close_stream(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    left_arm: Arm = YamSim("yam-can-left") if args.sim else YamNotSim("yam-can-left")
    right_arm: Arm = YamSim("yam-can-right") if args.sim else YamNotSim("yam-can-right")
    left_arm.start()
    right_arm.start()
    ctrl = PolicyCtrl(left_arm, right_arm)

    channel, runtime = (
        robo.RoboConfig(
            name="Follower",
            api_url="https://grpc.humanola.com",
            api_key="<YOUR_API_KEY>",
            robo_id="<YOUR_ROBO_ID>",
            min_log_level=10,
        )
        .attach_usb_camera(
            topic="cam/ego",
            id=0,
            width=2560,
            height=720,
            rate=30,
            kind=robo.ImageKind.Stereo,
            name="Egocentric Camera",
        )
        .attach_usb_camera(
            topic="cam/right",
            id=6,
            width=1280,
            height=720,
            rate=30,
            kind=robo.ImageKind.Stereo,
            name="Right Arm",
        )
        .attach_usb_camera(
            topic="cam/left",
            id=12,
            width=1280,
            height=720,
            rate=30,
            kind=robo.ImageKind.Mono,
            name="Left Arm",
        )
        .attach_data(
            topic=constants.SRC_DATA,
            name="Yam Arms",
            desc="Records the joint position of the yam arms",
            rate=30,
            v=ArmData(left_arm, right_arm),
            fields=arm_fields("left_arm") + arm_fields("right_arm"),
        )
        .attach_device_subscriber(
            topic="ctl/yam-policy",
            name="Yam policy follower",
            desc="Moves the yam arms to the SmolVLA actions",
            rate=30,
            v=ctrl,
        )
        .run()
    )
    try:
        runtime.wait_for_interrupt()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.reset()
        for arm in [left_arm, right_arm]:
            arm.close()
