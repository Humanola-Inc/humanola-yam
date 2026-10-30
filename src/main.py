import argparse
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from humanola import constants, dataset, robo
from native import ArmController, ArmState, MujocoSolver, Xr2Arm, XrConfig

RUNTIME = None

ARM_FIELDS = [
    dataset.Field(
        name="left_arm.joint1",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint2",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint3",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint4",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint5",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint6",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint7",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.joint8",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="left_arm.gripper",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint1",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint2",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint3",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint4",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint5",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint6",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint7",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.joint8",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="right_arm.gripper",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
]


def joints_frame(left_joints, right_joints) -> robo.DataFrame:
    frame = robo.DataFrame()
    for joint in left_joints[:6]:
        frame.attach(dataset.Num(joint))
    left_gripper = left_joints[6]
    frame.attach(dataset.Num(left_gripper * -0.04695))
    frame.attach(dataset.Num(left_gripper * -0.04695))
    frame.attach(dataset.Num(left_gripper))
    for joint in right_joints[:6]:
        frame.attach(dataset.Num(joint))
    right_gripper = right_joints[6]
    frame.attach(dataset.Num(right_gripper * -0.04695))
    frame.attach(dataset.Num(right_gripper * -0.04695))
    frame.attach(dataset.Num(right_gripper))
    return frame


class XrAction:
    def __init__(self):
        self.stream = None

    def open(self):
        self.stream = RUNTIME.get_hub().open_data(
            topic="act/arms",
            name="Yam Arms Action",
            desc="Records the joint targets sent to yam arms",
            to=[robo.Addr.Dataset],
            fields=ARM_FIELDS,
        )
        return self

    def record(self, left_joints, right_joints):
        self.stream.capture(joints_frame(left_joints, right_joints).update_tp_us())

    def close_stream(self):
        self.stream.close()
        self.stream = None


class XrCtrl:
    def __init__(self, left_arm: Xr2Arm, right_arm: Xr2Arm):
        self.arms = [left_arm, right_arm]
        self.action = XrAction()

    def open(self):
        self.action.open()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pool.map(Xr2Arm.open, self.arms)
        return self

    def recv_delta(self, prev: robo.Device, cur: robo.Device):
        left_joints, right_joints = self.pool.map(
            Xr2Arm.recv_delta,
            self.arms,
            [prev] * len(self.arms),
            [cur] * len(self.arms),
        )
        self.action.record(left_joints, right_joints)

    def close_stream(self):
        self.pool.map(Xr2Arm.close_stream, self.arms)
        self.pool.shutdown()
        self.action.close_stream()


class ArmData:
    def __init__(self, left_arm: ArmController, right_arm: ArmController):
        self.arms = [left_arm, right_arm]

    def open(self):
        self.pool = ThreadPoolExecutor(max_workers=2)
        return self

    def get_data(self):
        left_joints, right_joints = self.pool.map(ArmController.get_joints, self.arms)
        return joints_frame(left_joints, right_joints)

    def close_stream(self):
        self.pool.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()
    xml_path = (
        pathlib.Path(__file__).parent.parent
        / "libs/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"
    )

    left_arm = ArmController(
        "can1",
        state=ArmState(
            joints=np.zeros(7),
            solver=MujocoSolver(xml_path),
        ),
        sim=args.sim,
    )
    right_arm = ArmController(
        "can0",
        state=ArmState(
            joints=np.zeros(7),
            solver=MujocoSolver(xml_path),
        ),
        sim=args.sim,
    )

    channel, runtime = (
        robo.RoboConfig(
            api_url="https://grpc.humanola.com",
            api_key="<YOUR_API_KEY>",
            robo_id="<YOUR_ROBO_ID>",
            name="Yam",
        )
        .attach_device_subscriber(
            topic=constants.DEV_XR_CONTROLLER_TOPIC,
            name="xr",
            desc="",
            rate=60,
            v=XrCtrl(
                left_arm=Xr2Arm(left_arm, XrConfig.left()),
                right_arm=Xr2Arm(right_arm, XrConfig.right()),
            ),
        )
        .attach_data(
            topic=constants.SRC_DATA,
            name="Yam Arms",
            desc="",
            rate=120,
            v=ArmData(left_arm=left_arm, right_arm=right_arm),
            fields=ARM_FIELDS,
        )
        .run()
    )
    RUNTIME = runtime
    try:
        runtime.wait_for_interrupt()
    finally:
        l_t = threading.Thread(target=lambda: left_arm.close())
        l_t.start()
        r_t = threading.Thread(target=lambda: right_arm.close())
        r_t.start()
        l_t.join()
        r_t.join()
