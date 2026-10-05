import argparse
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from humanola import constants, dataset, robo
from native import ArmController, ArmSolver, ArmState, Xr2Arm, XrConfig


class XrCtrl:
    def __init__(self, left_arm: Xr2Arm, right_arm: Xr2Arm):
        self.arms = [left_arm, right_arm]

    def open(self):
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pool.map(Xr2Arm.open, self.arms)
        return self

    def recv_delta(self, prev: robo.Device, cur: robo.Device):
        self.pool.map(
            Xr2Arm.recv_delta,
            self.arms,
            [prev] * len(self.arms),
            [cur] * len(self.arms),
        )

    def close_stream(self):
        self.pool.map(Xr2Arm.close_stream, self.arms)
        self.pool.shutdown()


class ArmData:
    def __init__(self, left_arm: ArmController, right_arm: ArmController):
        self.arms = [left_arm, right_arm]

    def open(self):
        self.pool = ThreadPoolExecutor(max_workers=2)
        return self

    def get_data(self):
        left_joints, right_joints = self.pool.map(ArmController.get_joints, self.arms)
        frame = robo.DataFrame()
        for joint in left_joints[:6]:
            frame.attach(dataset.Num(joint))
        frame.attach(dataset.Num(0))
        frame.attach(dataset.Num(0))
        for joint in right_joints[:6]:
            frame.attach(dataset.Num(joint))
        frame.attach(dataset.Num(0))
        frame.attach(dataset.Num(0))
        return frame

    def close_stream(self):
        self.pool.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=bool)
    args = parser.parse_args()
    urdf_path = (
        pathlib.Path(__file__).parent.parent
        / "libs/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
    )
    if args.sim:
        sim = True
    else:
        sim = False

    left_arm = ArmController(
        "can1",
        state=ArmState(
            joints=np.zeros(7),
            solver=ArmSolver(urdf_path),
        ),
        sim=sim,
    )
    right_arm = ArmController(
        "can0",
        state=ArmState(
            joints=np.zeros(7),
            solver=ArmSolver(pathlib.Path(urdf_path)),
        ),
        sim=sim,
    )

    channel, runtime = (
        robo.RoboConfig(
            api_url="https://grpc.humanola.com",
            api_key="<YOUR_API_KEY>",
            robo_id="<YOUR_ROBO_ID>",
        )
        .attach_usb_camera(
            # topic="ego",
            # name="egocentric-camera",
            # desc="zed mini",
            # width=2560,
            # height=720,
            # rate=30,
            # kind=robo.ImageKind.Stereo,
            # id=0,
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
            topic="src:data",
            name="Left Yam",
            desc="",
            rate=120,
            v=ArmData(left_arm=left_arm, right_arm=right_arm),
            fields=[
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
                    dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
                ),
                dataset.Field(
                    name="left_arm.joint8",
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
                    dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
                ),
                dataset.Field(
                    name="right_arm.joint8",
                    dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
                ),
            ],
        )
        .run()
    )
    try:
        runtime.wait_for_interrupt()
    finally:
        l_t = threading.Thread(target=lambda: left_arm.close())
        l_t.start()
        r_t = threading.Thread(target=lambda: right_arm.close())
        r_t.start()
        l_t.join()
        r_t.join()
