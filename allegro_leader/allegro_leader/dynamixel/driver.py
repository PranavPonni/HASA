from threading import Lock
from typing import Protocol, Sequence

import numpy as np
from .sdk.group_sync_read import GroupSyncRead
from .sdk.group_sync_write import GroupSyncWrite
from .sdk.packet_handler import PacketHandler
from .sdk.port_handler import PortHandler
from .sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_LOBYTE,
)

# Control Table Addresses
ADDR_TORQUE_ENABLE            = 64
ADDR_GOAL_CURRENT             = 102
LEN_GOAL_CURRENT              = 2
ADDR_PRESENT_CURRENT          = 126
LEN_PRESENT_CURRENT           = 2
ADDR_PRESENT_VELOCITY         = 128
LEN_PRESENT_VELOCITY          = 4
ADDR_PRESENT_POSITION         = 132
LEN_PRESENT_POSITION          = 4
ADDR_OPERATING_MODE           = 11

# Torque Enable/Disable
TORQUE_ENABLE                 = 1
TORQUE_DISABLE                = 0

# Operating Modes
CURRENT_CONTROL_MODE                  = 0
VELOCITY_CONTROL_MODE                 = 1
POSITION_CONTROL_MODE                 = 3
EXTENDED_POSITION_CONTROL_MODE        = 4
CURRENT_BASED_POSITION_CONTROL_MODE   = 5
PWM_CONTROL_MODE                      = 16

# Torque-to-Current Mapping (mA per N·m)
TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000/2.69,
    "XL330_M077_T": 1000/(0.215/1.47),  # ≈6838 mA per N·m
}

class DynamixelDriverProtocol(Protocol):
    def set_current(self, currents: Sequence[float]): ...
    def torque_enabled(self) -> bool: ...
    def set_torque_mode(self, enable: bool): ...
    def get_positions(self) -> np.ndarray: ...
    def get_states(self) -> tuple: ...
    def close(self): ...

class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        ids: Sequence[int],
        servo_types: Sequence[str],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4000000
    ):
        self._ids = ids
        self._lock = Lock()

        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)

        # GroupSyncRead: read current(2) + velocity(4) + position(4) in one transaction
        start_addr = ADDR_PRESENT_CURRENT
        data_len = LEN_PRESENT_CURRENT + LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            start_addr,
            data_len
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT
        )

        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for ID {dxl_id}")

        self.torque_to_current_map = np.array([
            TORQUE_TO_CURRENT_MAPPING[servo]
            for servo in servo_types
        ])
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        val = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                comm, err = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id,
                    ADDR_TORQUE_ENABLE, val
                )
                if comm != COMM_SUCCESS or err != 0:
                    raise RuntimeError(f"Failed to set torque mode for ID {dxl_id}")
        self._torque_enabled = enable

    def set_operating_mode(self, mode: int):
        for dxl_id in self._ids:
            comm, err = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id,
                ADDR_OPERATING_MODE, mode
            )
            if comm != COMM_SUCCESS or err != 0:
                raise RuntimeError(f"Failed to set operating mode for ID {dxl_id}")

    def verify_operating_mode(self, expected: int):
        for dxl_id in self._ids:
            mode, comm, err = self._packetHandler.read1ByteTxRx(
                self._portHandler, dxl_id,
                ADDR_OPERATING_MODE
            )
            if comm != COMM_SUCCESS or err != 0 or mode != expected:
                raise RuntimeError(f"Mode mismatch for ID {dxl_id}")

    def set_current_control_mode(self):
        self.set_operating_mode(CURRENT_CONTROL_MODE)
        self.verify_operating_mode(CURRENT_CONTROL_MODE)

    def set_current_based_position_control_mode(self):
        self.set_operating_mode(CURRENT_BASED_POSITION_CONTROL_MODE)
        self.verify_operating_mode(CURRENT_BASED_POSITION_CONTROL_MODE)

    def get_states(self):
        """
        Read positions [rad], velocities [rad/s],
        currents [mA], torques [N·m] in one sync transaction.
        """
        raw_pos = np.zeros(len(self._ids), dtype=int)
        raw_vel = np.zeros(len(self._ids), dtype=int)
        raw_cur = np.zeros(len(self._ids), dtype=int)

        comm = self._groupSyncRead.txRxPacket()
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"Sync read failed: {comm}")

        for i, dxl_id in enumerate(self._ids):
            # current
            if not self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            ):
                raise RuntimeError(f"Current data not available for ID {dxl_id}")
            cur = self._groupSyncRead.getData(
                dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            )
            if cur > 0x7FFF:
                cur -= 0x10000
            raw_cur[i] = cur

            # velocity
            if not self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
            ):
                raise RuntimeError(f"Velocity data not available for ID {dxl_id}")
            vel = self._groupSyncRead.getData(
                dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
            )
            if vel > 0x7FFFFFFF:
                vel -= 0x100000000
            raw_vel[i] = vel

            # position
            if not self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            ):
                raise RuntimeError(f"Position data not available for ID {dxl_id}")
            pos = self._groupSyncRead.getData(
                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            raw_pos[i] = pos

        positions = raw_pos / 2048.0 * np.pi
        velocities = raw_vel * 0.229 * 2 * np.pi / 60
        currents = raw_cur.astype(float)
        torques = currents / self.torque_to_current_map

        return positions, velocities, currents, torques

    def set_current(self, currents: Sequence[float]):
        if len(currents) != len(self._ids):
            raise ValueError("Length mismatch")
        if not self._torque_enabled:
            raise RuntimeError("Enable torque first")
        clipped = np.clip(currents, -900, 900)
        for dxl_id, cur in zip(self._ids, clipped):
            val = int(cur)
            param = [DXL_LOBYTE(val), DXL_HIBYTE(val)]
            if not self._groupSyncWrite.addParam(dxl_id, param):
                raise RuntimeError(f"Queue failed for ID {dxl_id}")
        comm = self._groupSyncWrite.txPacket()
        if comm != COMM_SUCCESS:
            raise RuntimeError("Sync write failed")
        self._groupSyncWrite.clearParam()

    def set_torque(self, torques: Sequence[float]):
        currents = self.torque_to_current_map * np.array(torques)
        self.set_current(currents)

    def get_positions(self) -> np.ndarray:
        return self.get_states()[0]

    def close(self):
        self._portHandler.closePort()


def main():
    ids = [0,1,2,3,12,13,14,15]
    types = ["XL330_M077_T"] * len(ids)
    driver = DynamixelDriver(ids, types)
    driver.set_current_based_position_control_mode()
    driver.set_torque_mode(True)
    try:
        while True:
            p,v,c,t = driver.get_states()
            print(p, v, c, t)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
