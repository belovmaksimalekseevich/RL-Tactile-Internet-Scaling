"""
MQTT message schema for telepresence suit.

Topic structure:
  suit/{subject_id}/{sensor_id}/angles
  suit/{subject_id}/{sensor_id}/accel   (optional raw IMU)

Message payload (JSON):
{
  "ts":        float,   # timestamp in seconds
  "seq":       int,     # sequence number
  "sensor_id": int,     # 0-based sensor index
  "angles":    [float], # joint angles in degrees (or axis-angle rad)
  "scenario":  str      # "s0" | "s1" | "s2"
}
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import List

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
TOPIC_BASE = "suit"

@dataclass
class SensorMessage:
    ts: float
    seq: int
    sensor_id: int
    angles: List[float]
    scenario: str = "s0"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(payload: str) -> 'SensorMessage':
        d = json.loads(payload)
        return SensorMessage(**d)

    @staticmethod
    def topic(subject_id: str, sensor_id: int) -> str:
        return f"{TOPIC_BASE}/{subject_id}/{sensor_id}/angles"
