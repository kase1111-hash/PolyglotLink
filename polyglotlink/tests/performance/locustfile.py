"""
PolyglotLink Load Testing with Locust

Usage:
    # Start locust web UI:
    locust -f polyglotlink/tests/performance/locustfile.py

    # Run headless load test:
    locust -f polyglotlink/tests/performance/locustfile.py \
        --headless -u 100 -r 10 -t 60s \
        --host http://localhost:8080

    # Generate HTML report:
    locust -f polyglotlink/tests/performance/locustfile.py \
        --headless -u 100 -r 10 -t 60s \
        --host http://localhost:8080 \
        --html=load_test_report.html
"""

import json
import random
import time
from locust import HttpUser, task, between, events
from datetime import datetime


class IoTDevicePayloads:
    """Collection of realistic IoT device payloads for load testing."""

    @staticmethod
    def environmental_sensor():
        return {
            "temperature": round(random.uniform(15.0, 35.0), 2),
            "humidity": random.randint(30, 90),
            "pressure_hpa": round(random.uniform(980.0, 1050.0), 2),
            "co2_ppm": random.randint(300, 800),
            "voc_index": random.randint(0, 500),
            "device_id": f"env-sensor-{random.randint(1, 1000):04d}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    @staticmethod
    def power_meter():
        return {
            "voltage": round(random.uniform(220.0, 240.0), 2),
            "current": round(random.uniform(0.1, 15.0), 2),
            "power_w": round(random.uniform(10.0, 3600.0), 2),
            "energy_kwh": round(random.uniform(0.0, 10000.0), 2),
            "power_factor": round(random.uniform(0.8, 1.0), 3),
            "frequency_hz": round(random.uniform(49.5, 50.5), 2),
            "device_id": f"meter-{random.randint(1, 500):04d}",
            "ts": int(time.time() * 1000)
        }

    @staticmethod
    def gps_tracker():
        return {
            "lat": round(random.uniform(-90.0, 90.0), 6),
            "lng": round(random.uniform(-180.0, 180.0), 6),
            "altitude_m": round(random.uniform(0, 5000), 1),
            "speed_kmh": round(random.uniform(0, 200), 1),
            "heading": random.randint(0, 359),
            "accuracy": round(random.uniform(1.0, 50.0), 1),
            "satellites": random.randint(4, 12),
            "device_id": f"tracker-{random.randint(1, 2000):04d}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    @staticmethod
    def industrial_plc():
        return {
            "registers": [random.randint(0, 65535) for _ in range(10)],
            "status": random.choice([0, 1, 2, 3]),
            "alarm": random.choice([True, False]),
            "setpoint": round(random.uniform(0, 100), 1),
            "process_value": round(random.uniform(0, 100), 1),
            "error_code": random.choice([0, 0, 0, 0, 1, 2, 3]),
            "plc_id": f"plc-{random.randint(1, 100):03d}",
            "cycle_time_ms": random.randint(10, 100)
        }

    @staticmethod
    def nested_sensor():
        return {
            "device": {
                "id": f"complex-{random.randint(1, 500):04d}",
                "type": random.choice(["environmental", "industrial", "smart_home"]),
                "firmware": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 20)}"
            },
            "readings": {
                "temperature": {"value": round(random.uniform(15.0, 35.0), 2), "unit": "celsius"},
                "humidity": {"value": random.randint(30, 90), "unit": "percent"},
                "pressure": {"value": round(random.uniform(980.0, 1050.0), 2), "unit": "hpa"}
            },
            "meta": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sequence": random.randint(1, 1000000)
            }
        }


class PolyglotLinkUser(HttpUser):
    """Simulates IoT devices sending data to PolyglotLink HTTP endpoint."""

    wait_time = between(0.1, 1.0)  # Wait 100ms to 1s between requests

    def on_start(self):
        """Called when a simulated user starts."""
        self.payloads = IoTDevicePayloads()

    @task(30)
    def send_environmental_sensor_data(self):
        """Most common: environmental sensor data."""
        payload = self.payloads.environmental_sensor()
        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [environmental]"
        )

    @task(20)
    def send_power_meter_data(self):
        """Power meter readings."""
        payload = self.payloads.power_meter()
        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [power_meter]"
        )

    @task(15)
    def send_gps_tracker_data(self):
        """GPS tracker location updates."""
        payload = self.payloads.gps_tracker()
        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [gps_tracker]"
        )

    @task(10)
    def send_plc_data(self):
        """Industrial PLC data."""
        payload = self.payloads.industrial_plc()
        self.client.post(
            "/ingest/http",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/http [plc]"
        )

    @task(10)
    def send_nested_payload(self):
        """Complex nested payload."""
        payload = self.payloads.nested_sensor()
        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [nested]"
        )

    @task(5)
    def check_health(self):
        """Health check endpoint."""
        self.client.get("/health", name="/health")

    @task(5)
    def get_metrics(self):
        """Metrics endpoint."""
        self.client.get("/metrics", name="/metrics")

    @task(5)
    def send_batch_data(self):
        """Send batch of messages."""
        batch = [self.payloads.environmental_sensor() for _ in range(10)]
        self.client.post(
            "/ingest/batch",
            json=batch,
            headers={"Content-Type": "application/json"},
            name="/ingest/batch [10 messages]"
        )


class HighThroughputUser(HttpUser):
    """Simulates high-frequency IoT devices for stress testing."""

    wait_time = between(0.01, 0.05)  # Very short wait times

    def on_start(self):
        self.payloads = IoTDevicePayloads()

    @task
    def rapid_fire_sensor_data(self):
        """Rapid fire sensor data for stress testing."""
        payload = self.payloads.environmental_sensor()
        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [stress]"
        )


class LargePayloadUser(HttpUser):
    """Tests handling of large payloads."""

    wait_time = between(1.0, 3.0)

    @task
    def send_large_payload(self):
        """Send a large payload with many fields."""
        payload = {
            f"sensor_{i}": {
                "value": round(random.uniform(0, 100), 2),
                "unit": random.choice(["celsius", "percent", "ppm", "hpa"]),
                "quality": random.choice(["good", "warning", "error"]),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            for i in range(100)
        }
        payload["device_id"] = f"large-device-{random.randint(1, 100):03d}"

        self.client.post(
            "/ingest/mqtt",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/ingest/mqtt [large_payload]"
        )


# Event hooks for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request details for analysis."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("PolyglotLink Load Test Starting")
    print(f"Target host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("PolyglotLink Load Test Complete")
    print("=" * 60)
