"""GPS data provider classes."""
import serial
import pynmea2
from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime
import time
import requests

class LocationProvider(ABC):
    """Abstract base class for location providers."""

    @abstractmethod
    def get_location(self) -> Optional[Dict]:
        """Get current location data."""
        pass

    @abstractmethod
    def close(self):
        """Clean up provider resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name."""
        pass

class SerialGPSProvider(LocationProvider):
    """Handles GPS data from serial port devices."""

    def __init__(self, port: str, baudrate: int = 115200):
        """Initialize GPS provider.

        Args:
            port: Serial port name
            baudrate: Serial baud rate
        """
        self._name = f"Serial GPS ({port})"
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.track_history = []
        self.connect()

    def connect(self):
        """Establish connection to serial port."""
        if not self.serial_port or not self.serial_port.is_open:
            self.serial_port = serial.Serial(self.port, self.baudrate)
            print(f"Connected to {self.port}")

    def get_location(self) -> Optional[Dict]:
        """Read and parse GPS data."""
        if not self.serial_port or not self.serial_port.is_open:
            self.connect()
            if not self.serial_port:
                return None

        try:
            line = self.serial_port.readline().decode('ascii', errors='replace')
            data = {
                'latitude': None, 'longitude': None,
                'altitude': None, 'speed': None,
                'satellites': None, 'fix_quality': None,
                'time': None, 'date': None,
                'track_angle': None, 'hdop': None,
                'source': self.name
            }

            if line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                data['time'] = msg.timestamp
                data['date'] = msg.datestamp
                if msg.status == 'A':  # Valid fix
                    data['latitude'] = msg.latitude
                    data['longitude'] = msg.longitude
                    data['speed'] = msg.spd_over_grnd
                    data['track_angle'] = msg.true_course

                    # Store position history
                    if data['latitude'] and data['longitude']:
                        self.track_history.append((data['latitude'], data['longitude']))
                        if len(self.track_history) > 1000:  # Keep last 1000 points
                            self.track_history.pop(0)

            elif line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                data['fix_quality'] = msg.gps_qual
                data['satellites'] = msg.num_sats
                data['altitude'] = msg.altitude
                data['hdop'] = msg.horizontal_dil

            return data

        except (serial.SerialException, pynmea2.ParseError) as e:
            print(f"Error reading GPS: {e}")
            self.close()
            return None

    def close(self):
        """Close serial port connection."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"Closed {self.port}")

    @property
    def name(self) -> str:
        return self._name

class IPLocationProvider(LocationProvider):
    """Provides location data based on IP address."""

    def __init__(self):
        self._name = "IP Geolocation"
        self.last_update = 0
        self.cache_duration = 60  # Cache for 60 seconds
        self.cached_data = None

    def get_location(self):
        """Get location from IP address."""
        current_time = time.time()

        # Return cached data if available and recent
        if self.cached_data and (current_time - self.last_update) < self.cache_duration:
            return self.cached_data

        try:
            # Use ip-api.com (free, no API key required)
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                result = response.json()
                data = {
                    'latitude': result.get('lat'),
                    'longitude': result.get('lon'),
                    'source': self.name
                }

                self.cached_data = data
                self.last_update = current_time
                return data

        except Exception as e:
            print(f"Error getting IP location: {e}")
        return None

    def close(self):
        """Clean up resources (no-op for IP provider)."""
        pass

    @property
    def name(self):
        return self._name
