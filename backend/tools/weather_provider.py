"""
QORIN Real-Time Weather Provider
Queries authentic live weather conditions via Open-Meteo APIs (Geocoding + Forecast).
Includes WMO weather condition resolution, caching, structured metadata, and graceful error handling.
"""

from dataclasses import dataclass, asdict
import json
import time
from typing import Any, Dict, Optional, Tuple
import urllib.parse
import urllib.request


WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass
class WeatherObservation:
    location_name: str
    country: str
    admin1: str
    latitude: float
    longitude: float
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_pct: int
    wind_speed_kmh: float
    weather_condition: str
    weather_code: int
    observation_time: str
    source: str = "Open-Meteo Weather API (https://open-meteo.com)"
    source_url: str = "https://open-meteo.com"
    is_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_for_model(self) -> str:
        loc = f"{self.location_name}, {self.admin1}, {self.country}".strip(", ")
        return (
            f"Weather Report for {loc}:\n"
            f"- Temperature: {self.temperature_c:.1f}°C (Feels like: {self.apparent_temperature_c:.1f}°C)\n"
            f"- Conditions: {self.weather_condition}\n"
            f"- Relative Humidity: {self.relative_humidity_pct}%\n"
            f"- Wind Speed: {self.wind_speed_kmh:.1f} km/h\n"
            f"- Observation Time: {self.observation_time}\n"
            f"Source: {self.source} ({self.source_url})"
        )


class RealWeatherProvider:
    """Authentic live weather provider querying Open-Meteo services."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    CACHE_TTL_SECONDS = 600.0  # 10 minutes cache

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, Tuple[float, WeatherObservation]] = {}

    def _geocode(self, location_query: str) -> Optional[Dict[str, Any]]:
        clean_query = location_query.strip()
        params = urllib.parse.urlencode({
            "name": clean_query,
            "count": 1,
            "language": "en",
            "format": "json"
        })
        req = urllib.request.Request(
            f"{self.GEOCODING_URL}?{params}",
            headers={"User-Agent": "QORIN-Inference-Engine/1.0 (Educational/Research)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    results = data.get("results")
                    if results and len(results) > 0:
                        return results[0]
        except Exception:
            pass
        return None

    def get_weather(self, location: str, duration_days: int = 1) -> Tuple[bool, Optional[WeatherObservation], Optional[str]]:
        """
        Retrieves real-time weather observation for the given location.
        Returns: (success: bool, observation: Optional[WeatherObservation], error_message: Optional[str])
        """
        clean_loc = location.strip()
        if not clean_loc:
            return False, None, "Location string cannot be empty."

        cache_key = clean_loc.lower()
        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_obs = self._cache[cache_key]
            if now - cached_time < self.CACHE_TTL_SECONDS:
                obs_copy = WeatherObservation(**asdict(cached_obs))
                obs_copy.is_cached = True
                return True, obs_copy, None

        # 1. Geocode location
        geo = self._geocode(clean_loc)
        if not geo:
            return False, None, f"Could not resolve geographical coordinates for location: '{location}'."

        lat = geo.get("latitude")
        lon = geo.get("longitude")
        name = geo.get("name", location)
        admin1 = geo.get("admin1", "")
        country = geo.get("country", "")

        # 2. Query forecast API
        days = max(1, min(7, duration_days))
        params = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
            "forecast_days": days
        })
        req = urllib.request.Request(
            f"{self.FORECAST_URL}?{params}",
            headers={"User-Agent": "QORIN-Inference-Engine/1.0 (Educational/Research)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status != 200:
                    return False, None, f"Weather service responded with status {resp.status}."
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return False, None, f"Weather provider unavailable: {e}"

        current = data.get("current")
        if not current:
            return False, None, "Weather provider returned empty current conditions."

        wcode = current.get("weather_code", 0)
        condition_str = WMO_WEATHER_CODES.get(wcode, "Clear")
        obs_time = current.get("time", time.strftime("%Y-%m-%dT%H:%M"))

        obs = WeatherObservation(
            location_name=name,
            country=country,
            admin1=admin1,
            latitude=lat,
            longitude=lon,
            temperature_c=float(current.get("temperature_2m", 0.0)),
            apparent_temperature_c=float(current.get("apparent_temperature", 0.0)),
            relative_humidity_pct=int(current.get("relative_humidity_2m", 0)),
            wind_speed_kmh=float(current.get("wind_speed_10m", 0.0)),
            weather_condition=condition_str,
            weather_code=wcode,
            observation_time=obs_time,
            source=f"Open-Meteo Weather API (lat: {lat:.4f}, lon: {lon:.4f})",
            source_url="https://open-meteo.com",
            is_cached=False
        )

        self._cache[cache_key] = (now, obs)
        return True, obs, None
