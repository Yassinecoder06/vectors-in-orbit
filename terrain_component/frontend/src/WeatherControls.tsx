import { useState } from "react";

export type TimeOfDay = "dawn" | "day" | "sunset" | "night";
export type Weather = "clear" | "cloudy" | "foggy" | "snowy";

export interface EnvironmentSettings {
  timeOfDay: TimeOfDay;
  weather: Weather;
}

export interface EnvironmentColors {
  skyColor: string;
  groundColor: string;
  sunColor: string;
  ambientColor: string;
  fogColor: string;
  fogNear: number;
  fogFar: number;
  ambientIntensity: number;
  sunIntensity: number;
}

export const getEnvironmentColors = (timeOfDay: TimeOfDay, weather: Weather): EnvironmentColors => {
  const baseColors: Record<TimeOfDay, { sky: string; ground: string; sun: string; ambient: string; ambientIntensity: number; sunIntensity: number }> = {
    dawn: {
      sky: "#FFB6C1",
      ground: "#2F4F4F",
      sun: "#FFD700",
      ambient: "#FFA07A",
      ambientIntensity: 0.4,
      sunIntensity: 0.8,
    },
    day: {
      sky: "#87CEEB",
      ground: "#228B22",
      sun: "#FFFAF0",
      ambient: "#F0F8FF",
      ambientIntensity: 0.55,
      sunIntensity: 1.4,
    },
    sunset: {
      sky: "#FF6B35",
      ground: "#8B4513",
      sun: "#FF4500",
      ambient: "#FF8C00",
      ambientIntensity: 0.35,
      sunIntensity: 0.9,
    },
    night: {
      sky: "#0A0A20",
      ground: "#1A1A2E",
      sun: "#4169E1",
      ambient: "#191970",
      ambientIntensity: 0.15,
      sunIntensity: 0.1,
    },
  };

  const base = baseColors[timeOfDay];
  
  // Weather modifiers
  const weatherSettings: Record<Weather, { fogColor: string; fogNear: number; fogFar: number; skyOverride?: string; ambientOverride?: string }> = {
    clear: {
      fogColor: base.sky,
      fogNear: 200,
      fogFar: 500,
    },
    cloudy: {
      fogColor: "#9CA3AF",
      fogNear: 100,
      fogFar: 300,
    },
    foggy: {
      fogColor: "#9CA3AF",
      fogNear: 30,
      fogFar: 120,
      skyOverride: "#9CA3AF",
    },
    snowy: {
      fogColor: "#E5E7EB",
      fogNear: 60,
      fogFar: 200,
      skyOverride: "#E5E7EB",
      ambientOverride: "#D1D5DB",
    },
  };

  const weatherMod = weatherSettings[weather];
  
  return {
    skyColor: weatherMod.skyOverride ?? base.sky,
    groundColor: base.ground,
    sunColor: base.sun,
    ambientColor: weatherMod.ambientOverride ?? base.ambient,
    fogColor: weatherMod.fogColor,
    fogNear: weatherMod.fogNear,
    fogFar: weatherMod.fogFar,
    ambientIntensity: weather === "foggy" ? base.ambientIntensity * 0.7 : base.ambientIntensity,
    sunIntensity: weather === "foggy" ? base.sunIntensity * 0.5 : base.sunIntensity,
  };
};

export interface WeatherControlsProps {
  settings: EnvironmentSettings;
  onChange: (settings: EnvironmentSettings) => void;
  visible?: boolean;
}

const WeatherControls = ({ settings, onChange, visible = true }: WeatherControlsProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleTimeChange = (timeOfDay: TimeOfDay) => {
    onChange({
      ...settings,
      timeOfDay,
    });
  };

  const handleWeatherChange = (weather: Weather) => {
    onChange({
      ...settings,
      weather,
    });
  };

  if (!visible) return null;

  return (
    <div className={`weather-controls ${isExpanded ? "expanded" : ""}`}>
      <div className="weather-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span className="weather-icon">
          {settings.timeOfDay === "night" ? "🌙" : settings.weather === "snowy" ? "❄️" : settings.weather === "foggy" ? "🌫️" : "☀️"}
        </span>
        <span className="weather-title">Environment</span>
        <span className={`weather-chevron ${isExpanded ? "up" : "down"}`}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </span>
      </div>

      {isExpanded && (
        <div className="weather-content">
          <div className="weather-section">
            <label className="weather-label">Time of Day</label>
            <div className="weather-buttons">
              {(["dawn", "day", "sunset", "night"] as TimeOfDay[]).map((time) => (
                <button
                  key={time}
                  className={`weather-btn ${settings.timeOfDay === time ? "active" : ""}`}
                  onClick={() => handleTimeChange(time)}
                >
                  {time === "dawn" && "🌅"}
                  {time === "day" && "☀️"}
                  {time === "sunset" && "🌇"}
                  {time === "night" && "🌙"}
                </button>
              ))}
            </div>
          </div>

          <div className="weather-section">
            <label className="weather-label">Weather</label>
            <div className="weather-buttons">
              {(["clear", "cloudy", "foggy", "snowy"] as Weather[]).map((w) => (
                <button
                  key={w}
                  className={`weather-btn ${settings.weather === w ? "active" : ""}`}
                  onClick={() => handleWeatherChange(w)}
                >
                  {w === "clear" && "☀️"}
                  {w === "cloudy" && "☁️"}
                  {w === "foggy" && "🌫️"}
                  {w === "snowy" && "❄️"}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WeatherControls;
