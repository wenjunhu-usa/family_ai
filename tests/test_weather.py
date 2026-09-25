from family_ai.weather import extract_location, format_forecast, format_forecast_range


def test_extract_location():
    assert extract_location("城市（Testville）明天天气如何") == "Testville"
    assert extract_location("weather in Testville tomorrow") == "Testville"
    assert extract_location("明天Testville 天气") == "Testville"
    assert extract_location("Testville明天温度如何") == "Testville"


def test_format_forecast():
    text = format_forecast(
        {
            "place": "*",
            "date": "2026-08-08",
            "condition": "晴朗",
            "high_c": 30.0,
            "low_c": 20.0,
            "rain_probability": 10,
            "precipitation_mm": 0.0,
            "wind_kmh": 15.0,
            "gust_kmh": 25.0,
        }
    )
    assert "68–86°F" in text
    assert "Open-Meteo" in text


def test_format_forecast_range():
    sample = {
        "place": "*", "date": "2026-08-08",
        "condition": "晴朗", "high_c": 30.0, "low_c": 20.0,
        "rain_probability": 10, "precipitation_mm": 0.0,
        "wind_kmh": 15.0, "gust_kmh": 25.0,
    }
    text = format_forecast_range([sample, {**sample, "date": "2026-08-09"}, {**sample, "date": "2026-08-10"}])
    assert "未来 3 天" in text
    assert text.count("温度") == 3
