import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CITY_ALIASES = {}

WEATHER_CODES = {
    0: "晴朗", 1: "大致晴朗", 2: "局部多云", 3: "阴天", 45: "有雾",
    48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "较强毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 71: "小雪", 73: "中雪",
    75: "大雪", 80: "小阵雨", 81: "阵雨", 82: "强阵雨", 95: "雷暴",
    96: "雷暴伴小冰雹", 99: "雷暴伴冰雹",
}
WEATHER_CODES_EN = {
    0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
    48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain", 71: "Light snow",
    73: "Moderate snow", 75: "Heavy snow", 80: "Light showers", 81: "Showers",
    82: "Heavy showers", 95: "Thunderstorm", 96: "Thunderstorm with light hail",
    99: "Thunderstorm with hail",
}


def extract_location(text: str) -> str | None:
    parenthetical = re.search(r"[（(]([A-Za-z][A-Za-z .'-]{1,40})[）)]", text)
    if parenthetical:
        return parenthetical.group(1).strip()
    for chinese, english in CITY_ALIASES.items():
        if chinese in text:
            return english
    mixed = re.search(
        r"(?i)(?:今天|明天|后天|today|tomorrow)?\s*([A-Za-z][A-Za-z .'-]{1,40}?)\s*(?:今天|明天|后天|today|tomorrow)?\s*(?:的)?\s*(?:天气|气温|温度|降水|下雨|风力|预报|weather|forecast)",
        text,
    )
    if mixed:
        return mixed.group(1).strip()
    english = re.search(r"(?i)(?:weather|forecast)\s+(?:in|for)\s+([A-Za-z .'-]+?)(?:\s+today|\s+tomorrow|[?.!,]|$)", text)
    if english:
        return english.group(1).strip()
    chinese = re.search(r"(?:查询|看看|告诉我)?([\u4e00-\u9fff]{2,12})(?:今天|明天|后天|未来|的天气|天气|气温|温度|降水|风力)", text)
    return chinese.group(1) if chinese else None


def _get_json(base_url: str, params: dict) -> dict:
    request = Request(f"{base_url}?{urlencode(params)}", headers={"User-Agent": "FamilyAI/0.1"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def get_forecast_range(location: str, day_offset: int = 0, days: int = 1, language: str = "zh") -> list[dict]:
    geo = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "language": language, "format": "json"},
    )
    results = geo.get("results") or []
    if not results:
        raise ValueError((f"找不到地点：{location}" if language == "zh" else f"Location not found: {location}"))
    place = results[0]
    forecast = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max",
            "timezone": "auto",
            "forecast_days": min(16, max(3, day_offset + days)),
        },
    )
    daily = forecast["daily"]
    place_name = ", ".join(x for x in [place.get("name"), place.get("admin1"), place.get("country")] if x)
    forecasts = []
    for index in range(day_offset, min(day_offset + days, len(daily["time"]))):
        forecasts.append({
            "place": place_name,
            "date": daily["time"][index],
            "condition_code": daily["weather_code"][index],
            "condition": WEATHER_CODES.get(daily["weather_code"][index], f"天气代码 {daily['weather_code'][index]}"),
            "high_c": daily["temperature_2m_max"][index],
            "low_c": daily["temperature_2m_min"][index],
            "rain_probability": daily["precipitation_probability_max"][index],
            "precipitation_mm": daily["precipitation_sum"][index],
            "wind_kmh": daily["wind_speed_10m_max"][index],
            "gust_kmh": daily["wind_gusts_10m_max"][index],
        })
    return forecasts


def get_forecast(location: str, day_offset: int = 0) -> dict:
    return get_forecast_range(location, day_offset, 1)[0]


def format_forecast(data: dict, language: str = "zh") -> str:
    high_f = data["high_c"] * 9 / 5 + 32
    low_f = data["low_c"] * 9 / 5 + 32
    if language == "en":
        condition = WEATHER_CODES_EN.get(data.get("condition_code"), "Unknown conditions")
        return f"""{data['place']} · {data['date']}

Weather: {condition}
Temperature: {low_f:.0f}–{high_f:.0f}°F ({data['low_c']:.1f}–{data['high_c']:.1f}°C)
Maximum precipitation chance: {data['rain_probability']}%; expected precipitation: {data['precipitation_mm']} mm
Maximum sustained wind: {data['wind_kmh']} km/h; gusts: {data['gust_kmh']} km/h

Source: Open-Meteo live forecast
https://open-meteo.com/"""
    return f"""{data['place']} · {data['date']}

天气：{data['condition']}
温度：{data['low_c']:.1f}–{data['high_c']:.1f}°C（{low_f:.0f}–{high_f:.0f}°F）
最高降水概率：{data['rain_probability']}%，预计降水 {data['precipitation_mm']} mm
最大持续风速：{data['wind_kmh']} km/h；阵风：{data['gust_kmh']} km/h

数据来源：Open-Meteo 实时更新预报
https://open-meteo.com/"""


def format_forecast_range(items: list[dict], language: str = "zh") -> str:
    if len(items) == 1:
        return format_forecast(items[0], language)
    if language == "en":
        lines = [f"{items[0]['place']} · {len(items)}-day forecast", ""]
        for item in items:
            high_f = item["high_c"] * 9 / 5 + 32
            low_f = item["low_c"] * 9 / 5 + 32
            condition = WEATHER_CODES_EN.get(item.get("condition_code"), "Unknown conditions")
            lines.extend([
                f"{item['date']} | {condition}",
                f"Temperature: {low_f:.0f}–{high_f:.0f}°F ({item['low_c']:.1f}–{item['high_c']:.1f}°C)",
                f"Precipitation chance: {item['rain_probability']}%; precipitation: {item['precipitation_mm']} mm; maximum wind: {item['wind_kmh']} km/h; gusts: {item['gust_kmh']} km/h",
                "",
            ])
        lines.extend(["Source: Open-Meteo live forecast", "https://open-meteo.com/"])
        return "\n".join(lines)
    lines = [f"{items[0]['place']} · 未来 {len(items)} 天预报", ""]
    for item in items:
        high_f = item["high_c"] * 9 / 5 + 32
        low_f = item["low_c"] * 9 / 5 + 32
        lines.extend([
            f"{item['date']}｜{item['condition']}",
            f"温度 {item['low_c']:.1f}–{item['high_c']:.1f}°C（{low_f:.0f}–{high_f:.0f}°F）",
            f"降水概率 {item['rain_probability']}%，降水 {item['precipitation_mm']} mm；最大风速 {item['wind_kmh']} km/h，阵风 {item['gust_kmh']} km/h",
            "",
        ])
    lines.extend(["数据来源：Open-Meteo 实时更新预报", "https://open-meteo.com/"])
    return "\n".join(lines)
