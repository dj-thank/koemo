"""現在の予定タイトルを会議メモの既定タイトル候補にする軽量連携。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

WINDOWS_TZID_TO_IANA = {
    "tokyo standard time": "Asia/Tokyo",
    "utc": "UTC",
    "greenwich standard time": "Europe/London",
    "gmt standard time": "Europe/London",
    "eastern standard time": "America/New_York",
    "central standard time": "America/Chicago",
    "mountain standard time": "America/Denver",
    "pacific standard time": "America/Los_Angeles",
}


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    start: datetime
    end: datetime


def _unfold_ics(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        elif raw:
            lines.append(raw)
    return lines


def _unescape_ics(value: str) -> str:
    return (value.replace(r"\n", " ")
                 .replace(r"\N", " ")
                 .replace(r"\,", ",")
                 .replace(r"\;", ";")
                 .replace(r"\\", "\\")).strip()


def _parse_ics_datetime(value: str, local_tz, params: dict[str, str] | None = None) -> datetime | None:
    value = value.strip()
    params = params or {}
    if not value:
        return None
    tz = local_tz
    tzid = (params.get("TZID") or "").strip().strip('"')
    if tzid:
        try:
            tz = ZoneInfo(WINDOWS_TZID_TO_IANA.get(tzid.lower(), tzid))
        except Exception:
            tz = local_tz
    if re.fullmatch(r"\d{8}", value):
        return datetime.combine(date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}"),
                                time.min, tzinfo=tz).astimezone(local_tz)
    if value.endswith("Z"):
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).astimezone(local_tz)
        except ValueError:
            return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=tz).astimezone(local_tz)
    except ValueError:
        return None


def _split_ics_property(line: str) -> tuple[str, dict[str, str], str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    parts = key.split(";")
    name = parts[0].upper()
    params: dict[str, str] = {}
    for raw in parts[1:]:
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        params[k.upper()] = v
    return name, params, value


def load_ics_events(path, now=None) -> list[CalendarEvent]:
    """ICS内のVEVENTから SUMMARY / DTSTART / DTEND を読む。繰り返し展開はしない。"""
    p = Path(path)
    if not p.is_file():
        return []
    current = (now or datetime.now().astimezone())
    local_tz = current.tzinfo or datetime.now().astimezone().tzinfo
    events: list[CalendarEvent] = []
    event: dict[str, str] | None = None
    for line in _unfold_ics(p.read_text(encoding="utf-8-sig", errors="ignore")):
        if line == "BEGIN:VEVENT":
            event = {}
            continue
        if line == "END:VEVENT":
            if event:
                title = _unescape_ics(event.get("SUMMARY", ""))
                start = _parse_ics_datetime(event.get("DTSTART", ""), local_tz,
                                            event.get("_DTSTART_PARAMS", {}))
                end = _parse_ics_datetime(event.get("DTEND", ""), local_tz,
                                          event.get("_DTEND_PARAMS", {}))
                if title and start:
                    if end is None or end <= start:
                        end = start + timedelta(hours=1)
                    events.append(CalendarEvent(title, start, end))
            event = None
            continue
        if event is None:
            continue
        prop = _split_ics_property(line)
        if prop is None:
            continue
        name, params, value = prop
        if name in {"SUMMARY", "DTSTART", "DTEND"}:
            event[name] = value
            if params:
                event[f"_{name}_PARAMS"] = params
    return events


def _outlook_events(now, window_before_min, window_after_min) -> list[CalendarEvent]:
    try:
        import win32com.client  # type: ignore
    except Exception:
        return []
    try:
        app = win32com.client.Dispatch("Outlook.Application")
        ns = app.GetNamespace("MAPI")
        items = ns.GetDefaultFolder(9).Items
        items.Sort("[Start]")
        items.IncludeRecurrences = True
        start = now - timedelta(minutes=window_before_min)
        end = now + timedelta(minutes=window_after_min)
        flt = ("[Start] <= '" + end.strftime("%m/%d/%Y %I:%M %p") +
               "' AND [End] >= '" + start.strftime("%m/%d/%Y %I:%M %p") + "'")
        matched = items.Restrict(flt)
        out: list[CalendarEvent] = []
        for item in matched:
            title = str(getattr(item, "Subject", "") or "").strip()
            if not title:
                continue
            ev_start = getattr(item, "Start", None)
            ev_end = getattr(item, "End", None)
            if ev_start and ev_end:
                out.append(CalendarEvent(title, ev_start, ev_end))
        return out
    except Exception:
        return []


def _as_local(value, local_tz):
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=local_tz)
    return value.astimezone(local_tz)


def current_title(cfg, now=None) -> str:
    if not cfg.get("enable_calendar_title_hint", False):
        return ""
    current = now or datetime.now().astimezone()
    try:
        before = int(cfg.get("calendar_title_lookback_min", 15) or 15)
    except Exception:
        before = 15
    if before < 0:
        before = 15
    try:
        after = int(cfg.get("calendar_title_lookahead_min", 10) or 10)
    except Exception:
        after = 10
    if after < 0:
        after = 10
    events = []
    ics_path = (cfg.get("calendar_ics_path") or "").strip()
    if ics_path:
        events.extend(load_ics_events(ics_path, current))
    if cfg.get("calendar_outlook_enabled", False):
        events.extend(_outlook_events(current, before, after))
    window_start = current - timedelta(minutes=before)
    window_end = current + timedelta(minutes=after)
    local_tz = current.tzinfo or datetime.now().astimezone().tzinfo
    normalized = [
        CalendarEvent(e.title, _as_local(e.start, local_tz), _as_local(e.end, local_tz))
        for e in events
    ]
    in_progress = [e for e in normalized if e.start <= current <= e.end]
    if in_progress:
        in_progress.sort(key=lambda e: (e.start, e.end), reverse=True)
        return in_progress[0].title
    nearby = [e for e in normalized if e.start <= window_end and e.end >= window_start]
    nearby.sort(key=lambda e: (abs((e.start - current).total_seconds()), e.start))
    return nearby[0].title if nearby else ""


def apply_title_hint(cfg, generated_title, now=None) -> str:
    hint = current_title(cfg, now)
    if not hint:
        return generated_title
    return hint
