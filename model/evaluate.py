"""evaluate.py — condition validator and pass-rate metric"""
from __future__ import annotations
import datetime, re

MONTH_MAP = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12,
}
DAY_MAP = {0:"MON",1:"TUE",2:"WED",3:"THU",4:"FRI",5:"SAT",6:"SUN"}


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def is_valid_date_string(date_str: str) -> bool:
    """True if date_str is a valid dd-mm-yyyy date in [1-1-1800, 31-12-2200]."""
    try:
        d, m, y = (int(x) for x in date_str.strip().split("-"))
        if not (1800 <= y <= 2200):
            return False
        datetime.date(y, m, d)
        return True
    except (ValueError, AttributeError):
        return False


def check_conditions(
    date_str:   str,
    day_tok:    str,    # e.g. '[WED]'
    month_tok:  str,    # e.g. '[JAN]'
    leap_tok:   str,    # e.g. '[True]'
    decade_tok: str,    # e.g. '[192]'
) -> dict:
    result = dict(valid_date=False, day_ok=False, month_ok=False,
                  leap_ok=False, decade_ok=False, all_ok=False)
    if not is_valid_date_string(date_str):
        return result
    result["valid_date"] = True

    d, m, y   = (int(x) for x in date_str.strip().split("-"))
    date_obj  = datetime.date(y, m, d)
    strip     = lambda t: re.sub(r'[\[\]]', '', t).strip()

    result["day_ok"]    = DAY_MAP[date_obj.weekday()] == strip(day_tok)
    result["month_ok"]  = m == MONTH_MAP.get(strip(month_tok), -1)
    result["leap_ok"]   = _is_leap(y) == (strip(leap_tok) == "True")
    decade_start        = int(strip(decade_tok)) * 10
    result["decade_ok"] = decade_start <= y <= decade_start + 9
    result["all_ok"]    = all(result[k] for k in
                              ["valid_date","day_ok","month_ok","leap_ok","decade_ok"])
    return result


def evaluate_file(predictions_path: str) -> dict:
    """Reads predictions file; returns aggregate pass-rate stats."""
    lines  = open(predictions_path).read().strip().splitlines()
    total  = len(lines)
    counts = {k: 0 for k in ["valid","all","day","month","leap","decade"]}

    for line in lines:
        tokens = re.findall(r'\[([^\]]+)\]', line)
        parts  = line.rsplit(" ", 1)
        if len(parts) != 2 or len(tokens) < 4:
            continue
        date_str = parts[1].strip()
        toks     = [f"[{t}]" for t in tokens[:4]]
        r        = check_conditions(date_str, *toks)
        counts["valid"]  += r["valid_date"]
        counts["all"]    += r["all_ok"]
        counts["day"]    += r["day_ok"]
        counts["month"]  += r["month_ok"]
        counts["leap"]   += r["leap_ok"]
        counts["decade"] += r["decade_ok"]

    s = lambda n: n / total if total else 0.0
    return {
        "total":                 total,
        "valid_dates":           counts["valid"],
        "all_conditions_passed": counts["all"],
        "pass_rate":             s(counts["all"]),
        "per_condition_pass_rate": {k: s(counts[k])
                                    for k in ["day","month","leap","decade"]},
    }
