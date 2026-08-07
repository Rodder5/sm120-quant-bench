"""Seed-pinned gold generator for the tool-calling split.

300 items across four categories (selection, extraction, abstention, compound).
Templates plus fixed lexicons only: no LLM anywhere in gold generation, so every
expected answer is deterministic and auditable by reading this file. Nothing
here shares surface text with the ultrachat calibration pool.

Every value derives from the seed. No wall-clock, no os.urandom.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schemas import (TOOLS_BY_NAME, TRANSLATE_LANGUAGES, THERMOSTAT_MODES)  # noqa: E402

# -- fixed lexicons -----------------------------------------------------------
CITIES = ["Halifax", "Nagoya", "Tampere", "Cusco", "Windhoek", "Bergen",
          "Adelaide", "Galway", "Leipzig", "Monterrey", "Da Nang", "Kelowna"]
NAMES = ["Priya Nair", "Tomas Lindgren", "Amara Diallo", "Felix Braun",
         "Ingrid Solberg", "Mateo Vargas", "Hana Kobayashi", "Lars Vestergaard",
         "Zainab Osman", "Cormac Byrne", "Elif Demir", "Rosa Camacho"]
WORDS = ["petrichor", "sonder", "liminal", "apricity", "vellichor", "susurrus",
         "defenestrate", "borborygmus", "ultracrepidarian", "psithurism"]
SKUS = ["KB-1181", "MX-0440", "TR-7729", "LP-3316", "DK-9057", "CH-2205"]
METRICS = ["disk_latency_p99", "queue_depth", "cache_hit_rate", "gc_pause_ms",
           "packet_loss_pct", "replication_lag_s"]
CHANNELS = ["#ops-alerts", "#oncall", "sre@example.com", "night-shift@example.com"]
MEETING_TOPICS = ["Q3 budget review", "sprint retrospective", "vendor renewal",
                  "incident postmortem", "roadmap sync", "hiring pipeline"]
REMINDER_TOPICS = ["water the plants", "renew the car insurance",
                   "submit the expense report", "call the dentist",
                   "rotate the backups", "defrost the freezer"]
IMG_DIRS = ["scans", "exports", "raw", "assets"]
GENERAL_KNOWLEDGE = [
    "What is the capital of Mongolia?",
    "Explain how a heat pump works in a couple of sentences.",
    "Who wrote The Left Hand of Darkness?",
    "What is the difference between a stack and a queue?",
    "Roughly how far is the Moon from Earth?",
    "Why is the sky blue?",
    "Summarise the plot of Macbeth in two sentences.",
    "What does the acronym RAID stand for in storage?",
    "How do vaccines produce immunity?",
    "What year did the Berlin Wall come down?",
]

# Spoken time phrases with their canonical HH:MM values. The trap category.
SPOKEN_TIMES = [
    ("half past nine in the morning", "09:30"),
    ("quarter to three in the afternoon", "14:45"),
    ("quarter past seven in the evening", "19:15"),
    ("ten to noon", "11:50"),
    ("twenty past four in the afternoon", "16:20"),
    ("five to midnight", "23:55"),
]
# Written amounts with separators, and their canonical floats. Also traps.
MONEY = [
    ("$1,240.50", 1240.50), ("$86.07", 86.07), ("$12,003.00", 12003.00),
    ("$999.99", 999.99), ("$4,070.25", 4070.25), ("$310.10", 310.10),
]
WORD_QTYS = [("two", 2), ("three", 3), ("four", 4), ("five", 5),
             ("a dozen", 12), ("half a dozen", 6)]
DATES = [("March 5, 2027", "2027-03-05"), ("July 22, 2027", "2027-07-22"),
         ("November 30, 2026", "2026-11-30"), ("January 9, 2028", "2028-01-09"),
         ("September 14, 2027", "2027-09-14"), ("May 1, 2027", "2027-05-01")]

NEAR_MISS = {"schedule_meeting": "schedule_reminder",
             "schedule_reminder": "schedule_meeting"}


def _offer(rng, expected, extra_pool, k, force=None):
    """Build the offered-tools list: expected tool, optional forced distractor,
    filled from extra_pool to k tools, shuffled deterministically."""
    names = set()
    if expected:
        names.add(expected)
    if force:
        names.add(force)
    pool = [n for n in extra_pool if n not in names]
    rng.shuffle(pool)
    while len(names) < k and pool:
        names.add(pool.pop())
    out = sorted(names)
    rng.shuffle(out)
    return out


ALL_NAMES = list(TOOLS_BY_NAME)


def gen_selection(rng, n):
    """Correct tool among 3 to 5 offered, always including a near-miss or
    plausible distractor."""
    items = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:  # meeting vs reminder, the engineered confusion
            tool = rng.choice(["schedule_meeting", "schedule_reminder"])
            topic = rng.choice(MEETING_TOPICS) if tool == "schedule_meeting" else rng.choice(REMINDER_TOPICS)
            date_txt, date_iso = rng.choice(DATES)
            hh = rng.randrange(8, 18)
            when = f"{date_iso} {hh:02d}:00"
            if tool == "schedule_meeting":
                who = rng.sample(NAMES, 2)
                dur = rng.choice([15, 30, 45, 60])
                msg = (f"Set up a {topic} meeting with {who[0]} and {who[1]} on "
                       f"{date_txt} at {hh}:00 for {dur} minutes.")
                args = {"title": topic, "start_time": when,
                        "duration_minutes": dur, "attendees": who}
            else:
                msg = (f"Just for me, set a reminder to {topic} on {date_txt} "
                       f"at {hh}:00. No one else involved.")
                args = {"title": topic, "remind_at": when}
            offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4, 5]),
                             force=NEAR_MISS[tool])
        elif kind == 1:  # thermostat vs weather: both about temperature
            tool = "set_thermostat"
            mode = rng.choice(THERMOSTAT_MODES)
            msg = f"Switch the thermostat to {mode} mode."
            args = {"mode": mode}
            offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4]),
                             force="get_weather")
        elif kind == 2:  # define vs translate: both about words
            tool = "define_word"
            w = rng.choice(WORDS)
            msg = f"What does the word {w} actually mean?"
            args = {"word": w}
            offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4]),
                             force="translate_text")
        else:           # invoice vs transfer: both about money
            tool = "create_invoice"
            client = rng.choice(NAMES)
            amt = rng.choice(MONEY)
            msg = (f"Bill {client} for {amt[0]}, tax applies.")
            args = {"client": client, "amount": amt[1], "taxable": True}
            offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4, 5]),
                             force="transfer_funds")
        items.append({"category": "selection", "user_message": msg,
                      "tools_offered": offered,
                      "expected": {"tool": tool, "args": args}})
    return items


def gen_extraction(rng, n):
    """Argument values embedded in natural phrasing: money with separators,
    spoken times, quantities as words, digit-string ids."""
    items = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:  # transfer with formatted money + digit-string account
            money_txt, money_val = rng.choice(MONEY)
            acct = "".join(str(rng.randrange(10)) for _ in range(10))
            msg = (f"Please send {money_txt} over to account number {acct}.")
            tool, args = "transfer_funds", {"amount": money_val, "account_id": acct}
        elif kind == 1:  # flight with worded passenger count and price cap
            o, d = rng.sample(CITIES, 2)
            qty_txt, qty = rng.choice(WORD_QTYS[:4])
            money_txt, money_val = rng.choice(MONEY)
            msg = (f"Book flights from {o} to {d} for {qty_txt} of us, "
                   f"nothing over {money_txt} a seat.")
            tool = "book_flight"
            args = {"origin": o, "destination": d, "passengers": qty,
                    "max_price": money_val}
        elif kind == 2:  # reminder with spoken time on a written date
            topic = rng.choice(REMINDER_TOPICS)
            time_txt, time_val = rng.choice(SPOKEN_TIMES)
            date_txt, date_iso = rng.choice(DATES)
            msg = (f"Remind me to {topic} at {time_txt} on {date_txt}.")
            tool = "schedule_reminder"
            args = {"title": topic, "remind_at": f"{date_iso} {time_val}"}
        else:           # image resize with dimensions in prose
            d = rng.choice(IMG_DIRS)
            f = f"{d}/img_{rng.randrange(1000, 9999)}.png"
            w, h = rng.choice([(1920, 1080), (800, 600), (1024, 1024), (640, 480)])
            msg = (f"Take {f} and make it {w} by {h} pixels.")
            tool = "resize_image"
            args = {"path": f, "width": w, "height": h}
        offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4, 5]))
        items.append({"category": "extraction", "user_message": msg,
                      "tools_offered": offered,
                      "expected": {"tool": tool, "args": args}})
    return items


def gen_abstention(rng, n):
    """No offered tool applies. Expected tool is null: the model should answer
    in prose, not force a call."""
    items = []
    for _ in range(n):
        msg = rng.choice(GENERAL_KNOWLEDGE)
        # Offer 3 to 5 tools, none of which answer a general-knowledge question.
        # define_word and translate_text are excluded: word questions could
        # legitimately route there, and gold must not punish a defensible call.
        pool = [t for t in ALL_NAMES if t not in ("define_word", "translate_text")]
        offered = _offer(rng, None, pool, rng.choice([3, 4, 5]))
        items.append({"category": "abstention", "user_message": msg,
                      "tools_offered": offered,
                      "expected": {"tool": None, "args": None}})
    return items


def gen_compound(rng, n):
    """Nested objects and arrays filled from a single message."""
    items = []
    for _ in range(n):
        kind = rng.randrange(3)
        if kind == 0:  # order: customer object + line-item array
            name = rng.choice(NAMES)
            email = name.split()[0].lower() + "@example.com"
            k = rng.choice([2, 3])
            skus = rng.sample(SKUS, k)
            qtys = [rng.choice(WORD_QTYS) for _ in range(k)]
            parts = [f"{q[0]} of {s}" for q, s in zip(qtys, skus)]
            msg = (f"New order for {name} ({email}): " + ", ".join(parts) + ".")
            tool = "create_order"
            args = {"customer": {"name": name, "email": email},
                    "items": [{"sku": s, "qty": q[1]} for q, s in zip(qtys, skus)]}
        elif kind == 1:  # alert: thresholds object + channels array
            m = rng.choice(METRICS)
            warn = rng.randrange(50, 90)
            crit = warn + rng.randrange(5, 10)
            chans = rng.sample(CHANNELS, rng.choice([1, 2]))
            msg = (f"Watch {m}: warn at {warn}, page at {crit}, "
                   f"notify " + " and ".join(chans) + ".")
            tool = "configure_alert"
            args = {"metric": m,
                    "thresholds": {"warning": float(warn), "critical": float(crit)},
                    "channels": chans}
        else:           # meeting: attendee array + duration from phrasing
            topic = rng.choice(MEETING_TOPICS)
            who = rng.sample(NAMES, 3)
            date_txt, date_iso = rng.choice(DATES)
            hh = rng.randrange(9, 16)
            dur_txt, dur = rng.choice([("an hour", 60), ("half an hour", 30),
                                       ("45 minutes", 45)])
            msg = (f"Get {who[0]}, {who[1]} and {who[2]} together for a {topic} "
                   f"on {date_txt} at {hh}:00, {dur_txt} should do it.")
            tool = "schedule_meeting"
            args = {"title": topic, "start_time": f"{date_iso} {hh:02d}:00",
                    "duration_minutes": dur, "attendees": who}
        offered = _offer(rng, tool, ALL_NAMES, rng.choice([3, 4, 5]),
                         force=NEAR_MISS.get(tool))
        items.append({"category": "compound", "user_message": msg,
                      "tools_offered": offered,
                      "expected": {"tool": tool, "args": args}})
    return items


def generate(seed, n=300):
    """Deterministic gold: four category streams, each with its own rng so
    adding a category never reshuffles the others."""
    per = n // 4
    items = (gen_selection(random.Random(f"{seed}:toolcall:selection"), per)
             + gen_extraction(random.Random(f"{seed}:toolcall:extraction"), per)
             + gen_abstention(random.Random(f"{seed}:toolcall:abstention"), per)
             + gen_compound(random.Random(f"{seed}:toolcall:compound"), n - 3 * per))
    for i, it in enumerate(items):
        it["id"] = f"toolcall-{i:05d}"
    return items


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    for it in generate(args.seed, args.n):
        print(json.dumps(it))
