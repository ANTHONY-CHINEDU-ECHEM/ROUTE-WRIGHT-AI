"""Instruction tuning export.

Turns the incident base into supervised fine tuning data in the chat messages
JSONL format. Two record types are produced:

grounded
    The user turn is a full evidence pack and the assistant turn is the verified
    extractive answer, teaching any open model to answer strictly from retrieved
    evidence with correct citations.
case
    The user turn describes one incident and the assistant turn gives the
    documented fix, result and lesson, distilling the incident base itself.
"""

import json
import random

from .answer import SYSTEM_PROMPT, evidence_pack
from .evaluate import build_queries
from .utils import clean_dashes, sub

CASE_SYSTEM = ("You are Routewright AI, a senior last mile delivery operations advisor. Explain how a delivery "
               "incident was handled, what happened and what other operators should learn. Never use hyphens or dashes.")


def grounded_records(engine, limit=1000, seed=7):
    rng = random.Random(seed)
    half = int(limit) // 2
    questions = [q["text"] for q in build_queries(half, seed)]
    for row in rng.sample(range(engine.index.size), sub(int(limit), half)):
        questions.append("This is happening to us: " + engine.case(row)["incident_summary"] + " What should we do?")
    for q in questions:
        result = engine.ask(q, provider="extractive")
        if result["verification"]["passed"]:
            yield {"type": "grounded", "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evidence_pack(q, result)},
                {"role": "assistant", "content": result["answer"]},
            ]}


def case_records(engine, limit=None):
    total = engine.index.size if limit is None else min(int(limit), engine.index.size)
    for row in range(total):
        c = engine.case(row)
        prompt = ("{op} runs {service} in a {area} area using {tool}. {summary} How was this handled and what "
                  "should we learn?").format(op=c["operator_name"], service=c["service_type"].lower(),
                                             area=c["area_type"].lower(), tool=c["planning_tool"].lower(),
                                             summary=c["incident_summary"])
        reply = "{} {} [{}]".format(c["resolution_narrative"], c["lessons_learned"], c["incident_id"])
        yield {"type": "case", "messages": [
            {"role": "system", "content": CASE_SYSTEM},
            {"role": "user", "content": clean_dashes(prompt)},
            {"role": "assistant", "content": clean_dashes(reply)},
        ]}


def export(engine, path, grounded=1000, cases=None, log=print):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"grounded": 0, "case": 0}
    with open(path, "w", encoding="utf8") as fh:
        for rec in grounded_records(engine, grounded):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["grounded"] += 1
        for rec in case_records(engine, cases):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["case"] += 1
    log("  wrote {grounded} grounded and {case} case records".format(**counts))
    return counts
