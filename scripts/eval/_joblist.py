#!/usr/bin/env python3
"""Print job_id / status / name / created / started for `qz train ListJobs` JSON on stdin.

A helper file rather than a -c one-liner: quoting f-strings inside a bash heredoc has
already cost one syntax error, and ListJobs paging drops old jobs permanently, so the
parse has to work the first time.
"""
import sys
import json


def walk(o):
    if isinstance(o, dict):
        if "job_id" in o or "jobId" in o:
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def g(j, *names):
    for n in names:
        if j.get(n) not in (None, ""):
            return j[n]
    return "-"


def main():
    d = json.load(sys.stdin)
    js = list(walk(d))
    print(f"{len(js)} job(s)")
    hdr = ("job_id", "status", "name", "created", "started", "ended")
    print("{:40s} {:14s} {:40s} {:20s} {:20s} {:20s}".format(*hdr))
    for j in js:
        row = (
            str(g(j, "job_id", "jobId"))[:40],
            str(g(j, "status", "state", "job_status"))[:14],
            str(g(j, "name", "job_name", "jobName"))[:40],
            str(g(j, "created_at", "createdAt", "create_time"))[:20],
            str(g(j, "started_at", "startedAt", "start_time", "run_at"))[:20],
            str(g(j, "ended_at", "endedAt", "end_time", "finished_at"))[:20],
        )
        print("{:40s} {:14s} {:40s} {:20s} {:20s} {:20s}".format(*row))


if __name__ == "__main__":
    main()
