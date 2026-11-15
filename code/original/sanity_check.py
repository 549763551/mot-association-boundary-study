#!/usr/bin/env python3
"""Single-sequence sanity check over every configured tracker."""
import sys
import os
import json

sys.path.insert(0, "/root/mot_exp/code/TrackEval")
from evaluate import _build_config, _read_summary
from run_full_experiment import build_run_list
import trackeval

SEQ = "dancetrack0004"
BENCH = "dancetrack"
names = [r[0] for r in build_run_list()]
sm = "/tmp/seqmap_one.txt"
with open(sm, "w") as f:
    f.write("name\n" + SEQ + "\n")

d, e, m = _build_config(BENCH, names)
d["SEQMAP_FILE"] = sm
print("SANITY_START", len(names), "trackers", flush=True)
ev = trackeval.Evaluator(e)
ds = trackeval.datasets.MotChallenge2DBox(d)
ms = [trackeval.metrics.HOTA(m), trackeval.metrics.CLEAR(m), trackeval.metrics.Identity(m)]
ev.evaluate([ds], ms)

stats_path = "/root/mot_exp/results/run_stats.json"
stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
base = "/root/mot_exp/eval_data/dancetrack/trackers/MOT17-val/"
for t in names:
    s = _read_summary(base + t)
    st = stats.get("dancetrack/" + t, {}).get(SEQ, {})
    print("SANITY %-20s HOTA=%.4f IDF1=%.4f IDSW=%3.0f AssA=%.4f "
          "trig=%.3f cost=%.4f adopt=%d abstain=%d" % (
              t, s.get("HOTA", 0), s.get("IDF1", 0), s.get("IDSW", 0), s.get("AssA", 0),
              st.get("trigger_rate", 0), st.get("cost_ratio", 0),
              st.get("n_adopt", 0), st.get("n_abstain", 0)), flush=True)
print("SANITY_DONE", flush=True)
