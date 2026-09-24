#!/usr/bin/env python3
"""Emit the three QZ submit bodies for the clean DiT width/depth ablation.

Writes one JSON per rung under scripts/ablation_ditwd/bodies/. Does NOT submit.
Values come from .omo/evidence/qz-training-skill/STANDING_SUBMIT_CONFIG.md.
"""
import json
from pathlib import Path

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
LOGDIR = f"{REPO}/logs/qz"
RUNGS = [(512, 6), (768, 8), (1024, 12)]

STANDING = dict(
    workspace_id="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    project_id="project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
    logic_compute_group_id="lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
    framework="pytorch",
    task_priority=4,
)
FRAMEWORK_CONFIG = [dict(
    image="docker.sii.shaipower.online/inspire-studio/relay2:v2",
    image_type="SOURCE_PRIVATE",
    instance_count=1,
    # 1-GPU spec: the trainer ran world_size=1 in every historical log, so extra
    # cards sit idle. Peak was 53.1GB on the 768x8 rung; 1024x12 adds ~143M params
    # (~2.3GB more with AdamW states + grads), still well inside an 80GB H100.
    spec_id="79fe954a-be92-4772-ac0b-94ad8a79b7bb",
    shm_gi=100,
)]

out = Path(__file__).parent / "bodies"
out.mkdir(exist_ok=True)

for h, d in RUNGS:
    name = f"abl-ditwd-{h}x{d}-scratch"
    log = f"{LOGDIR}/{name}.log"
    cmd = (
        f"mkdir -p {LOGDIR} && cd {REPO} && "
        f"bash scripts/ablation_ditwd/run_rung.sh {h} {d} > {log} 2>&1"
    )
    body = dict(
        STANDING,
        name=name,
        description=(
            f"Clean DiT width/depth ablation rung dit_hidden={h} dit_depth={d}, "
            "NO resume (random-init planner), 50000 steps, batch 16, save every 5000. "
            "Comparable only to the other two rungs of this set, not to the "
            "resume-derived 50k global table."
        ),
        command=cmd,
        # Measured: the 768x8 global run did 50000 steps in 539m56s (9.0h) at
        # step/s=1.5, world_size=1, mem=53.1GB
        # (outputs/2026-09-17/11-04-28/train_state_hijacking_dit.log).
        # 72h = 8x headroom, and the 1024x12 rung is the slowest of the three.
        max_running_time_ms="259200000",
        framework_config=FRAMEWORK_CONFIG,
    )
    p = out / f"{name}.json"
    p.write_text(json.dumps(body, indent=2) + "\n")
    print(f"wrote {p}")
    print(f"  cmd: {cmd}")
