# Public Benchmarks for LACES Evaluation

This document lists the public benchmarks used for evaluating the LACES models (raw-rwkv, fixed-basis, and dynamic-basis) as discussed and prepared for Qizhi job submission.

## Core Benchmark Suite

The following benchmarks form the primary evaluation suite for the H200 jobs:

- **MMLU-Pro**
- **MMLU-Redux**
- **GPQA Diamond**
- **GSM8K**
- **C-Eval**

## Additional Benchmarks Mentioned in Technical Reports

These benchmarks were cited in the `DiffRwkv` and `LACES` technical reports and provide further context for model capabilities.

### Language Modeling & Commonsense

- LAMBADA
- PIQA
- HellaSwag
- Winogrande
- ARC-Easy
- ARC-Challenge
- GLUE (including BoolQ, COPA, CB, WiC, RTE)
- SciQ
- SQuAD
- StoryCloze

### Knowledge, Reading Comprehension, & Instruction

- MMLU (standard)
- OpenBookQA (OBQA)
- RACE
- IFEval
- BFCL V4
- TruthfulQA
- Known Unknowns (from BigBench)
- bAbILong
- PG19 (for PPL)
- WikiText-2 (for PPL)
- PTB (Penn Treebank, for PPL)
- 1BW (One Billion Word, for PPL)
