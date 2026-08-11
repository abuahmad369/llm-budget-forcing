# Archive — superseded work

**Nothing in this folder describes the project in this repository.**

These documents belong to an earlier direction that was abandoned after
measurement. They are kept because recording *why* a direction was dropped is
part of the record.

## The abandoned direction

The original plan was to add **tool-calling and agentic multi-turn loops** to
VibeThinker-3B, composing mechanisms from Tool-N1, Search-R1, GiGPO, ARPO, and
ReTool.

It was dropped for two independently fatal reasons, both measured:

1. **Compute.** Median reasoning trace was 15,961 tokens. K=4 cost 8.87
   GPU-hours, so K=32 — which the planned methods required — projected to
   ~71 GPU-hours against a 30 hr/week cap.
2. **No headroom.** A baseline survey found samples disagreed on only **1 of 30**
   problems. Every planned method (CLR, trajectory ensembling, self-refinement)
   works by selecting among *disagreeing* candidates. There was nothing to select.

The same baseline survey produced the finding the project pivoted to: accuracy
tracked *termination*, not correctness.

## Contents

| File | What it is |
|---|---|
| `CSE465_Project_Report_VibeThinker_Agent.docx` / `.pdf` | Design report for the tool-calling/agentic plan |
| `Viva_QA_Bengali.docx` | Bengali viva prep for the same abandoned plan |
| `COST_AND_RESOURCES.md` | Cost analysis assuming a training-based approach |

All three contain estimates that later measurement contradicted. Do not quote
numbers from them.
