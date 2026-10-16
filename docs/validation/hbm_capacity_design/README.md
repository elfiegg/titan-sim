# 192 GiB versus 256 GiB capacity study

Read the [design study](../../HBM_CAPACITY_DESIGN_STUDY.md) for the decision framing, methodology, comparison tables, and limitations.

`study.json` contains the eight selected configurations, counter conversions, nominal capacity margins, 32 V3 CPU interpolations, measured batch-sweep throughput, calibration checks, component-transfer comparisons, historical V4 logger semantics, and source hashes. `case_summary.csv` provides the compact case data. The PNG/SVG figures visualize the recorded counters and V3 batch sweep.

The primary comparison uses binary GiB, following the manager's design wording. Decimal GB sensitivity is retained. Six cases have recorded framework reservation below 192 GiB; two already exceed it in allocated storage. These are partial-rank, partial-window counter screens, not complete-device fit certifications. V4 has unhealthy gradients. Neither average capacity-induced throughput loss nor an economic benefit has been measured.

The V3 CPU forecasts interpolate a frozen calibrated profile; anchor agreement is not independent validation. GPT-OSS component-transfer errors concern post-step allocated storage, not transient peaks. V4 full-model capacity conclusions use measured counters, not an independent CPU peak prediction. Do not add generic overhead factors to calibrated totals without avoiding double counting.

The revised report is delivered in tab 2 of the user's existing Google document, with embedded native capacity bars, allocation comparisons, and configuration diagrams. `report.pdf` preserves the five-page visual report. The earlier eight-case dataset and PNG/SVG charts remain supporting evidence; the revised narrative is organized around V3, GPT-OSS and V4 decisions.
