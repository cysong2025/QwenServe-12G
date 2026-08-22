# E06 Correctness Canary

Generated at: 2026-08-22T11:45:28.545521+00:00

Overall status: **PASS**
Configuration equivalence: **PASS**
Task quality no-regression: **PASS**

Dataset SHA-256 match: YES
Prompt matches: 24/24
Outputs matching across all four cells: 24/24
Expected-answer matches A/B/C/D: 22/22/22/22
APC hit rate C/D: 84.89%/84.89%

A=8192/OFF, B=2048/OFF, C=8192/ON, D=2048/ON. The gate requires all four outputs to match for every case and both APC cells to record cache hits. Base-model mistakes shared by all cells are reported but are not treated as configuration regressions.

| Case | Group | Prompt | Expected A/B/C/D | All outputs match |
|---|---|---|---:|---|
| capacity-01 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-02 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-03 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-04 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-05 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-06 | capacity_lookup | MATCH | FAIL/FAIL/FAIL/FAIL | MATCH |
| capacity-07 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| capacity-08 | capacity_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-01 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-02 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-03 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-04 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-05 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-06 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-07 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| deployment-08 | deployment_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-01 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-02 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-03 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-04 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-05 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-06 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
| incident-07 | incident_lookup | MATCH | FAIL/FAIL/FAIL/FAIL | MATCH |
| incident-08 | incident_lookup | MATCH | PASS/PASS/PASS/PASS | MATCH |
