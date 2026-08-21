# E04 Correctness Canary

Generated at: 2026-08-21T01:35:30.423472+00:00

Overall status: **FAIL**
APC equivalence: **PASS**
Task quality: **FAIL**

Dataset SHA-256 match: YES
Prompt matches: 24/24
OFF expected-answer matches: 22/24
ON expected-answer matches: 22/24
OFF/ON output matches: 24/24
ON prefix cache hit rate: 84.89%

| Case | Group | Prompt | OFF expected | ON expected | OFF/ON output |
|---|---|---|---|---|---|
| capacity-01 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-02 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-03 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-04 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-05 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-06 | capacity_lookup | MATCH | FAIL | FAIL | MATCH |
| capacity-07 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| capacity-08 | capacity_lookup | MATCH | PASS | PASS | MATCH |
| deployment-01 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-02 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-03 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-04 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-05 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-06 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-07 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| deployment-08 | deployment_lookup | MATCH | PASS | PASS | MATCH |
| incident-01 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-02 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-03 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-04 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-05 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-06 | incident_lookup | MATCH | PASS | PASS | MATCH |
| incident-07 | incident_lookup | MATCH | FAIL | FAIL | MATCH |
| incident-08 | incident_lookup | MATCH | PASS | PASS | MATCH |

## Task Quality Failures

| Case | Expected | OFF generated | ON generated |
|---|---|---|---|
| capacity-06 | A30 | T4 | T4 |
| incident-07 | ap-south-1 | us-east-1 | us-east-1 |
