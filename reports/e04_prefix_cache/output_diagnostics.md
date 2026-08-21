# E04 Output Diagnostics

Generated at: 2026-08-21T01:14:14.247626+00:00

This diagnostic reads existing detailed result JSON files only; it does not run the model.
The multiset metric ignores completion order, while the positional metric compares each request slot.

Exact multiset-matching pairs: 1/18
Positional matches: 1636/1800
Multiset overlap: 1639/1800

| Condition | C | Rep | Seed | OFF/ON outputs | Positional match | Multiset overlap | Exact multiset |
|---|---:|---:|---:|---:|---:|---:|---|
| capacity_reuse90_p1792 | 8 | 1 | 20310821 | 100/100 | 87 (87.00%) | 88 (88.00%) | NO |
| capacity_reuse90_p1792 | 8 | 2 | 20310921 | 100/100 | 89 (89.00%) | 89 (89.00%) | NO |
| capacity_reuse90_p1792 | 8 | 3 | 20311021 | 100/100 | 80 (80.00%) | 81 (81.00%) | NO |
| reuse0_p1024 | 4 | 1 | 20260821 | 100/100 | 88 (88.00%) | 88 (88.00%) | NO |
| reuse0_p1024 | 4 | 2 | 20260921 | 100/100 | 98 (98.00%) | 98 (98.00%) | NO |
| reuse0_p1024 | 4 | 3 | 20261021 | 100/100 | 100 (100.00%) | 100 (100.00%) | YES |
| reuse50_p1024 | 4 | 1 | 20270821 | 100/100 | 95 (95.00%) | 95 (95.00%) | NO |
| reuse50_p1024 | 4 | 2 | 20270921 | 100/100 | 96 (96.00%) | 96 (96.00%) | NO |
| reuse50_p1024 | 4 | 3 | 20271021 | 100/100 | 91 (91.00%) | 91 (91.00%) | NO |
| reuse90_p1024 | 4 | 1 | 20280821 | 100/100 | 91 (91.00%) | 91 (91.00%) | NO |
| reuse90_p1024 | 4 | 2 | 20280921 | 100/100 | 84 (84.00%) | 84 (84.00%) | NO |
| reuse90_p1024 | 4 | 3 | 20281021 | 100/100 | 89 (89.00%) | 90 (90.00%) | NO |
| reuse90_p1792 | 4 | 1 | 20300821 | 100/100 | 90 (90.00%) | 90 (90.00%) | NO |
| reuse90_p1792 | 4 | 2 | 20300921 | 100/100 | 92 (92.00%) | 92 (92.00%) | NO |
| reuse90_p1792 | 4 | 3 | 20301021 | 100/100 | 89 (89.00%) | 89 (89.00%) | NO |
| reuse90_p256 | 4 | 1 | 20290821 | 100/100 | 93 (93.00%) | 93 (93.00%) | NO |
| reuse90_p256 | 4 | 2 | 20290921 | 100/100 | 91 (91.00%) | 91 (91.00%) | NO |
| reuse90_p256 | 4 | 3 | 20291021 | 100/100 | 93 (93.00%) | 93 (93.00%) | NO |
