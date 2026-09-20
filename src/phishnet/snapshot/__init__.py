"""Phase 4 page snapshots: fetch once, freeze, extract-only to model.

`fetch.py` performs the single governed fetch per row (outcome taxonomy in
§3.2); `extract.py` derives the model-visible extract (raw HTML never sent);
`tier1.py` scores rows with the Phase 3 headline path (row (a)); `bands.py`
holds the half-open bucketing shared by the cascade and the tests; `step0.py`
builds the fetch set; `trigger.py` applies the Step-0 gate purely.
"""
