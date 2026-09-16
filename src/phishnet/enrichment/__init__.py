"""Phase 3 enrichment: one provider interface, batch now, live in Phase 6.

Batch enrichment and the future live provider share the same interface,
cache key, and output schema. Phase 6 swaps the implementation with no
changes to the model's inputs. `eval.py` is untouched: enrichment joins
offline from a versioned snapshot; predictors never touch the network.
"""
