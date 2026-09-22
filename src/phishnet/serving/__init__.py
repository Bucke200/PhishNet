"""Phase 6 serving package.

`tier1` scores with the Phase 3 row (a) LightGBM on the fast path; `app`
exposes it over HTTP. The legacy hard-vote urlset pipeline (`phishnet.api`)
is retired here: it was never the Phase 3 champion and carried a MongoDB
write path (`/report`) that this phase removes.
"""

from phishnet.serving.tier1 import (  # noqa: F401
    REGISTERED_THRESHOLDS,
    Tier1Servable,
)

__all__ = ["Tier1Servable", "REGISTERED_THRESHOLDS"]
