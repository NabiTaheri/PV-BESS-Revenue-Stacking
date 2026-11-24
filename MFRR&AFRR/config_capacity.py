from __future__ import annotations

# Keep capacity-market configuration synchronized with DA/mFRR configuration.
# Rename this file to config_capacity.py and keep market_config.py next to it.
from market_config import *  # noqa: F401,F403

# Capacity-specific overrides can be placed below this line if needed.
# The defaults imported from market_config.py are production-safe and are used by:
# - FCR_fast_da_vs_capacity_oc.py
# - afrr_mfrr_capacity_value_no_da.py
