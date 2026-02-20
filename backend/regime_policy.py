from __future__ import annotations

from typing import Any, Dict

REGIME_PROFILE_DEFAULT = "balanced"
REGIME_KEYS = ("trend", "chop", "vol_spike")
REGIME_LABELS = {
    "trend": "Trend",
    "chop": "Chop",
    "vol_spike": "Vol Spike",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _params(
    edge_min_up: float,
    edge_min_down: float,
    max_vol_5m: float,
    max_model_prob_up: float,
    max_model_prob_down: float,
    min_down_mom_1m_abs: float,
    risk_multiplier: float,
    fee_buffer: float = 0.03,
) -> Dict[str, float]:
    return {
        "edge_min": _clamp(edge_min_up, 0.0, 1.0),
        "edge_min_up": _clamp(edge_min_up, 0.0, 1.0),
        "edge_min_down": _clamp(edge_min_down, 0.0, 1.0),
        "max_vol_5m": _clamp(max_vol_5m, 0.0, 1.0),
        "max_model_prob_up": _clamp(max_model_prob_up, 0.0, 1.0),
        "max_model_prob_down": _clamp(max_model_prob_down, 0.0, 1.0),
        "min_down_mom_1m_abs": _clamp(min_down_mom_1m_abs, 0.0, 1.0),
        "risk_multiplier": _clamp(risk_multiplier, 0.0, 2.0),
        "fee_buffer": _clamp(fee_buffer, 0.0, 1.0),
    }


REGIME_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "balanced": {
        # Anchor "balanced" to the legacy profitable baseline, then modulate by regime.
        "trend": _params(0.11, 0.18, 0.0020, 0.75, 1.00, 0.0030, 1.00),
        "chop": _params(0.12, 0.20, 0.0019, 0.73, 0.92, 0.0035, 0.80),
        "vol_spike": _params(0.15, 0.24, 0.0018, 0.70, 0.88, 0.0045, 0.55),
    },
    "conservative": {
        "trend": _params(0.13, 0.22, 0.0019, 0.72, 0.90, 0.0035, 0.80),
        "chop": _params(0.16, 0.25, 0.0017, 0.68, 0.84, 0.0040, 0.55),
        "vol_spike": _params(0.20, 0.30, 0.0016, 0.65, 0.80, 0.0050, 0.35),
    },
    "aggressive": {
        "trend": _params(0.09, 0.15, 0.0022, 0.78, 1.00, 0.0028, 1.15),
        "chop": _params(0.10, 0.16, 0.0020, 0.75, 0.94, 0.0032, 0.90),
        "vol_spike": _params(0.13, 0.20, 0.0024, 0.72, 0.90, 0.0038, 0.65),
    },
}


def normalize_profile(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in REGIME_PROFILES else REGIME_PROFILE_DEFAULT


def normalize_regime(raw: Any) -> str:
    txt = str(raw or "").strip().lower().replace("-", " ").replace("_", " ")
    if "vol" in txt:
        return "vol_spike"
    if "trend" in txt:
        return "trend"
    if "chop" in txt:
        return "chop"
    return "chop"


def regime_params(profile: Any, regime: Any) -> Dict[str, float]:
    profile_key = normalize_profile(profile)
    regime_key = normalize_regime(regime)
    base = REGIME_PROFILES.get(profile_key, REGIME_PROFILES[REGIME_PROFILE_DEFAULT]).get(
        regime_key, REGIME_PROFILES[REGIME_PROFILE_DEFAULT]["chop"]
    )
    out = dict(base)
    out["regime_key"] = regime_key
    out["regime_label"] = REGIME_LABELS.get(regime_key, regime_key)
    return out


def policy_snapshot(profile: Any) -> Dict[str, Any]:
    profile_key = normalize_profile(profile)
    profile_map = REGIME_PROFILES.get(profile_key, REGIME_PROFILES[REGIME_PROFILE_DEFAULT])
    regimes: Dict[str, Any] = {}
    for key in REGIME_KEYS:
        params = dict(profile_map.get(key, REGIME_PROFILES[REGIME_PROFILE_DEFAULT][key]))
        params["regime_key"] = key
        params["regime_label"] = REGIME_LABELS.get(key, key)
        regimes[key] = params
    return {
        "mode": "regime_auto",
        "profile": profile_key,
        "regimes": regimes,
    }
