__all__ = [
    "odeint",
    "CCFMFlowMatcher",
    "CCFMVariancePreservingFlowMatcher",
    "CCFMUNet",
]


def __getattr__(name: str):
    if name == "odeint":
        from hazematching.odeint import odeint

        return odeint

    if name in {
        "CCFMFlowMatcher",
        "CCFMVariancePreservingFlowMatcher",
        "CCFMUNet",
    }:
        from hazematching.flow_matching import (
            CCFMFlowMatcher,
            CCFMUNet,
            CCFMVariancePreservingFlowMatcher,
        )

        return {
            "CCFMFlowMatcher": CCFMFlowMatcher,
            "CCFMVariancePreservingFlowMatcher": CCFMVariancePreservingFlowMatcher,
            "CCFMUNet": CCFMUNet,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
