SUBSETS: tuple[str, ...] = (
    "zebrafish",
    "organoids1",
    "organoids2",
    "microtubule",
    "neuron",
)

DATASET_DESCRIPTIONS: dict[str, str] = {
    "zebrafish": "Zebrafish",
    "organoids1": "Organoids1",
    "organoids2": "Organoids2",
    "microtubule": "Microtubule",
    "neuron": "Neuron",
}


def _normalise_name(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


_ALIASES = {
    _normalise_name(key): key for key in SUBSETS
} | {
    _normalise_name(description): key
    for key, description in DATASET_DESCRIPTIONS.items()
}


def canonical_subset(subset: str) -> str:
    try:
        return _ALIASES[_normalise_name(subset)]
    except KeyError as exc:
        raise ValueError(
            f"subset must be one of {format_subset_options()}, got {subset!r}"
        ) from exc


def format_subset_options() -> str:
    return ", ".join(SUBSETS)
