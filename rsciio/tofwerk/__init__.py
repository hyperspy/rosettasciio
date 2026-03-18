from ._api import available_signals, compute_peak_data_from_eventlist, file_reader

__all__ = [
    "available_signals",
    "compute_peak_data_from_eventlist",
    "file_reader",
]


def __dir__():
    return sorted(__all__)
