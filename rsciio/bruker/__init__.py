from ._api import file_reader
from . import api


__all__ = [
    "file_reader",
    "api",
]


def __dir__():
    return sorted(__all__)
