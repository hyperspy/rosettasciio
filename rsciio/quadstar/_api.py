# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# RosettaSciIO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RosettaSciIO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Reader for Balzers/Pfeiffer Quadstar SAC and SBC binary files.

The SAC file format stores mass spectrometry scan analog data produced by
the Quadstar software (versions 4.x and later). Each file contains a general
header, one or more trace definitions (channels), and timestamped scan
data for each trace across multiple measurement cycles.

The binary format was reverse-engineered with the help of the
`sac2dat <https://www.bubek.org/sac2dat.php>`_ tool by Dr. Moritz Bubek
and the `yadg <https://github.com/dgbowl/yadg>`_ project by Nicolas Vetsch.

The SBC file format stores mass spectrometry scan bargraph data produced by
the Quadstar software (versions 4.x and later). Each file contains a general
header but, unlike the SAC format, it does not contain trace definitions.
In Scan Bargraph mode, only the maximum peak intensities and the corresponding
mass numbers are collected. This is done by the peak detection (Peak-L and Peak-F)
integrated in the QMS. The data block starts after a variable-length header and
is parsed from file geometry and header values. Each cycle stores a fixed-size
prefix followed by interleaved (mass label, intensity) float32 pairs.

"""

import datetime
import logging
import struct

import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)

# ===========================================================================
# Binary structure dtypes
# ===========================================================================

# General file header: 200 bytes starting at offset 0x00.
_general_header_dtype = np.dtype(
    [
        ("data_index", "<i2"),  # 0x00
        ("software_id", "<i4"),  # 0x02
        ("version_major", "|u1"),  # 0x06
        ("version_minor", "|u1"),  # 0x07
        ("second", "|u1"),  # 0x08
        ("minute", "|u1"),  # 0x09
        ("hour", "|u1"),  # 0x0A
        ("day", "|u1"),  # 0x0B
        ("month", "|u1"),  # 0x0C
        ("year", "|u1"),  # 0x0D  (offset from 1900)
        ("username", "|S86"),  # 0x0E
        ("n_timesteps", "<i4"),  # 0x64
        ("n_traces", "<i2"),  # 0x68
        ("timestep_length", "<i4"),  # 0x6A
    ]
)

# Trace header: 9 bytes per trace starting at offset 0xC8.
_trace_header_dtype = np.dtype(
    [
        ("type", "|u1"),
        ("info_position", "<i4"),
        ("data_position", "<i4"),
    ]
)

# Trace info: 137 bytes at the position given by trace header.
_trace_info_dtype = np.dtype(
    [
        ("data_format", "<u2"),  # +0x00
        ("y_title", "|S13"),  # +0x02
        ("y_unit", "|S13"),  # +0x0F
        ("unknown_a", "|u1"),  # +0x1C
        ("x_title", "|S13"),  # +0x1D
        ("x_unit", "|S13"),  # +0x2A
        ("comment", "|S59"),  # +0x38  (up to 0x72)
        ("unknown_b", "|u4"),  # +0x73 (approx)
        ("unknown_c", "|u4"),  # +0x77 (approx)
        ("first_mass", "<f4"),  # +0x7A
        ("scan_width", "<u2"),  # +0x7E
        ("values_per_mass", "|u1"),  # +0x80
        ("zoom_start", "<f4"),  # +0x81
        ("zoom_end", "<f4"),  # +0x85
    ]
)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _decode_bytes(val):
    """Decode a numpy bytes field to a stripped string."""
    if isinstance(val, bytes):
        return val.decode("latin-1", errors="replace").rstrip("\x00").strip()
    return str(val)


def _read_general_header(buf):
    """Parse the general header from the start of a SAC buffer."""
    header = np.frombuffer(buf, dtype=_general_header_dtype, count=1, offset=0)[0]
    return {name: header[name].item() for name in _general_header_dtype.names}


def _read_trace_headers(buf, n_traces):
    """Read all trace headers starting at 0xC8."""
    headers = np.frombuffer(buf, dtype=_trace_header_dtype, count=n_traces, offset=0xC8)
    return [
        {name: h[name].item() for name in _trace_header_dtype.names} for h in headers
    ]


def _read_trace_info(buf, info_position):
    """Read trace info block at the given position."""
    info = np.frombuffer(buf, dtype=_trace_info_dtype, count=1, offset=info_position)[0]
    return {name: info[name].item() for name in _trace_info_dtype.names}


def _find_first_data_position(trace_headers):
    """Find the data position of the first ScanAnalog trace (type 0x11)."""
    for header in trace_headers:
        if header["type"] == 0x11:
            return header["data_position"]
    return None


def _build_datetime(header):
    """Build a datetime from the general header time fields."""
    try:
        return datetime.datetime(
            year=1900 + header["year"],
            month=header["month"],
            day=header["day"],
            hour=header["hour"],
            minute=header["minute"],
            second=header["second"],
        )
    except (ValueError, OverflowError):
        _logger.warning("Could not parse measurement date from header.")
        return None


# ===========================================================================
# SAC reader
# ===========================================================================


def _read_sac_file(filename, buf, gen, base_datetime, use_uniform_signal_axis):
    """Read mass spectrometry data from a Quadstar SAC (Scan Analog) buffer.

    Parameters
    ----------
    filename : str or Path
        Original file path (for metadata).
    buf : bytes
        Full file contents.
    gen : dict
        Parsed general header.
    base_datetime : datetime.datetime or None
        Measurement start time.

    Returns a list of signal dictionaries, one per trace.
    """
    n_timesteps = gen["n_timesteps"]
    n_traces = gen["n_traces"]
    timestep_length = gen["timestep_length"]

    _logger.debug(
        "SAC header: %d timesteps, %d traces, timestep_length=%d",
        n_timesteps,
        n_traces,
        timestep_length,
    )

    # ---- Trace headers ----------------------------------------------------
    trace_headers = _read_trace_headers(buf, n_traces)
    data_pos_0 = _find_first_data_position(trace_headers)

    if data_pos_0 is None:
        raise ValueError("No ScanAnalog data traces (type 0x11) found in SAC file.")

    # ---- Base timestamp ---------------------------------------------------
    uts_base_s = struct.unpack_from("<I", buf, 0xC2)[0]
    # Tenths of milliseconds, convert to milliseconds.
    uts_base_ms = struct.unpack_from("<H", buf, 0xC6)[0] * 0.1

    # ---- Read all timesteps and traces ------------------------------------
    trace_data = {}  # trace_index -> list of 1-D arrays (one per timestep)
    trace_timestamps = {}  # trace_index -> list of float (unix timestamps)
    trace_infos = {}  # trace_index -> info dict (same across timesteps)

    for n in range(n_timesteps):
        ts_offset = n * timestep_length

        # Read timestep UTS offset (6 bytes before first data position).
        uts_off_s = struct.unpack_from("<I", buf, data_pos_0 - 6 + ts_offset)[0]
        uts_off_ms = struct.unpack_from("<H", buf, data_pos_0 - 2 + ts_offset)[0] * 0.1
        uts_timestamp = (uts_base_s + uts_off_s) + (uts_base_ms + uts_off_ms) * 1e-3

        for ti, header in enumerate(trace_headers):
            if header["type"] != 0x11:
                continue

            info = _read_trace_info(buf, header["info_position"])
            n_datapoints = info["scan_width"] * info["values_per_mass"]

            ts_data_pos = header["data_position"] + ts_offset

            # Full-scale range exponent (int16 at offset +4 from data block).
            fsr_exp = struct.unpack_from("<h", buf, ts_data_pos + 4)[0]
            fsr = 10.0**fsr_exp

            # Read the actual measurement values (float32 array).
            yvals = np.frombuffer(
                buf, offset=ts_data_pos + 6, dtype="<f4", count=n_datapoints
            ).copy()

            # Values exceeding FSR are sensor overflow – replace with NaN.
            yvals[yvals > fsr] = np.nan

            if ti not in trace_data:
                trace_data[ti] = []
                trace_timestamps[ti] = []
                trace_infos[ti] = info

            trace_data[ti].append(yvals)
            trace_timestamps[ti].append(uts_timestamp)

    # ---- Build rosettasciio signal dictionaries ---------------------------
    signals = []

    for ti in sorted(trace_data.keys()):
        info = trace_infos[ti]
        stacked = np.stack(trace_data[ti], axis=0)  # (n_timesteps, n_datapoints)
        timestamps = np.array(trace_timestamps[ti])

        n_pts = info["scan_width"] * info["values_per_mass"]
        first_mass = float(info["first_mass"])
        scan_width = int(info["scan_width"])

        # Mass-to-charge axis
        mz_scale = scan_width / n_pts if n_pts > 0 else 1.0

        x_title = _decode_bytes(info.get("x_title", b""))
        x_unit = _decode_bytes(info.get("x_unit", b""))
        y_title = _decode_bytes(info.get("y_title", b""))
        y_unit = _decode_bytes(info.get("y_unit", b""))
        comment = _decode_bytes(info.get("comment", b""))

        signal_axis = {
            "name": x_title if x_title else "Mass-to-charge ratio",
            "units": x_unit if x_unit else "m/z",
            "offset": first_mass,
            "scale": mz_scale,
            "size": n_pts,
            "navigate": False,
        }

        axes = []
        if stacked.shape[0] > 1:
            # Navigation axis: timestep index or explicit non-uniform timestamps.
            time_axis = {
                "name": "Time",
                "units": "s",
                "size": stacked.shape[0],
                "navigate": True,
                "index_in_array": 0,
            }
            relative_timestamps = timestamps - timestamps[0]
            if use_uniform_signal_axis:
                time_axis["offset"] = 0.0
                time_axis["scale"] = np.diff(relative_timestamps).mean()
            else:
                time_axis["axis"] = relative_timestamps
            axes.append(time_axis)
            signal_axis["index_in_array"] = 1
            data = stacked
        else:
            signal_axis["index_in_array"] = 0
            data = stacked[0]

        axes.append(signal_axis)

        # Original metadata
        original_metadata = {
            "general_header": {
                k: _decode_bytes(v) if isinstance(v, bytes) else v
                for k, v in gen.items()
            },
            "trace_info": {
                k: _decode_bytes(v) if isinstance(v, bytes) else v
                for k, v in info.items()
            },
            "timestamps": timestamps.tolist(),
        }

        metadata = {
            "General": {
                "original_filename": str(filename),
            },
            "Signal": {
                "signal_type": "MS",
                "quantity": f"{y_title} ({y_unit})" if y_title else "Intensity",
            },
        }
        if base_datetime is not None:
            metadata["General"]["date"] = base_datetime.date().isoformat()
            metadata["General"]["time"] = base_datetime.time().isoformat()
        if comment:
            metadata["General"]["notes"] = comment

        signals.append(
            {
                "data": data,
                "axes": axes,
                "metadata": metadata,
                "original_metadata": original_metadata,
            }
        )

    if not signals:
        raise ValueError("No data traces could be read from the SAC file.")

    return signals


# ===========================================================================
# SBC reader
# ===========================================================================


def _read_sbc_file(filename, buf, gen, base_datetime, use_uniform_signal_axis):
    """Read mass spectrometry data from a Quadstar SBC (Scan Bargraph) buffer.

    Parameters
    ----------
    filename : str or Path
        Original file path (for metadata).
    buf : bytes
        Full file contents.
    gen : dict
        Parsed general header.
    base_datetime : datetime.datetime or None
        Measurement start time.

    Returns a list containing a single signal dictionary.
    """
    # SBC layout (reverse-engineered):
    #
    # Header fields used:
    #   0x64 -> n_timesteps (number of cycles)
    #   0xCB -> scan_width (num_masses = scan_width + 1)
    #
    # Per-cycle payload:
    #   [13-byte prefix][num_masses * (mass_f32, intensity_f32)]
    #
    # Prefix structure (13 bytes):
    #   bytes 0..3   : uint32 time offset in seconds
    #   bytes 4..7   : uint32 unknown field
    #   byte  8      : marker byte (typically 0xFB)
    #   bytes 9..12  : uint32 unknown field
    #
    # From this, stride and data start are:
    #   stride     = num_masses * 8 + 13
    #   data_start = file_size - n_timesteps * stride
    file_size = len(buf)
    num_cycles = gen["n_timesteps"]

    # Number of masses from header (scan width + 1).
    width = struct.unpack_from("<H", buf, 0xCB)[0]
    num_masses = width + 1
    mass_data_size = num_masses * 8  # interleaved (mass, intensity) float32 pairs

    # Prefix size is always 13 bytes per cycle.
    prefix_size = 13
    stride = mass_data_size + prefix_size

    # Calculate data block start from file geometry.
    # Formula: data_start = file_size - (num_cycles * stride)
    data_start = file_size - (num_cycles * stride)

    if data_start < 0x100 or data_start >= file_size:
        raise ValueError(
            f"Invalid SBC file geometry: data_start={data_start} "
            f"(file_size={file_size}, cycles={num_cycles}, masses={num_masses})"
        )

    _logger.debug(
        "SBC header: %d cycles, %d masses, stride=%d, data_start=0x%X",
        num_cycles,
        num_masses,
        stride,
        data_start,
    )

    data_buf = buf[data_start:]

    # Data is interleaved as [mass1, int1, mass2, int2, ...] in float32.
    # Use cycle 0 to recover the nominal mass axis labels.
    first_cycle = data_buf[prefix_size : prefix_size + mass_data_size]
    masses = np.frombuffer(first_cycle, dtype="<f4")[0::2].copy()

    # Extract intensity channels (odd elements of each interleaved pair).
    intensities = np.empty((num_cycles, num_masses), dtype=np.float32)
    for i in range(num_cycles):
        offset = i * stride + prefix_size
        cycle_buf = data_buf[offset : offset + mass_data_size]
        intensities[i, :] = np.frombuffer(cycle_buf, dtype="<f4")[1::2]

    # Extract per-cycle time offsets from prefix (first 4 bytes = seconds).
    timestamps = np.empty(num_cycles, dtype=np.float64)
    for i in range(num_cycles):
        t_off = struct.unpack_from("<I", data_buf, i * stride)[0]
        timestamps[i] = float(t_off)

    # ---- Build signal dictionary ------------------------------------------
    first_mass = float(masses[0])
    last_mass = float(masses[-1])
    mz_scale = (last_mass - first_mass) / (num_masses - 1) if num_masses > 1 else 1.0

    signal_axis = {
        "name": "Mass-to-charge ratio",
        "units": "m/z",
        "offset": first_mass,
        "scale": mz_scale,
        "size": num_masses,
        "navigate": False,
    }

    axes = []
    if num_cycles > 1:
        time_axis = {
            "name": "Time",
            "units": "s",
            "size": num_cycles,
            "navigate": True,
            "index_in_array": 0,
        }
        relative_timestamps = timestamps - timestamps[0]
        if use_uniform_signal_axis:
            if len(relative_timestamps) > 1:
                t_scale = np.diff(relative_timestamps).mean()
            else:
                t_scale = 1.0
            time_axis["offset"] = 0.0
            time_axis["scale"] = t_scale if t_scale > 0 else 1.0
        else:
            time_axis["axis"] = relative_timestamps
        axes.append(time_axis)
        signal_axis["index_in_array"] = 1
        data = intensities
    else:
        signal_axis["index_in_array"] = 0
        data = intensities[0]

    axes.append(signal_axis)

    original_metadata = {
        "general_header": {
            k: _decode_bytes(v) if isinstance(v, bytes) else v for k, v in gen.items()
        },
        "sbc_parameters": {
            "num_masses": num_masses,
            "stride": stride,
            "prefix_size": prefix_size,
            "data_start": data_start,
        },
        "masses": masses.tolist(),
        "timestamps": timestamps.tolist(),
    }

    metadata = {
        "General": {
            "original_filename": str(filename),
        },
        "Signal": {
            "signal_type": "MS",
            "quantity": "Intensity",
        },
    }
    if base_datetime is not None:
        metadata["General"]["date"] = base_datetime.date().isoformat()
        metadata["General"]["time"] = base_datetime.time().isoformat()

    return [
        {
            "data": data,
            "axes": axes,
            "metadata": metadata,
            "original_metadata": original_metadata,
        }
    ]


# ===========================================================================
# Public API
# ===========================================================================


def file_reader(filename, lazy=False, use_uniform_signal_axis=True):
    """
    Read mass spectrometry data from a Balzers/Pfeiffer Quadstar SAC or
    SBC file.

    For SAC files, each trace (channel) is returned as a separate signal
    dictionary.  For SBC files, a single signal dictionary is returned
    containing the scan bargraph intensities.

    The signal axis corresponds to the mass-to-charge ratio (M/Z) and
    the navigation axis represents sequential measurement cycles
    (timesteps).  When only a single timestep is present the data is
    returned as a 1-D spectrum.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axis : bool, default=True
        If `True`, the time navigation axis is returned as a uniform axis
        (`offset` + `scale`). If `False`, the time navigation axis is returned
        as an explicit non-uniform `axis` array using per-cycle timestamps.

    %s
    """
    if lazy:
        raise NotImplementedError("Lazy loading is not supported.")

    with open(filename, "rb") as f:
        buf = f.read()

    gen = _read_general_header(buf)
    base_datetime = _build_datetime(gen)

    if str(filename).lower().endswith(".sbc"):
        return _read_sbc_file(
            filename,
            buf,
            gen,
            base_datetime,
            use_uniform_signal_axis=use_uniform_signal_axis,
        )

    return _read_sac_file(
        filename,
        buf,
        gen,
        base_datetime,
        use_uniform_signal_axis=use_uniform_signal_axis,
    )


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_UNSUPPORTED_DOC,
    RETURNS_DOC,
)
