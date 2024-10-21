import gc
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from rsciio.ripple import _api as ripple

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
exspy = pytest.importorskip("exspy", reason="exspy not installed")


# Tuple of tuples (data shape, signal_dimensions)
SHAPES_SDIM = (
    ((3,), (1,)),
    ((2, 3), (1, 2)),
    ((2, 3, 4), (1, 2)),
)

TEST_DATA_PATH = Path(__file__).parent / "data" / "ripple"


def test_write_unsupported_data_shape():
    data = np.arange(5 * 10 * 15 * 20).reshape((5, 10, 15, 20))
    s = hs.signals.Signal1D(data)
    with pytest.raises(TypeError):
        s.save("test_write_unsupported_data_shape.rpl")


def test_write_unsupported_data_type():
    data = np.arange(5 * 10 * 15).reshape((5, 10, 15)).astype(np.float16)
    s = hs.signals.Signal1D(data)
    with pytest.raises(IOError):
        s.save("test_write_unsupported_data_type.rpl")


# Test failing
# def test_write_scalar():
#    data = np.array([2])
#    with tempfile.TemporaryDirectory() as tmpdir:
#        s = hs.signals.BaseSignal(data)
#        fname = os.path.join(tmpdir, 'test_write_scalar_data.rpl')
#        s.save(fname)
#        s2 = hs.load(fname)
#        np.testing.assert_allclose(s.data, s2.data)


def test_write_without_metadata(tmp_path):
    data = np.arange(5 * 10 * 15).reshape((5, 10, 15))
    s = hs.signals.Signal1D(data)
    fname = tmp_path / "test_write_without_metadata.rpl"
    s.save(fname)
    s2 = hs.load(fname)
    np.testing.assert_allclose(s.data, s2.data)
    # for windows
    del s2
    gc.collect()


def test_write_with_metadata(tmp_path):
    data = np.arange(5 * 10).reshape((5, 10))
    s = hs.signals.Signal1D(data)
    s.metadata.set_item("General.date", "2016-08-06")
    s.metadata.set_item("General.time", "10:55:00")
    s.metadata.set_item("General.title", "Test title")
    fname = tmp_path / "test_write_with_metadata.rpl"
    s.save(fname)
    s2 = hs.load(fname)
    np.testing.assert_allclose(s.data, s2.data)
    assert s.metadata.General.date == s2.metadata.General.date
    assert s.metadata.General.title == s2.metadata.General.title
    assert s.metadata.General.time == s2.metadata.General.time
    del s2
    gc.collect()


def generate_parameters():
    parameters = []
    for dtype in ripple.dtype2keys.keys():
        for shape, dims in SHAPES_SDIM:
            for dim in dims:
                for metadata in [True, False]:
                    parameters.append(
                        {
                            "dtype": dtype,
                            "shape": shape,
                            "dim": dim,
                            "metadata": metadata,
                        }
                    )
    return parameters


def _get_filename(s, metadata):
    filename = "test_ripple_sdim-%i_ndim-%i_%s%s.rpl" % (
        s.axes_manager.signal_dimension,
        s.axes_manager.navigation_dimension,
        s.data.dtype.name,
        "_meta" if metadata else "",
    )
    return filename


def _create_signal(shape, dim, dtype, metadata):
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if dim == 1:
        if len(shape) > 2:
            s = exspy.signals.EELSSpectrum(data)
            if metadata:
                s.set_microscope_parameters(
                    beam_energy=100.0, convergence_angle=1.0, collection_angle=10.0
                )
        else:
            s = exspy.signals.EDSTEMSpectrum(data)
            if metadata:
                s.set_microscope_parameters(
                    beam_energy=100.0,
                    live_time=1.0,
                    tilt_stage=2.0,
                    azimuth_angle=3.0,
                    elevation_angle=4.0,
                    energy_resolution_MnKa=5.0,
                )
    else:
        s = hs.signals.BaseSignal(data).transpose(signal_axes=dim)
    if metadata:
        s.metadata.General.date = "2016-08-06"
        s.metadata.General.time = "10:55:00"
        s.metadata.General.title = "Test title"
    for i, axis in enumerate(s.axes_manager._axes):
        i += 1
        axis.offset = i * 0.5
        axis.scale = i * 100
        axis.name = "%i" % i
        if axis.navigate:
            axis.units = "m"
        else:
            axis.units = "eV"

    return s


@pytest.mark.parametrize("pdict", generate_parameters())
def test_data(pdict, tmp_path):
    dtype, shape, dim, metadata = (
        pdict["dtype"],
        pdict["shape"],
        pdict["dim"],
        pdict["metadata"],
    )
    s = _create_signal(shape=shape, dim=dim, dtype=dtype, metadata=metadata)
    filename = _get_filename(s, metadata)
    s.save(tmp_path / filename)
    s_just_saved = hs.load(tmp_path / filename)
    s_ref = hs.load(TEST_DATA_PATH / filename)
    try:
        for stest in (s_just_saved, s_ref):
            npt.assert_array_equal(s.data, stest.data)
            assert s.data.dtype == stest.data.dtype
            assert s.axes_manager.signal_shape == stest.axes_manager.signal_shape
            assert (
                s.axes_manager.navigation_shape == stest.axes_manager.navigation_shape
            )
            assert s.metadata.General.title == stest.metadata.General.title
            mdpaths = ("Signal.signal_type",)
            if s.metadata.Signal.signal_type == "EELS" and metadata:
                mdpaths += (
                    "Acquisition_instrument.TEM.convergence_angle",
                    "Acquisition_instrument.TEM.beam_energy",
                    "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
                )
            elif "EDS" in s.metadata.Signal.signal_type and metadata:
                mdpaths += (
                    "Acquisition_instrument.TEM.Stage.tilt_alpha",
                    "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                    "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                    "Acquisition_instrument.TEM.Detector." "EDS.energy_resolution_MnKa",
                    "Acquisition_instrument.TEM.Detector.EDS.live_time",
                )
            if metadata:
                mdpaths = (
                    "General.date",
                    "General.time",
                    "General.title",
                )
            for mdpath in mdpaths:
                assert s.metadata.get_item(mdpath) == stest.metadata.get_item(mdpath)
            for saxis, taxis in zip(s.axes_manager._axes, stest.axes_manager._axes):
                taxis.convert_to_units()
                assert saxis.scale == taxis.scale
                assert saxis.offset == taxis.offset
                assert saxis.units == taxis.units
                assert saxis.name == taxis.name
    except Exception:
        raise
    finally:
        # As of v0.8.5 the data in the ripple files are loaded as memmaps
        # instead of array. In Windows the garbage collector doesn't close
        # the file before attempting to delete it making the test fail.
        # The following lines simply make sure that the memmap is closed.
        # del s_just_saved.data
        # del s_ref.data
        del s_just_saved
        del s_ref
        gc.collect()


def generate_files():
    """Generate the test files that are distributed with HyperSpy.

    Unless new features are introduced there shouldn't be any need to recreate
    the files.

    """
    for dtype in ripple.dtype2keys.keys():
        for shape, dims in SHAPES_SDIM:
            for dim in dims:
                for metadata in [True, False]:
                    s = _create_signal(
                        shape=shape, dim=dim, dtype=dtype, metadata=metadata
                    )
                    filename = _get_filename(s, metadata)
                    s.save(TEST_DATA_PATH / filename, overwrite=True)
