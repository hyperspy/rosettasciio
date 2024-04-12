# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import importlib
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version

imageio = pytest.importorskip("imageio")

from rsciio.image import file_writer  # noqa: E402

testfile_dir = (Path(__file__).parent / "data" / "image").resolve()


@pytest.mark.skipif(
    Version(imageio.__version__) < Version("2.23"),
    reason="needs imageio >=2.23",
)
@pytest.mark.parametrize(("dtype"), ["uint8", "int32", bool])
@pytest.mark.parametrize(("ext"), ["png", "bmp", "gif", "jpg"])
def test_save_load_cycle_grayscale(dtype, ext, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    s = hs.signals.Signal2D(np.arange(128 * 128).reshape(128, 128).astype(dtype))

    if dtype == "int32" and ext in ["bmp", "jpg"]:
        # BMP and JPG does not support uint32.
        return
    print(f"Saving-loading cycle for the extension `{ext}` with dtype `{dtype}`")
    filename = tmp_path / f"test_image.{ext}"
    s.save(filename)
    hs.load(filename)


@pytest.mark.parametrize(("color"), ["rgb8", "rgba8"])
@pytest.mark.parametrize(("ext"), ["png", "bmp", "gif", "jpeg"])
def test_save_load_cycle_color(color, ext, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    dim = 4 if "rgba" in color else 3
    dtype = "uint8" if "8" in color else "uint16"
    if dim == 4 and ext == "jpeg":
        # JPEG does not support alpha channel.
        return
    print("color:", color, "; dim:", dim, "; dtype:", dtype)
    s = hs.signals.Signal1D(
        np.arange(128 * 128 * dim).reshape(128, 128, dim).astype(dtype)
    )
    s.change_dtype(color)

    print("Saving-loading cycle for the extension:", ext)
    filename = tmp_path / f"test_image.{ext}"
    s.save(filename)
    hs.load(filename)


@pytest.mark.skipif(
    Version(imageio.__version__) < Version("2.23"),
    reason="needs imageio >=2.23",
)
@pytest.mark.parametrize(("dtype"), ["uint8", "int32"])
@pytest.mark.parametrize(("ext"), ["png", "bmp", "gif", "jpg"])
def test_save_load_cycle_kwds(dtype, ext, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    s = hs.signals.Signal2D(np.arange(128 * 128).reshape(128, 128).astype(dtype))

    if dtype == "int32" and ext in ["bmp", "jpg"]:
        # BMP and JPG does not support uint32.
        return

    print(f"Saving-loading cycle for the extension `{ext}` with dtype `{dtype}`")
    filename = tmp_path / f"test_image.{ext}"
    if ext == "png":
        kwds = {"optimize": True}
    elif ext == "jpg":
        kwds = {"quality": 100, "optimize": True}
    elif ext == "gif":
        kwds = {"subrectangles": "True", "palettesize": 128}
    else:
        kwds = {}
    s.save(filename, **kwds)
    hs.load(filename, pilmode="L", as_grey=True)


@pytest.mark.skipif(
    Version(imageio.__version__) < Version("2.23"),
    reason="needs imageio >=2.23",
)
@pytest.mark.parametrize(("ext"), ["png", "bmp", "gif", "jpg"])
def test_export_scalebar(ext, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pytest.importorskip("matplotlib_scalebar")
    # Use np.uint8 to be able to save as BMP
    pixels = 64
    data = np.arange(pixels**2).reshape((pixels, pixels)).astype(np.uint8)
    s = hs.signals.Signal2D(data)
    s.axes_manager[0].units = "nm"
    s.axes_manager[1].units = "nm"

    filename = tmp_path / f"test_scalebar_export.{ext}"
    if ext in ["bmp", "gif"]:
        with pytest.raises(ValueError):
            s.save(filename, scalebar=True)
        with pytest.raises(ValueError):
            s.save(filename, output_size=512)
        s.save(filename)
    else:
        s.save(filename, scalebar=True)
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


@pytest.mark.parametrize(("units"), ["1/nm", "1 / nm", "1 / nanometer", "1/nanometer"])
def test_export_scalebar_reciprocal(tmp_path, units):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = 512
    s = hs.signals.Signal2D(
        np.arange(pixels**2).reshape((pixels, pixels)).astype("int32")
    )
    for axis in s.axes_manager.signal_axes:
        axis.units = units
        axis.scale = 0.1

    filename = tmp_path / "test_scalebar_export.png"
    s.save(filename, scalebar=True, scalebar_kwds={"location": "lower right"})
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


def test_export_scalebar_undefined_units(tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = 512
    s = hs.signals.Signal2D(
        np.arange(pixels**2).reshape((pixels, pixels)).astype("int32")
    )

    filename = tmp_path / "test_scalebar_export.png"
    s.save(filename, scalebar=True, scalebar_kwds={"location": "lower right"})
    s_reload = hs.load(filename)
    assert s.data.shape == s_reload.data.shape


def test_non_uniform(tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].convert_to_non_uniform_axis()

    filename = tmp_path / "test_export_size.jpg"
    with pytest.raises(TypeError):
        s.save(filename)


def test_export_scalebar_different_scale_units(tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pytest.importorskip("matplotlib_scalebar")
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].scale = 2

    filename = tmp_path / "test_export_size.jpg"
    with pytest.raises(ValueError):
        s.save(filename, scalebar=True)

    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))
    s.axes_manager[0].units = "nm"

    filename = tmp_path / "test_export_size.jpg"
    with pytest.raises(ValueError):
        s.save(filename, scalebar=True)


@pytest.mark.parametrize("output_size", (512, [512, 512]))
def test_export_output_size(output_size, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = 16
    s = hs.signals.Signal2D(np.arange(pixels**2).reshape((pixels, pixels)))

    fname = tmp_path / "test_export_size.jpg"
    s.save(fname, scalebar=True, output_size=output_size)
    s_reload = hs.load(fname)
    assert s_reload.data.shape == (512, 512)


@pytest.mark.parametrize("scalebar", [True, False])
@pytest.mark.parametrize("output_size", (None, 512, (512, 512)))
def test_export_output_size_non_square(output_size, tmp_path, scalebar):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = (8, 16)
    s = hs.signals.Signal2D(
        np.arange(np.multiply(*pixels), dtype=np.uint8).reshape(pixels)
    )

    fname = tmp_path / "test_export_size_non_square.jpg"
    s.save(fname, output_size=output_size, scalebar=scalebar)
    s_reload = hs.load(fname)

    if output_size is None:
        output_size = (8, 16)
    if isinstance(output_size, int):
        output_size = (output_size * np.divide(*pixels), output_size)

    assert s_reload.data.shape == output_size


@pytest.mark.parametrize("output_size", (None, 512))
@pytest.mark.parametrize("aspect", (1, 0.5))
def test_export_output_size_aspect(aspect, output_size, tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = (256, 256)
    s = hs.signals.Signal2D(np.arange(np.multiply(*pixels)).reshape(pixels))

    fname = tmp_path / "test_export_size_non_square_aspect.jpg"
    s.save(
        fname, scalebar=True, output_size=output_size, imshow_kwds=dict(aspect=aspect)
    )
    s_reload = hs.load(fname)

    if output_size is None:
        output_size = s.data.shape[0]
    assert s_reload.data.shape == (output_size * aspect, output_size)


def test_save_image_navigation(tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = 16
    s = hs.signals.Signal2D(
        np.arange(pixels**2).reshape((pixels, pixels)).astype("int32")
    )

    fname = tmp_path / "test_save_image_navigation.png"
    s.T.save(fname, scalebar=True)


def test_error_library_no_installed(tmp_path):
    axis = {
        "_type": "UniformDataAxis",
        "name": None,
        "units": None,
        "navigate": False,
        "is_binned": False,
        "size": 128,
        "scale": 1.0,
        "offset": 0.0,
    }
    signal_dict = {"data": np.arange(128 * 128).reshape(128, 128), "axes": [axis, axis]}

    matplotlib = importlib.util.find_spec("matplotlib")
    if matplotlib is None:
        # When matplotlib is not installed, raises an error to inform user
        # that matplotlib is necessary
        with pytest.raises(ValueError):
            file_writer(tmp_path / "test_image_error.jpg", signal_dict, output_size=64)

        with pytest.raises(ValueError):
            file_writer(
                tmp_path / "test_image_error.jpg", signal_dict, imshow_kwds={"a": "b"}
            )


def test_renishaw_wire():
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    s = hs.load(testfile_dir / "renishaw_wire.jpg")
    assert s.data.shape == (480, 752)
    for axis, scale, offset, name in zip(
        s.axes_manager.signal_axes,
        [2.42207446, 2.503827],
        [19105.5, -6814.538],
        ["y", "x"],
    ):
        np.testing.assert_allclose(axis.scale, scale)
        np.testing.assert_allclose(axis.offset, offset)
        axis.name == name
        axis.units == "µm"


def test_export_output_size_iterable_length_1(tmp_path):
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    pixels = (256, 256)
    s = hs.signals.Signal2D(np.arange(np.multiply(*pixels)).reshape(pixels))

    fname = tmp_path / "test_export_output_size_iterable_length_1.jpg"
    with pytest.raises(ValueError):
        s.save(fname, output_size=(256,))
