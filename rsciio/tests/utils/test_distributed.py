import pytest

from rsciio.utils._distributed import get_chunk_slice


@pytest.mark.parametrize(
    "shape", ((10, 20, 30, 512, 512), (20, 30, 512, 512), (10, 512, 512), (512, 512))
)
def test_get_chunk_slice(shape):
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=-1)  # 1 chunk
    assert chunk_arr.shape == (1,) * len(shape) + (len(shape), 2)
    assert chunk == tuple([(i,) for i in shape])

    chunks = (1,) * (len(shape) - 2) + (-1, -1)
    # Everything is 1 chunk
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=chunks)
    assert chunk_arr.shape == shape[:-2] + (1, 1) + (len(shape), 2)
    assert chunk == (
        tuple([(1,) * i for i in shape[:-2]]) + tuple([(i,) for i in shape[-2:]])
    )
