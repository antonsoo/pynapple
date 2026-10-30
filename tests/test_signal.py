import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy.signal import hilbert

import pynapple as nap


@pytest.mark.parametrize(
    "param_name, invalid_value, exception",
    [
        (
            "data",
            None,
            pytest.raises(
                TypeError, match="`data` must be `Tsd`, got <class 'NoneType'>"
            ),
        ),
        (
            "data",
            "invalid_data",
            pytest.raises(TypeError, match="`data` must be `Tsd`, got <class 'str'>"),
        ),
        (
            "epochs",
            None,
            pytest.raises(
                TypeError,
                match="`epochs` must be `IntervalSet`, got <class 'NoneType'>",
            ),
        ),
        (
            "epochs",
            "invalid_epoch",
            pytest.raises(
                TypeError,
                match="`epochs` must be `IntervalSet`, got <class 'str'>",
            ),
        ),
        (
            "frequency_band",
            (10, "not_a_number"),
            pytest.raises(
                TypeError, match="`frequency_band` must contain numeric values"
            ),
        ),
        (
            "frequency_band",
            (10, 5),
            pytest.raises(
                ValueError,
                match=re.escape("`frequency_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "threshold_band",
            (1, "not_a_number"),
            pytest.raises(
                TypeError, match="`threshold_band` must contain numeric values"
            ),
        ),
        (
            "threshold_band",
            (10, 5),
            pytest.raises(
                ValueError,
                match=re.escape("`threshold_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "duration_band",
            (1, "not_a_number"),
            pytest.raises(
                TypeError, match="`duration_band` must contain numeric values"
            ),
        ),
        (
            "duration_band",
            (5,),
            pytest.raises(
                ValueError, match="`duration_band` must have length 2, got 1"
            ),
        ),
        (
            "duration_band",
            (5, 1),
            pytest.raises(
                ValueError,
                match=re.escape("`duration_band` must be (min, max) with min < max"),
            ),
        ),
        (
            "min_interval",
            "string",
            pytest.raises(TypeError, match="`min_interval` must be a number"),
        ),
        (
            "min_interval",
            -0.1,
            pytest.raises(ValueError, match="`min_interval` must be >= 0"),
        ),
        (
            "fs",
            "string",
            pytest.raises(TypeError, match="`fs` must be a number or None"),
        ),
        (
            "fs",
            -1,
            pytest.raises(ValueError, match="`fs` must be > 0"),
        ),
        (
            "sliding_window_size",
            "string",
            pytest.raises(TypeError, match="`sliding_window_size` must be an integer"),
        ),
        (
            "sliding_window_size",
            -5,
            pytest.raises(ValueError, match="`sliding_window_size` must be > 0"),
        ),
        (
            "sliding_window_size",
            0,
            pytest.raises(ValueError, match="`sliding_window_size` must be > 0"),
        ),
        # Valid cases (does not raise exceptions)
        (
            "data",
            nap.Tsd(t=np.linspace(0, 5, 100), d=np.sin(np.linspace(0, 5, 100))),
            does_not_raise(),
        ),
        ("epochs", nap.IntervalSet(start=0, end=5), does_not_raise()),
        ("frequency_band", (10, 30), does_not_raise()),
        ("threshold_band", (1, 10), does_not_raise()),
        ("duration_band", (0.1, 2), does_not_raise()),
        ("min_interval", 0.02, does_not_raise()),
        ("fs", 1000, does_not_raise()),
        ("fs", None, does_not_raise()),
        ("sliding_window_size", 51, does_not_raise()),
    ],
)
def test_detect_oscillatory_events_input_types(param_name, invalid_value, exception):
    # Create some valid input values for other parameters
    duration = 5
    fs = 1000
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 25 * t)
    ts = nap.Tsd(t=t, d=signal)
    epoch = nap.IntervalSet(start=0, end=duration)
    min_dur = 0.1
    max_dur = 2
    min_inter = 0.02
    frequency_band = (10, 30)
    threshold_band = (1, 10)

    # Modify the parameter based on the test case
    kwargs = {
        "data": ts,
        "epochs": epoch,
        "frequency_band": frequency_band,
        "threshold_band": threshold_band,
        "duration_band": (min_dur, max_dur),
        "min_interval": min_inter,
        "fs": fs,
        "sliding_window_size": 51,
    }
    kwargs[param_name] = invalid_value

    with exception:
        nap.detect_oscillatory_events(**kwargs)


@pytest.mark.parametrize(
    "frequency_band, threshold_band, num_events, start, end",
    [
        ((10, 30), (1, 10), 1, 0, 2),
        ((40, 60), (1, 10), 1, 3, 5),
        ((100, 150), (1, 10), 0, None, None),
    ],
)
def test_detect_oscillatory_events(
    frequency_band, threshold_band, num_events, start, end
):
    fs = 1000
    duration = 5
    min_dur = 0.1
    max_dur = 2
    min_inter = 0.02

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    # 25 Hz oscillation from 0-2s
    freq_1 = 25
    mask1 = (t >= 0) & (t < 2)
    signal[mask1] = np.sin(2 * np.pi * freq_1 * t[mask1])

    # 50 Hz oscillation from 3-5s
    freq_2 = 50
    mask2 = (t >= 3) & (t < 5)
    signal[mask2] = np.sin(2 * np.pi * freq_2 * t[mask2])

    ts = nap.Tsd(t=t, d=signal)
    epochs = nap.IntervalSet(start=0, end=duration)
    osc_ep = nap.detect_oscillatory_events(
        ts,
        epochs,
        frequency_band,
        threshold_band,
        (min_dur, max_dur),
        min_inter,
    )

    assert len(osc_ep) == num_events  # Only one event in given frequency_band

    if num_events > 0:
        # Start and end should be close to actuals +/- a small amount
        detected_start = osc_ep.start[0]
        detected_end = osc_ep.end[0]
        assert np.isclose(start, detected_start, atol=0.05)
        assert np.isclose(end, detected_end, atol=0.05)

        # Check we store power, amplitude, and peak_time
        for key in ["power", "amplitude", "peak_time"]:
            assert key in osc_ep._metadata

        # Check peak_time is within the interval
        peak_time = osc_ep._metadata["peak_time"][0]
        assert start <= peak_time <= end


@pytest.mark.parametrize(
    "func",
    [
        nap.apply_hilbert_transform,
        nap.compute_hilbert_envelope,
        nap.compute_hilbert_phase,
    ],
)
@pytest.mark.parametrize(
    "input, expectation",
    [
        (nap.Tsd(d=np.ones(3), t=[1, 2, 3]), does_not_raise()),
        (nap.TsdFrame(d=np.ones((3, 3)), t=[1, 2, 3]), does_not_raise()),
        (
            np.ones((3, 3)),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            nap.TsdTensor(d=np.ones((3, 3, 3)), t=[1, 2, 3]),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            [],
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            nap.IntervalSet(1, 2),
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
        (
            None,
            pytest.raises(TypeError, match="data should be a Tsd or TsdFrame."),
        ),
    ],
)
def test_hilbert_type_errors(input, func, expectation):
    with expectation:
        func(input)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
                axis=1,
            ),
        ),
    ],
)
def test_apply_hilbert_transform(data):
    result = nap.apply_hilbert_transform(data)
    expected = hilbert(data.values, axis=0)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
                axis=1,
            ),
        ),
    ],
)
def test_compute_hilbert_phase(data):
    result = nap.compute_hilbert_phase(data)
    analytic_signal = hilbert(data.values, axis=0)
    phase = np.angle(analytic_signal)
    expected = np.mod(np.unwrap(phase), 2 * np.pi)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize(
    "data",
    [
        nap.Tsd(t=np.linspace(0, 1, 500), d=np.sin(np.linspace(0, 1, 500))),
        nap.TsdFrame(
            t=np.linspace(0, 1, 500),
            d=np.stack(
                [
                    np.sin(np.linspace(0, 1, 500)),
                    np.cos(np.linspace(0, 1, 500)),
                ],
                axis=1,
            ),
        ),
    ],
)
def test_compute_hilbert_envelope(data):
    result = nap.compute_hilbert_envelope(data)
    analytic_signal = hilbert(data.values, axis=0)
    expected = np.abs(analytic_signal)

    assert isinstance(result, type(data))
    np.testing.assert_array_equal(data.time_support, result.time_support)
    np.testing.assert_array_equal(data.times(), result.times())

    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize(
    "func",
    [
        nap.apply_hilbert_transform,
        nap.compute_hilbert_envelope,
        nap.compute_hilbert_phase,
    ],
)
def test_hilbert_epochs_are_independent(frame, func):
    times = np.concatenate([np.arange(13) / 10, 10 + np.arange(17) / 10])
    values = np.concatenate([np.sin(np.arange(13)), np.zeros(17)])
    epochs = nap.IntervalSet([0, 10], [1.2, 11.6])
    if frame:
        data = nap.TsdFrame(
            t=times,
            d=np.column_stack([values, 2 * values]),
            columns=["first", "second"],
            time_support=epochs,
        )
    else:
        data = nap.Tsd(t=times, d=values, time_support=epochs)
    result = func(data)
    for epoch in epochs:
        # Separately processing an epoch must not depend on another epoch.
        expected = func(data.restrict(epoch))
        np.testing.assert_allclose(result.restrict(epoch).values, expected.values)
    np.testing.assert_array_equal(result.values[13:], 0)
    np.testing.assert_array_equal(result.times(), data.times())
    np.testing.assert_array_equal(result.time_support, epochs)
    if frame:
        np.testing.assert_array_equal(result.columns, data.columns)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_hilbert_padded_epoch_reference(dtype):
    # The first epoch needs padding from 13 to 14 samples; the second is
    # already a fast length. A support interval without samples is retained.
    times = np.concatenate([np.arange(13), 30 + np.arange(16)])
    values = np.arange(58, dtype=dtype).reshape(29, 2)[:, ::-1]
    epochs = nap.IntervalSet([0, 20, 30], [12, 21, 45])
    data = nap.TsdFrame(t=times, d=values, time_support=epochs, columns=["a", "b"])
    data.set_info(location=["left", "right"])
    # Independent discrete Fourier definition of the analytic signal.
    expected = []
    for chunk, length in [(values[:13], 14), (values[13:], 16)]:
        frequencies = np.arange(length)
        samples = np.arange(length)
        matrix = np.exp(-2j * np.pi * frequencies[:, None] * samples / length)
        padded = np.pad(chunk, ((0, length - len(chunk)), (0, 0)))
        spectrum = matrix @ padded
        multiplier = np.zeros(length)
        multiplier[0] = multiplier[length // 2] = 1
        multiplier[1 : length // 2] = 2
        expected.append(
            (matrix.conj().T @ (spectrum * multiplier[:, None]) / length)[: len(chunk)]
        )
    result = nap.apply_hilbert_transform(data)
    np.testing.assert_allclose(
        result.values, np.concatenate(expected), rtol=1e-5, atol=1e-5
    )
    assert result.dtype == hilbert(values, axis=0).dtype
    np.testing.assert_array_equal(result.time_support, epochs)
    np.testing.assert_array_equal(result.location, data.location)


def test_hilbert_without_scipy_fft(monkeypatch):
    import sys

    from scipy.fftpack import next_fast_len

    data = nap.Tsd(t=np.arange(13), d=np.arange(13, dtype=float))
    expected = hilbert(data.values, N=next_fast_len(len(data)))[: len(data)]
    monkeypatch.setitem(sys.modules, "scipy.fft", None)
    result = nap.apply_hilbert_transform(data)
    np.testing.assert_array_equal(result.values, expected)
