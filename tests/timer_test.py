"""Tests for timing and timing-log utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from genslm_embeddings.timer import timeit_decorator
from genslm_embeddings.timer import TimeLogger
from genslm_embeddings.timer import Timer
from genslm_embeddings.timer import TimeStats


def test_timer_records_elapsed_time_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance_times = iter([1_000_000_000, 1_250_000_000])
    unix_times = iter([100.0, 100.25])
    logged: list[TimeStats] = []
    monkeypatch.setattr(
        'genslm_embeddings.timer.time.perf_counter_ns',
        lambda: next(performance_times),
    )
    monkeypatch.setattr(
        'genslm_embeddings.timer.time.time',
        lambda: next(unix_times),
    )
    monkeypatch.setattr(TimeLogger, 'log', lambda self, ts: logged.append(ts))

    with (
        Timer('embedding', '2') as timer,
        pytest.raises(RuntimeError, match='still running'),
    ):
        _ = timer.elapsed_s

    assert timer.elapsed_ns == 250_000_000
    assert timer.elapsed_ms == 250.0
    assert timer.elapsed_s == 0.25
    assert logged == [TimeStats(('embedding', '2'), 0.25, 100.0, 100.25)]


def test_timeit_decorator_preserves_function_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[TimeStats] = []
    monkeypatch.setattr(TimeLogger, 'log', lambda self, ts: logged.append(ts))

    @timeit_decorator('batch')
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5
    assert add.__name__ == 'add'
    assert logged[0].tags == ('batch', 'add')


def test_time_logger_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = TimeLogger()
    expected = TimeStats(('embed', 'batch-1'), 1.25, 10.0, 11.25)
    logger.log(expected)
    output = capsys.readouterr().out
    log_path = tmp_path / 'timings.log'
    log_path.write_text(f'prefix {output}')

    parsed = logger.parse_logs(log_path)
    assert list(parsed[0].tags) == list(expected.tags)
    assert float(parsed[0].elapsed_s) == expected.elapsed_s
    assert float(parsed[0].start_unix) == expected.start_unix
    assert float(parsed[0].end_unix) == expected.end_unix
