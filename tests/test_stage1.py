"""Stage 1 validation, per docs/mesh_partitioning_test_spec_v3.md."""

from __future__ import annotations

import numpy as np
import pytest

from meshpartition.stage0 import ingest
from meshpartition.stage1 import triage

from fixtures import fixture_a_sphere, fixture_c_shirt, fixture_c_shirt_two_torsos


def test_1_1_classification():
    """1.1 -- the torso is Major, buttons are Satellite."""
    working, _ = ingest(fixture_c_shirt())
    result = triage(working, n_pieces=5)

    assert len(result.major_ids) == 1  # the torso
    assert len(result.satellite_ids) == 3  # three buttons

    torso = result.major_ids[0]
    for satellite in result.satellite_ids:
        assert result.component_areas[satellite] < result.component_areas[torso]
        assert result.satellite_host[satellite] == torso


def test_1_2_allocation_math():
    """1.2 -- a single major component with N=5 receives exactly 5 pieces."""
    working, _ = ingest(fixture_a_sphere())
    result = triage(working, n_pieces=5)

    assert len(result.major_ids) == 1
    only_major = result.major_ids[0]
    assert result.piece_counts[only_major] == 5
    assert sum(result.piece_counts.values()) == 5


def test_1_3_infeasibility_catch():
    """1.3 -- requesting fewer pieces than major components raises an error."""
    working, _ = ingest(fixture_c_shirt_two_torsos())

    with pytest.raises(ValueError, match="too low"):
        triage(working, n_pieces=1)
