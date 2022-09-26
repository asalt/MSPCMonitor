import pytest
from pathlib import Path

import pandas as pd
import os
import subprocess

#pytest.f

# def test_installed():
#     out = subprocess.run(["mspcmonitor", "--help"])
#     assert out.returncode == 0


def test_create():
    out = subprocess.run(["mspcmonitor", "db", "create", "--force"])
    assert out.returncode == 0
    assert os.path.exists('iSPEC.db') == True

def test_metadata():
    cmd = ["mspcmonitor", "db", "add-experiments-table", "../data/20221904_ispec_cur_Experiments_metadata_short.csv"]
    out = subprocess.run(cmd)
    assert out.returncode == 0
    # expru
    cmd = ["mspcmonitor", "db", "add-experimentruns-table", "../data/20222404_ispec_cur_ExperimentRuns_metadata_export_short.csv"]
    out = subprocess.run(cmd)
    assert out.returncode == 0