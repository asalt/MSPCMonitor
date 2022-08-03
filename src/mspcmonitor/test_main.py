# test_main.py
from pathlib import Path
from re import M
import pytest
from .test_fixtures import sqlengine
from . import models

from sqlmodel import Session


def get_db(engine=sqlengine):
    db = Session(engine)
    crud.create_tables(db)
    try:
        return db
    finally:
        db.close()


def create_dummy_rawfiles(tmp_path: pytest.TempPathFactory):

    NFILES = 0
    NRAWDIRS = 2
    raw_file_paths = [tmp_path.mktemp("raw") for _ in range(NRAWDIRS)]
    for raw_file_path in raw_file_paths:
        (raw_file_path / "11111_1_1.raw").touch()
        (raw_file_path / "11111_2_1.raw").touch()
        (raw_file_path / "11112_1_1.raw").touch()
        NFILES += 1

    return tmp_path


@pytest.fixture(scope="session")
def path_with_rawfiles(tmp_path_factory) -> pytest.TempPathFactory:
    res = create_dummy_rawfiles(tmp_path_factory)
    return res


def test_monitor_for_rawfiles_path_fixture(sqlengine, path_with_rawfiles):
    """
    an integration test
    we need a mock database and filesystem
        we need rawfiles in our filesystem
        we test the routine for:
            1) finding a rawfile
            2) calculating interesting metrics
            3) updating the database
    """
    tmp_path = path_with_rawfiles.getbasetemp()
    globres = Path(tmp_path).glob("**/*.raw")
    globres = [x for x in globres]

    # sorted(Path(tmp_path).glob('**/*raw'))

    assert len(globres) == 6  # TODO keep track of this better
    # models.monitor_for_rawfiles()


# from . import main
# from . import main

# from .main import monitor_for_rawfiles
from . import dbutils
from .dbutils import monitor_for_rawfiles
from . import crud


def test_create_exprun(sqlengine):
    recno = 123
    exp = crud.create_experiment(get_db(sqlengine), models.Experiment(recno=recno))
    exprun = crud.create_experimentrun(
        get_db(sqlengine),
        models.ExperimentRun(runno=1, searchno=1),
        recno=recno,
    )
    # crud.get_all_experiments(get_db(sqlengine))
    # crud.get_all_experimentruns(get_db(sqlengine))
    assert len(crud.get_all_experimentruns(get_db(sqlengine))) == 1


def test_monitor_for_rawfiles(sqlengine, path_with_rawfiles):

    # dbutils.engine = sqlengine  # monkey patch

    # probably a more unified way to do this

    tmp_path = path_with_rawfiles.getbasetemp()

    exp = crud.create_exp(get_db(sqlengine), models.Experiment(recno=11111))
    exp = crud.create_exp(get_db(sqlengine), models.Experiment(recno=11112))
    # with get_db(sqlengine) as db:
    for recno in (11111, 11112):
        exp = crud.create_exp(get_db(sqlengine), models.Experiment(recno=recno))
        for rno in (1, 2):
            crud.create_experimentrun(
                get_db(sqlengine),
                models.ExperimentRun(runno=rno, searchno=1),
                recno=recno,
            )

    # res = crud.get_rawfiles(session)
    res = crud.get_rawfiles(get_db(sqlengine))
    monitor_results = monitor_for_rawfiles(tmp_path, sqlengine)
    res = crud.get_rawfiles(get_db(sqlengine))
    assert len(res) > 0
