# test_models_and_importers.py
# starting with the tutorial on
# https://sqlmodel.tiangolo.com/tutorial/fastapi/tests/
import pytest
from pathlib import Path

import pandas as pd

# from fastapi.testclient import TestClient
import sqlalchemy
from mspcmonitor import importers, crud
import sqlmodel
from sqlmodel import Session, SQLModel, create_engine
from io import StringIO
from . import importers, models
from .importers import Experiments_Importer, get_db

# tmp_path = Path("PYTEST_TMPDIR/test_create_file0")


# return (raw1, raw2)


@pytest.fixture
def sqlengine():

    from sqlmodel import create_engine, SQLModel

    SQLALCHEMY_DATABASE_URL = "sqlite://"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, echo=True
    )
    SQLModel.metadata.create_all(engine)
    yield engine


@pytest.fixture
def e2g_qual():
    f = Path(__file__).parent.joinpath("testdata/test_e2g_qual.tsv")
    return f


@pytest.fixture
def e2g_quant():
    f = Path(__file__).parent.joinpath("testdata/test_e2g_quant.tsv")
    return f


@pytest.fixture
def ispec_exp_export_table() -> StringIO:

    importer = Experiments_Importer(datafile=ispec_exp_export_table, engine=sqlengine)
    ispec_import_headers = "\t".join(
        Experiments_Importer.model.Config.schema_extra["ispec_column_mapping"].values()
    )
    # just write 1 line of meaningless data
    file = StringIO(f"{ispec_import_headers}\n99999\t1\t\6")
    file.seek(0)
    return file


def test_get_fixture(sqlengine):
    "receive a fixture check if it is expected object"
    assert isinstance(sqlengine, sqlalchemy.engine.Engine)
    # assert type(sqlengine) == type(sqlmodel.engine)


def test_experiments_importer_import1(sqlengine, ispec_exp_export_table):

    Experiments_Importer = importers.Experiments_Importer
    importer = Experiments_Importer(datafile=ispec_exp_export_table, engine=sqlengine)
    importer.insert_data()
    # turn this into a test
    with get_db(sqlengine) as db:
        exps = db.exec("select * from experiment").fetchall()
        assert len(exps) == 1
        exp = exps[0]
    assert exp.recno == 99999
    # print()
    # print(f"""experiments in database:
    # {e for e in exps}
    # \n
    # """
    # )


# def test_rawfile_importer_case1(sqlengine):
#    importers.RawFileImporter()


# def test_crud_get_all_exps(sqlengine):
#    with get_db(sqlengine) as db:
#        exps = db.exec("select * from experiment").fetchall()
#    # print(crud.get_all_experiments(db))


def test_e2g_quan(sqlengine):
    pass


def test_crud_add_exprun_before_exp(sqlengine):
    model = models.ExperimentRun(recno=99999, runno=643, searchno=6)
    with pytest.raises(AttributeError):
        crud.create_experimentrun(get_db(sqlengine), model, recno=99999)


def test_crud_add_exprun_after_exp(sqlengine):
    exp = models.Experiment(recno=99999)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(recno=99999, runno=643, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99999)


# snapshot test


def test_first_snapshot(snapshot):
    snapshot.assert_match(sqlmodel.engine)


# todo fix
def _test_e2g_qual(sqlengine, e2g_qual):
    exp = models.Experiment(recno=99999)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=643, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99999)

    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    res = importer.insert_data()
    with get_db(sqlengine) as db:
        e2gqual = db.exec("select * from e2gqual").fetchall()
        assert len(e2gqual) != 0


def test_e2g_qual_wrongfile(sqlengine, e2g_quant):
    importer = importers.E2G_QUAL_Importer(e2g_quant, sqlengine)
    res = importer.insert_data()
    with get_db(sqlengine) as db:
        e2gqual = db.exec("select * from e2gqual").fetchall()
        assert len(e2gqual) == 0


def test_simple(sqlengine):

    """
    tests
    importers.get_ispec_exp_export
    split up
    """

    # test_engine = importers._make_inmemory_db_and_get_engine()
    test_engine = sqlengine
    file = importers.get_ispec_exp_export()
    print(file)
    file.seek(0)
    Experiments_Importer = importers.Experiments_Importer
    get_db = importers.get_db

    importer = importers.Experiments_Importer(datafile=file, engine=test_engine)
    importer.insert_data()
    # turn this into a test
    with get_db(test_engine) as db:
        exps = db.exec("select * from experiment").fetchall()
        print()
        print(
            f"""experiments in database:
        {e for e in exps}
        \n
        """
        )

    print("====\nclose the db, now open and query\n")
    print(crud.get_all_experiments(db))

    print("====\nclose the db, now open and query\n")
    # turn this into a test
    file.seek(0)
    importer.insert_data(
        before_db_close_func=lambda db: print(crud.get_all_experiments(db))
    )
    with get_db(importer.engine) as db:
        print(crud.get_all_experiments(db))

    print(crud.get_all_experiments(db))


if __name__ == "__main__":
    pass
    # test_simple()
#
# this is all from the tutorial

# from .main import app, get_session  #
# from mspcmonitor.database import (
#     engine,
# )  # this is the production database connection engine
#
# # redefine engine
# engine = create_engine(":::memory:::", echo=True)
#
#
# def test_create_hero():
#     # Some code here omitted, we will see it later 👈
#     client = TestClient(app)  #
#
#     response = client.post(  #
#         "/heroes/", json={"name": "Deadpond", "secret_name": "Dive Wilson"}
#     )
#     # Some code here omitted, we will see it later 👈
#     data = response.json()  #
#
#     assert response.status_code == 200  #
#     assert data["name"] == "Deadpond"  #
#     assert data["secret_name"] == "Dive Wilson"  #
#     assert data["age"] is None  #
#     assert data["id"] is not None  #