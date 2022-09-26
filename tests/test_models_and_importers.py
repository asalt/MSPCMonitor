# test_models_and_importers.py
# starting with the tutorial on
# https://sqlmodel.tiangolo.com/tutorial/fastapi/tests/
import pytest
from pathlib import Path

import pandas as pd

# from fastapi.testclient import TestClient
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from mspcmonitor import importers, crud
import sqlmodel
from sqlmodel import Session, SQLModel, create_engine
from io import StringIO
from mspcmonitor import importers, models
from mspcmonitor.importers import Experiments_Importer, get_db

# tmp_path = Path("PYTEST_TMPDIR/test_create_file0")


# return (raw1, raw2)


@pytest.fixture(scope="function")
def sqlengine():

    from sqlmodel import create_engine, SQLModel

    # SQLALCHEMY_DATABASE_URL = "sqlite:///ispec.db"
    SQLALCHEMY_DATABASE_URL = "sqlite:///"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
    )

    # if not Path("ispec.db").exists() or True: # TODO separate teardown ?
    #     # import pdb; pdb.set_trace()
    #     SQLModel.metadata.create_all(engine)
    # crud.create_tables(get_db(SQLALCHEMY_DATABASE_URL))

    SQLModel.metadata.create_all(engine)
    yield engine


@pytest.fixture(scope="module")
def genetable():
    f = Path(__file__).parent.joinpath("testdata/genetable_short100.tsv")
    return f


@pytest.fixture(scope="module")
def e2g_qual():
    f = Path(__file__).parent.joinpath(
        "testdata/99995_426_6_labelnone_e2g_QUAL_short.tsv"
    )
    return f


@pytest.fixture(scope="module")
def psm_qual():
    f = Path(__file__).parent.joinpath(
        "testdata/99995_426_6_labelnone_psms_QUAL_short.tsv"
    )
    return f


@pytest.fixture(scope="module")
def psm_quant():
    f = Path(__file__).parent.joinpath(
        "testdata/99995_426_6_labelnone_0_psms_QUANT_short.tsv"
    )
    return f


@pytest.fixture(scope="module")
def e2g_quant():
    f = Path(__file__).parent.joinpath(
        "testdata/99995_426_6_labelnone_e2g_QUANT_short.tsv"
    )
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


# def test_first_snapshot(snapshot):
#     snapshot.assert_match(sqlmodel.engine)


def test_crud_exprun(sqlengine):
    exp = models.Experiment(recno=99998)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=643, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99998)
    # print(crud.get_all_experimentruns(get_db(sqlengine)))
    # print(crud.get_exprun_by_recrun(get_db(sqlengine), recno=99998,runno=643,searchno=6))
    res = crud.get_exprun_by_recrunsearch(
        get_db(sqlengine), recno=99998, runno=643, searchno=6
    )
    assert res is not None
    # assert len(res) == 1
    assert isinstance(res, models.ExperimentRun)
    assert res.experiment.recno == 99998
    assert res.runno == 643


def test_e2g_qual_without_exprecord(sqlengine, e2g_qual):

    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        e2gqual = db.exec("select * from e2gqual").fetchall()
        assert len(e2gqual) == 0


def test_import_genestable(sqlengine, genetable):

    # setup
    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()

    with get_db(sqlengine) as db:
        res = db.exec("select * from gene").fetchall()
        # import pdb; pdb.set_trace()
        assert len(res) > 0


def test_e2g_qual(sqlengine, e2g_qual, genetable):

    # make sure genes table is populated
    # otherwise no e2g data will be imported
    # importer = importers.GenesImporter(genetable, sqlengine)
    # importer.insert_data()

    # now make a rec and runno
    # TODO move all this to another fixture
    RECNO = 99995
    RUNNO = 426
    exp = models.Experiment(recno=RECNO)
    crud.create_experiment(get_db(sqlengine), exp)
    # crud.create_experiment(sqlengine, exp)
    model = models.ExperimentRun(runno=RUNNO, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=RECNO)
    # DONE

    #
    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()
    #

    # crud.create_experimentrun(sqlengine, model, recno=RECNO)
    # res0 = get_db(sqlengine).exec('select * from experimentrun').fetchall()
    # res = crud.get_exprun_by_recrun(get_db(sqlengine), recno=RECNO,runno=643,searchno=6)
    # test again, no particular reason
    # test_import_genestable(sqlengine, genetable)

    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        e2gqual = db.exec("select * from e2gqual").fetchall()
    assert len(e2gqual) != 0


def test_psm_qual(sqlengine, psm_qual, genetable, e2g_qual):

    # setup
    RECNO = 99995
    RUNNO = 426

    exp = models.Experiment(recno=RECNO)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=RUNNO, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99995)
    #
    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()
    # test_import_genestable(sqlengine, genetable)

    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()

    #
    importer = importers.PSM_QUAL_Importer(psm_qual, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        psmqual = db.exec("select * from psmqual").fetchall()
    assert len(psmqual) != 0


def test_psm_qual_genes_not_in_db(sqlengine, psm_qual, genetable, e2g_qual):
    """ """

    # setup
    RECNO = 99995
    RUNNO = 426

    exp = models.Experiment(recno=RECNO)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=RUNNO, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99995)
    #
    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()

    #
    importer = importers.PSM_QUAL_Importer(psm_qual, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        psmqual = db.exec("select * from psmqual").fetchall()
    assert len(psmqual) != 0


def test_psm_qual_wrongfile(sqlengine, psm_quant, genetable):
    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()
    importer = importers.PSM_QUAL_Importer(psm_quant, sqlengine)
    # no more raise inegrity error, instead just make nothing (and this is logged)
    # with pytest.raises(IntegrityError):
    #     importer.insert_data()
    # except IntegrityError: #expected
    #     pass
    with get_db(sqlengine) as db:
        psmqual = db.exec("select * from psmqual").fetchall()
    assert len(psmqual) == 0


def test_psm_quant(sqlengine, psm_qual, psm_quant, genetable, e2g_qual):
    # setup
    RECNO = 99995
    RUNNO = 426

    exp = models.Experiment(recno=RECNO)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=RUNNO, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=99995)
    #
    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()
    # test_import_genestable(sqlengine, genetable)
    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()
    #

    importer = importers.PSM_QUAL_Importer(psm_qual, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        psmqual = db.exec("select * from psmqual").fetchall()
    assert len(psmqual) != 0
    importer = importers.PSM_QUANT_Importer(psm_quant, sqlengine)
    importer.insert_data()
    with get_db(sqlengine) as db:
        psmquant = db.exec("select * from psmquant").fetchall()
    assert len(psmquant) != 0


def test_e2g_quant(sqlengine, e2g_quant, e2g_qual, genetable):
    # setup
    RECNO = 99995
    RUNNO = 426

    exp = models.Experiment(recno=RECNO)
    crud.create_exp(get_db(sqlengine), exp)
    model = models.ExperimentRun(runno=RUNNO, searchno=6)
    crud.create_experimentrun(get_db(sqlengine), model, recno=RECNO)

    importer = importers.GenesImporter(genetable, sqlengine)
    importer.insert_data()
    #

    importer = importers.E2G_QUAL_Importer(e2g_qual, sqlengine)
    importer.insert_data()

    importer = importers.E2G_QUANT_Importer(e2g_quant, sqlengine)
    importer.insert_data()

    with get_db(sqlengine) as db:
        e2gquant = db.exec("select * from e2gquant").fetchall()
        assert len(e2gquant) != 0


# no longer include this "wrong file" logic inside the importers
# def test_e2g_qual_wrongfile(sqlengine, e2g_quant):
#     importer = importers.E2G_QUAL_Importer(e2g_quant, sqlengine)
#     res = importer.insert_data()
#     with get_db(sqlengine) as db:
#         e2gqual = db.exec("select * from e2gqual").fetchall()
#         assert len(e2gqual) == 0


# def test_simple(sqlengine):
#
#     """
#     tests
#     importers.get_ispec_exp_export
#     split up
#     """
#
#     # test_engine = importers._make_inmemory_db_and_get_engine()
#     test_engine = sqlengine
#     file = importers.get_ispec_exp_export()
#     print(file)
#     file.seek(0)
#     Experiments_Importer = importers.Experiments_Importer
#     get_db = importers.get_db
#
#     importer = importers.Experiments_Importer(datafile=file, engine=test_engine)
#     importer.insert_data()
#     # turn this into a test
#     with get_db(test_engine) as db:
#         exps = db.exec("select * from experiment").fetchall()
#         print()
#         print(
#             f"""experiments in database:
#         {e for e in exps}
#         \n
#         """
#         )
#
#     print("====\nclose the db, now open and query\n")
#     print(crud.get_all_experiments(db))
#
#     print("====\nclose the db, now open and query\n")
#     # turn this into a test
#     file.seek(0)
#     importer.insert_data(
#         before_db_close_func=lambda db: print(crud.get_all_experiments(db))
#     )
#     with get_db(importer.engine) as db:
#         print(crud.get_all_experiments(db))
#
#     print(crud.get_all_experiments(db))


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
