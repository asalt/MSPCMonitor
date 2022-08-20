# importers.py
from tqdm import tqdm # version 4.62.2

from functools import lru_cache
from io import StringIO
from sqlmodel import Session
import pandas as pd
import sqlmodel
from . import crud
from . import database
from . import models

import logging
logging.basicConfig(level=0)

# Efrom .database import engine
from typing import Optional, Dict, List, Any, Type

# from . import crud, models, schemas
"""
"""

from dataclasses import dataclass, field

# from .dbutils import get_db


def get_db(engine=database.engine):
    db = Session(bind=engine)
    try:
        return db
    finally:
        db.close()


@lru_cache()
def _get_recnos(key="default"):
    exps = crud.get_all_experiments(get_db())
    _recnos = [x.recno for x in exps]
    return _recnos


def check_if_recno_exists(recno):
    recnos = _get_recnos()
    return recno in recnos


def _make_inmemory_db_and_get_engine():

    from sqlmodel import create_engine, SQLModel

    SQLALCHEMY_DATABASE_URL = "sqlite://"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, echo=True
    )
    SQLModel.metadata.create_all(engine)
    return engine


# @lru_cache()
# def get_recnos_in_db(key="default", engine=database.engine):
#     exps = crud.get_all_experiments(get_db(engine))
#     _recnos = [x.recno for x in exps]
#     return _recnos
# def read(self):
#     if self.datafile is None:
#         raise ValueError("datafile must be set")
#     self.df = pd.read_table(self.datafile)
#     return self.df

# def check_if_recno_exists(self, recno):
#     recnos = get_recnos_in_db()
#     return recno in recnos

from dataclasses import dataclass
from abc import ABCMeta, ABC, abstractmethod


# @dataclass(frozen=True)
# class AbstractBaseImporter(ABC):
#
#     column_mapping: Optional[dict] = None
#     datafile: Optional[str] = None
#     engine: Any = None  # TODO add type
#     model: sqlmodel.SQLModel = None
#
#     @abstractmethod
#     def get_data(self):
#         return "no"
#
#     @abstractmethod
#     def make_model(self):
#         return "no"

# class AbstractBaseImporterFactory(ABC):
#     @abstractmethod
#     def get_importer(self, **kwargs) -> AbstractBaseImporter:
#         pass


# @dataclass(frozen=True)
# class Importer(AbstractBaseImporter):
#    def get_data(self):
#        return "yes"
#    def make_model(self):
#        return "yes"

# class ImporterFactory(AbstractBaseImporterFactory, Importer):
#     def get_importer(self) -> AbstractBaseImporter:
#         return Importer(datafile=self.datafile, engine=self.engine)

# these are file importers that map column names to database column names


@dataclass(frozen=True)
class Importer:
    datafile: Optional[str] = None
    engine: Any = database.engine  # TODO add type
    model: sqlmodel.SQLModel = None

    # def __post_init__(self):
    #     self.column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
    #     print(f"My field is {self.some_field}")

    def post_get_data(self, data) -> pd.DataFrame:
        return data

    def get_data(self, **kwargs):
        sep=kwargs.pop('sep', '\t')
        if str(self.datafile).endswith('.csv'):
            sep=','
        try:
            df = pd.read_table(self.datafile, sep=sep, **kwargs)
        except UnicodeDecodeError:
            df = pd.read_excel(self.datafile, **kwargs)
        return self.post_get_data(df)

        # return pd.read_table(self.datafile, **kwargs)

    def make_model(self, row: pd.Series, model: sqlmodel.SQLModel, db=None, **kws) -> sqlmodel.SQLModel:
        if row.empty:
            logging.warning("Empty row sent to make model")
        column_mapping = model.Config.schema_extra.get("ispec_column_mapping")
        if column_mapping is None:
            raise ValueError("column_mapping must be set in model config schema_extra")
        for dbcol, col in column_mapping.items():
            if bool(dbcol) == False or col not in row:
                continue
            kws[dbcol] = row[col]
            # raise ValueError(f"no data mapped correctly")
        # if bool(kws) == False:
        #     return None
        # if db is None:
        #     db = get_db(self.engine)

        model_instance = model(
            **kws
        )  # sqlmodel does not care about misc. kwargs (by default(
        # import ipdb; ipdb.set_trace()
        model_instance = self.post_make_model(model_instance)
        return model_instance

    def post_make_model(self, model):
        return model

    def make_models(self, data=None):
        logging.info(f"making models")
        if data is None:
            logging.warning(f"No data")
            return
        tqdm.pandas(desc='model making progress')
        models = data.progress_apply(self.make_model, axis=1, model=self.model)
        # models = data.apply(self.make_model, axis=1, model=self.model)
        models = list(filter(None, models))
        logging.info(f"Made {len(models)} models")
        return models

    def insert_data(self, data_kws=None, before_db_close_func: callable = None):
        """
        this inserts models 1 by 1
        """
        if data_kws is None:
            data_kws = dict()
        # import pdb; pdb.set_trace()
        data = self.get_data(**data_kws)
        if data is None:
            return
        if data.empty == True:
            logging.info("No data to load")
            return

        column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        if all(x not in data.columns for x in column_mapping.values()):
            _missing = set(column_mapping.values()) - set(data.columns)
            logging.warning(f"Could not find the following columns:")
            logging.warning(", ".join(_missing))
            logging.warning(data.columns)
            logging.warning(column_mapping)
            logging.warning(f"No data")
            return (None,)

        print(f"{self}: added")
        models = self.make_models(data)
        if models is None:
            return

        with get_db(self.engine) as db:
            for model in tqdm(models, desc=f"writing to database"):
                crud.add_and_commit(db, model)

        if before_db_close_func is not None:
            pass
            # before_db_close_func(db)

        return models

    def insert_data_multiple(
        self, data=None, data_kws=None, before_db_close_func: callable = None
    ):
        if data_kws is None:
            data_kws = dict()
        models = self.make_models(self.get_data(**data_kws))
        if models is None:
            return
        # if models is None:
        #     return
        with get_db(self.engine) as db:
            crud.add_multiple_and_commit(db, models)
        #
        # check
        # if check_if_recno_exists(row.exp_EXPRecNo) == True:
        #    print(f"{row.exp_EXPRecNo} already exists")
        #    return
        # exp = models.Experiment(**kws)
        # print(exp)
        # return exp


class ImporterWithChecker(Importer):
    def post_get_data(self, data: pd.DataFrame):
        import pdb

        pdb.set_trace()
        if "EXPRecNo" not in data.columns:
            return
        _recno = data.iloc[0]["EXPRecNo"]
        _runno = data.iloc[0]["EXPRunNo"]
        _searchno = data.iloc[0]["EXPSearchNo"]
        _res = crud.get_exprun_by_recrunsearch(get_db(self.engine), _recno, _runno, _searchno)
        if _res is None:
            return
        return data


# =======================================================
#
#  _____                           _
# |_   _|                         | |
#   | | _ __ ___  _ __   ___  _ __| |_ ___ _ __ ___
#   | || '_ ` _ \| '_ \ / _ \| '__| __/ _ \ '__/ __|
#  _| || | | | | | |_) | (_) | |  | ||  __/ |  \__ \
#  \___/_| |_| |_| .__/ \___/|_|   \__\___|_|  |___/
#                | |
# =======================================================
# these are file-based importers.
# The input is a file with column headers.
# they get mapped to the appropriate database
# column names and inserted in the database


@dataclass(frozen=True)
class Experiments_Importer(Importer):
    model: models.Experiment = models.Experiment

    def post_get_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "exp_EXPRecNo" not in data.columns:
            logging.warning(f"could not find `exp_EXPRecNo` column. Check data file")
            logging.warning(f"exiting")
            return

        print(len(data))
        print(data.columns)
        _existing_recnos = _get_recnos()
        return data[
            (~data.exp_EXPRecNo.isna()) & (~data.exp_EXPRecNo.isin(_existing_recnos))
        ].sort_values(by="exp_EXPRecNo", ascending=False)

    def post_make_model(self, model):
        # check if already exists
        _recno = model.recno
        # import ipdb; ipdb.set_trace()
        if check_if_recno_exists(_recno) == True:
            logging.warning(f"{_recno} already exists")
            return
        return model


@dataclass(frozen=True)
class ExperimentRuns_Importer(Importer):
    model: models.ExperimentRun = models.ExperimentRun

    def post_get_data(self, data):
        if "exprun_EXPRecNo" not in data.columns:
            logging.warning(f"could not find `exprun_EXPRecNo` column. Check data file")
            logging.warning(f"exiting")
            return
        _existing_recnos = _get_recnos()
        return data[data.exprun_EXPRecNo.isin(_existing_recnos)]

    def make_model(self, row, **kws):

        column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        kws = dict()
        # add all values
        for dbcol, col in column_mapping.items():
            if col == "":
                continue
            if col not in row:
                continue
            kws[dbcol] = row[col]

        #
        # check
        _recno = row["exprun_EXPRecNo"]
        _runno = row["exprun_EXPRunNo"]
        _searchno = row["exprun_EXPSearchNo"]
        #
        if check_if_recno_exists(_recno) == False:
            print(f"{_recno} does not exist")
            return
        experiment = crud.get_experiment_by_recno(get_db(), _recno)
        if experiment is None:
            print(f"{_recno} does not exist")
            return
        #
        #
        #kws["experiment_id"] = experiment.id
        with get_db() as db:
            experiment = crud.get_experiment_by_recno(db, _recno)
            exprun = crud.get_exprun_by_recrunsearch(db, _recno, _runno, _searchno)
            #print(kws)
            if exprun is not None:
                print(f"{_recno}:{_runno} already exists")
                return
            kws["experiment_id"] = experiment.id
            #kws["experiment"] = experiment
            #kws.pop('recno')
            #kws.pop('model')
            exprun = models.ExperimentRun(**kws)
            #exprun = models.ExperimentRun(recno=recno,runno=runno,searchno=searchno, experiment=experiment, instrument=instrument, is_imported=is_imported, is_grouped=is_grouped)
        return exprun


@dataclass(frozen=True)
class E2G_QUAL_Importer(Importer):
    model: models.E2GQual = models.E2GQual

    def make_models(self, data: pd.DataFrame):
        if data is None:
            logging.warning(f"No data")
            return
        #
        recno = int(data.iloc[0]["EXPRecNo"])
        runno = int(data.iloc[0]["EXPRunNo"])
        searchno = int(data.iloc[0]["EXPSearchNo"])
        #
        gene_identifiers = data['GeneID']
        # TODO:
        # query database for gene identifiers
        # for any new genes, add them to the database?
        ## added

        #
        # crud.get_exprun_by_recrun(get_db(self.engine), recno, runno, searchno)
        # get_db(self.engine).exec('select * from experimentrun').fetchall()
        with get_db(self.engine) as db:
            res = crud.get_exprun_by_recrunsearch(db, recno, runno, searchno)
            if res is None:
                logging.warning(f"{recno}_{runno}_{searchno} not found")
                return
            #assert len(res) == 2
            exprun, exp = res
            kws = dict(
                experiment = exp,
                experimentrun = exprun,
                experiment_id = exp.id, #
                experimentrun_id = exprun.id,
                db=db # pass the same db connection instance
            )
            logging.info(f"making models")
            tqdm.pandas(desc='model making progress')
            models = data.apply(self.make_model, axis=1, model=self.model, **kws)
        # models = data.progress_apply(self.make_model, axis=1, model=self.model)
        # models = list(filter(None, models))
        # logging.info(f"Made {len(models)} models")
        return models

    def make_model(self, row, db=None, **kws):
        column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        kws = dict()
        kws = dict()
        # add all values

        for dbcol, col in column_mapping.items():
            if col == "":
                continue
            if col not in row:
                continue
            kws[dbcol] = row[col]


        geneid = row['GeneID']
        #
        #generecord = crud.get_gene_by_id(get_db(self.engine), geneid)
        if db is None: # not good
            raise ValueError()
        generecord = crud.get_gene_by_id(db, geneid)
        if generecord is not None:
            kws['geneid'] = generecord
        else:
            # or make a new record
            #pass
            return

            #kws['geneid'] = geneid


        #import pdb; pdb.set_trace()
        model = self.model(**kws)
        return model

@dataclass(frozen=True)
class E2G_QUANT_Importer(Importer):
    model: models.E2GQuant = models.E2GQuant
    def post_make_model(self, model):
        return model


@dataclass(frozen=True)
class PSM_QUAL_Importer(Importer):
    model: models.PSMQual = models.PSMQual


@dataclass(frozen=True)
class PSM_QUANT_Importer(Importer):
    model: models.PSMQuant = models.PSMQuant

@dataclass(frozen=True)
class GenesImporter(Importer):
    model: models.Gene = models.Gene
    def post_get_data(self, data) -> pd.DataFrame:
        """
        omit any geneids already present in the database
        """
        data = super().post_get_data(data)
        genes = crud.get_all_genes(get_db(self.engine))
        existing_geneids = [g.geneid for g in genes]
        data = data[ ~data.GeneID.isin(existing_geneids) ]
        return data


# the important tests have been moved out into test
def test_simple():
    from mspcmonitor import importers, crud

    test_engine = importers._make_inmemory_db_and_get_engine()
    file = importers.get_ispec_exp_export()
    print(file)
    file.seek(0)
    Experiments_Importer = importers.Experiments_Importer
    get_db = importers.get_db

    importer = Experiments_Importer(datafile=file, engine=test_engine)
    importer.insert_data()
    with get_db(test_engine) as db:
        exps = db.exec("select * from experiment").fetchall()
        print(exps)

    print(crud.get_all_experiments(db))

    file.seek(0)
    importer.insert_data(
        before_db_close_func=lambda db: print(crud.get_all_experiments(db))
    )
    with get_db(importer.engine) as db:
        print(crud.get_all_experiments(db))

    print(crud.get_all_experiments(db))


def test():

    from io import StringIO
    from mspcmonitor import importers, crud

    test_engine = importers._make_inmemory_db_and_get_engine()

    # importers.Experiments_Importer().check_if_recno_exists(99999)

    Experiments_Importer = importers.Experiments_Importer
    # assert Experiments_Importer.column_mapping is None
    # assert Experiments_Importer().column_mapping is not None  # is dict

    # ispec_import_headers = "\t".join(Experiments_Importer().column_mapping.values())
    ispec_import_headers = "\t".join(
        Experiments_Importer.model.Config.schema_extra["ispec_column_mapping"].values()
    )
    file = StringIO(f"{ispec_import_headers}\n99999\t1\t\6")
    file.seek(0)

    importer = Experiments_Importer(datafile=file, engine=test_engine)
    importer.get_data()  # returns a dataframe

    # if we try again we get an error
    try:
        importer.get_data()
    except Exception as e:
        print(f"error: {e}")
        print("we got an error")

    # seek back
    file.seek(0)
    # importer.get_data() # can read again
    importer.insert_data(
        before_db_close_func=lambda db: print(crud.get_all_experiments(db))
    )
    with get_db(importer.engine) as db:
        print(crud.get_all_experiments(db))
    crud.get_all_experiments()

    from sqlalchemy import text

    # importer.engine.connect().execute(text("show tables"))

    with importers.get_db(importer.engine) as db:
        crud.get_all_experiments(db)

    # get_db(test_engine)
    _make_inmemory_db_and_get_engine = importers._make_inmemory_db_and_get_engine

    # with importers.get_db(importer.engine) as db:
    test_engine = _make_inmemory_db_and_get_engine()
    get_db = importers.get_db
    with get_db(test_engine) as db:
        db.exec(
            """INSERT INTO experiment (recno, label, extractno, date, digest_enzyme, cell_tissue, geno) VALUES (22222, '', 2, '', '', '', '')"""
        )
        db.commit()
        db.exec("select * from experiment")
        exps = crud.get_all_experiments(db)
        print(exps)

    # sql = """
    #     SELECT name FROM sqlite_schema
    #     WHERE type='table'
    #     ORDER BY name;
    # """
