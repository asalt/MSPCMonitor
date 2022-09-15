# importers.py
from tqdm import tqdm  # version 4.62.2
from pathlib import Path

from functools import lru_cache
from io import StringIO
from sqlmodel import Session
import sqlmodel
import pandas as pd
from . import crud
from . import database
from . import models

import logging

from .logging_config import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)
# logging.basicConfig(level=0)

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


# @lru_cache()
# def _get_recnos(key="default"):
#     exps = crud.get_all_experiments(get_db())
#     _recnos = [x.recno for x in exps]
#     return _recnos
#
#
# def check_if_recno_exists(recno):
#     recnos = _get_recnos()
#     return recno in recnos
#

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
    datafile: Optional[Path] = None
    engine: Any = database.engine  # TODO add type
    model: sqlmodel.SQLModel = None

    # def __post_init__(self):
    #     self.column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
    #     print(f"My field is {self.some_field}")

    def post_get_data(self, data) -> pd.DataFrame:
        return data

    def get_data(self, **kwargs) -> pd.DataFrame:
        sep = kwargs.pop("sep", "\t")
        if str(self.datafile).endswith(".csv"):
            sep = ","
        try:
            df = pd.read_table(self.datafile, sep=sep, **kwargs)
        except UnicodeDecodeError:
            df = pd.read_excel(self.datafile, **kwargs)
        return self.post_get_data(df)

        # return pd.read_table(self.datafile, **kwargs)

    def make_model(
        self,
        row: pd.Series,
        model: sqlmodel.SQLModel = None,
        db: sqlmodel.orm.session.Session = None,
        **kwargs,
    ) -> sqlmodel.SQLModel:
        if model is None and self.model is None:
            raise ValueError("Must specify model")
        if row.empty:
            logger.warning("Empty row sent to make model")
        column_mapping = model.Config.schema_extra.get("ispec_column_mapping")
        if column_mapping is None:
            raise ValueError("column_mapping must be set in model config schema_extra")
        for dbcol, col in column_mapping.items():
            if bool(dbcol) == False or col not in row:
                continue
            kwargs[dbcol] = row[col]
            # raise ValueError(f"no data mapped correctly")
        # if bool(kwargs) == False:
        #     return None
        # if db is None:
        #     db = get_db(self.engine)

        model_instance = model(
            **kwargs
        )  # sqlmodel does not care about misc. kwargs (by default(
        # import ipdb; ipdb.set_trace()
        model_instance = self.post_make_model(model_instance)
        return model_instance

    def post_make_model(self, model):
        return model

    def make_models(self, data=None, db=None, **kwargs):
        if data is None:
            logger.warning(f"No data")
            return
        logger.info(f"{self.__class__} : making models")
        tqdm.pandas(desc="model making progress")
        models = data.progress_apply(self.make_model, axis=1, model=self.model, db=db, **kwargs)
        # models = data.apply(self.make_model, axis=1, model=self.model)
        models = list(filter(None, models))
        # import pdb; pdb.set_trace()
        logger.info(f"Made {len(models)} models")
        return models

    def insert_data(self, data_kwargs=None, before_db_close_func: callable = None):
        """
        this inserts models 1 by 1
        """
        if data_kwargs is None:
            data_kwargs = dict()
        # import pdb; pdb.set_trace()
        data = self.get_data(**data_kwargs)
        if data is None:
            return
        if data.empty == True:
            logger.info("No data to load")
            return

        column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        if all(x not in data.columns for x in column_mapping.values()):
            _missing = set(column_mapping.values()) - set(data.columns)
            logger.warning(f"Could not find the following columns:")
            logger.warning(", ".join(_missing))
            logger.warning(data.columns)
            logger.warning(column_mapping)
            logger.warning(f"No data")
            return (None,)

        # print(f"{self}: added")

        #
        with get_db(self.engine) as db:
            # here we can change the order of when things happen
            # right now all models are made, then added 1 by 1
            models = self.make_models(data, db=db)
            if models is None:
                return
            for model in tqdm(models, desc=f"writing to database"):
                crud.add_and_commit(db, model)

        if before_db_close_func is not None:
            pass
            # before_db_close_func(db)

        return models


class ImporterWithChecker(Importer):
    def post_get_data(self, data: pd.DataFrame):
        import pdb

        pdb.set_trace()
        if "EXPRecNo" not in data.columns:
            return
        _recno = data.iloc[0]["EXPRecNo"]
        _runno = data.iloc[0]["EXPRunNo"]
        _searchno = data.iloc[0]["EXPSearchNo"]
        _res = crud.get_exprun_by_recrunsearch(
            get_db(self.engine), _recno, _runno, _searchno
        )
        if _res is None:
            return
        return data

class ImporterCreateMissingGenes(Importer):
    def make_model(self, row, db=None, **kwargs):
        if db is None:  # not good
            raise ValueError(f"{self.__class__} : must pass open db")
        # kwargs = dict()
        # add all values
        logger.info(f"{self.__class__} : making models")

        geneid = row["GeneID"]

        generecord = crud.get_gene_by_id(db, geneid)
        if generecord is not None:
            kwargs["geneid"] = generecord
        else:
            logger.warning(f"{self} : {geneid} not found in database, creating a new record")
            if "GeneSymbol" in row:
                genesymbol = row["GeneSymbol"]
            elif "Symbol" in row:
                genesymbol = row["Symbol"]
            else:
                genesymbol = None
            try:
                taxonid = row["TaxonID"]
            except KeyError:
                taxonid = None
            generecord = models.Gene(geneid=geneid,  symbol=genesymbol, taxonid=taxonid)
            crud.add_and_commit(db, generecord)
            # create it
            return

        # import pdb; pdb.set_trace()
        model = super().make_model(row=row, db=db, **kwargs)
        # model = self.model(**kwargs)
        # import ipdb; ipdb.set_trace()
        return model


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
            logger.warning(f"could not find `exp_EXPRecNo` column. Check data file")
            logger.warning(f"exiting")
            return

        # print(len(data))
        # print(data.columns)
        _res = crud.get_all_experiments(get_db(self.engine))
        _existing_recnos = [x.recno for x in _res]
        return data[
            (~data.exp_EXPRecNo.isna()) & (~data.exp_EXPRecNo.isin(_existing_recnos))
        ].sort_values(by="exp_EXPRecNo", ascending=False)

    def post_make_model(self, model):
        # check if already exists
        recno = model.recno
        # check again
        _res = crud.get_all_experiments(get_db(self.engine))
        existing_recnos = [x.recno for x in _res]
        #
        if recno in existing_recnos == True:
            logger.warning(f"{recno} already exists")
            return
        return model


@dataclass(frozen=True)
class ExperimentRuns_Importer(Importer):
    model: models.ExperimentRun = models.ExperimentRun

    def post_get_data(self, data):
        if "exprun_EXPRecNo" not in data.columns:
            logger.warning(f"could not find `exprun_EXPRecNo` column. Check data file")
            logger.warning(f"exiting")
            return
        _existing_recnos = _get_recnos()
        return data[data.exprun_EXPRecNo.isin(_existing_recnos)]

    def make_model(self, row, **kwargs):

        column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        kwargs = dict()
        # add all values
        for dbcol, col in column_mapping.items():
            if col == "":
                continue
            if col not in row:
                continue
            kwargs[dbcol] = row[col]

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
        # kwargs["experiment_id"] = experiment.id
        with get_db() as db:
            experiment = crud.get_experiment_by_recno(db, _recno)
            exprun = crud.get_exprun_by_recrunsearch(db, _recno, _runno, _searchno)
            # print(kwargs)
            if exprun is not None:
                print(f"{_recno}:{_runno} already exists")
                return
            kwargs["experiment_id"] = experiment.id
            # kwargs["experiment"] = experiment
            # kwargs.pop('recno')
            # kwargs.pop('model')
            exprun = models.ExperimentRun(**kwargs)
            # exprun = models.ExperimentRun(recno=recno,runno=runno,searchno=searchno, experiment=experiment, instrument=instrument, is_imported=is_imported, is_grouped=is_grouped)
        return exprun


@dataclass(frozen=True)
class E2G_QUAL_Importer(ImporterCreateMissingGenes):
    model: models.E2GQual = models.E2GQual

    def post_get_data(self, data):
        assert data.EXPRecNo.nunique() == 1
        assert data.EXPRunNo.nunique() == 1
        assert data.EXPSearchNo.nunique() == 1
        return data

    def make_models(self, data: pd.DataFrame, db=None):
        #
        recno = int(data.iloc[0]["EXPRecNo"])
        runno = int(data.iloc[0]["EXPRunNo"])
        searchno = int(data.iloc[0]["EXPSearchNo"])
        #
        gene_identifiers = data["GeneID"]
        #
        exprun = crud.get_exprun_by_recrunsearch(db, recno, runno, searchno)
        if exprun is None:
            logger.warning(f"{self.__class__} : {recno}_{runno}_{searchno} not found")
            return
        exprun  # , exp = res
        kwargs = dict(
            experimentrun=exprun,
            experimentrun_id=exprun.id,
        )
        models = super().make_models(data, db=db, **kwargs)

        return models

    # def make_model(self, row, db=None, **kwargs):
    #     if db is None:  # not good
    #         raise ValueError(f"{self.__class__} : must pass open db")
    #     # kwargs = dict()
    #     # add all values
    #     logger.info(f"{self.__class__} : making models")

    #     geneid = row["GeneID"]

    #     generecord = crud.get_gene_by_id(db, geneid)
    #     if generecord is not None:
    #         kwargs["geneid"] = generecord
    #     else:
    #         logger.warning(f"{self} : {geneid} not found in database, creating a new record")
    #         if "GeneSymbol" in row:
    #             genesymbol = row["GeneSymbol"]
    #         elif "Symbol" in row:
    #             genesymbol = row["Symbol"]
    #         else:
    #             genesymbol = None
    #         try:
    #             taxonid = row["TaxonID"]
    #         except KeyError:
    #             taxonid = None
    #         generecord = models.Gene(geneid=geneid,  symbol=genesymbol, taxonid=taxonid)
    #         crud.add_and_commit(db, generecord)
    #         # create it
    #         return

    #     # import pdb; pdb.set_trace()
    #     model = super().make_model(row=row, db=db, **kwargs)
    #     # model = self.model(**kwargs)
    #     # import ipdb; ipdb.set_trace()
    #     return model


@dataclass(frozen=True)
class E2G_QUANT_Importer(E2G_QUAL_Importer):
    model: models.E2GQuant = models.E2GQuant

    def make_model(self, row, db=None, **kwargs):
        model = super().make_model(row, db, **kwargs)
        # column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")

        exprun = kwargs["experimentrun"]
        geneid = row["GeneID"]
        generecord = crud.get_gene_by_id(db, geneid)

        statement = (
            db.query(models.E2GQual)
            .where(models.E2GQual.geneid == generecord)
            .join(models.Gene)
        )
        _res = statement.first()
        # qq = db.exec(statement)
        # _res = qq.first()
        if _res is not None:
            e2gqual = _res
        else:
            logger.warning(
                f"{exprun.experiment.recno} {exprun.runno} {exprun.searchno}, {generecord}"
                f"{self} : e2gqual not found for {geneid}"
            )
            return
        # this is not enforced yet but it will be
        # assert len(e2gqual) == 1 :
        kwargs["e2gqual"] = e2gqual
        if generecord is not None:
            kwargs["geneid"] = generecord
        else:
            logger.warning(f"{self} : {geneid} not found in database")
            return

        model = self.model(**kwargs)
        return model

    def post_make_model(self, model):
        return model


@dataclass(frozen=True)
class PSM_QUAL_Importer(E2G_QUAL_Importer):
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
        # data = super().post_get_data(data)
        genes = crud.get_all_genes(get_db(self.engine))
        existing_geneids = [g.geneid for g in genes]
        if not data.GeneID.is_unique:
            logger.warning("Input genes table is not unique on geneid")
        data = data.drop_duplicates('GeneID')
        data = data[~data.GeneID.isin(existing_geneids)]
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
