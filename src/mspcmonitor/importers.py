# importers.py
import numpy as np
from tqdm import tqdm  # version 4.62.2
from pathlib import Path

from sqlalchemy import and_, or_
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
logger.setLevel(logging.INFO)
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


@lru_cache()
def get_all_recnos(engine=database.engine, key="default"):

    exps = crud.get_all_experiments(get_db(engine))
    _recnos = [x.recno for x in exps]
    return _recnos


def check_if_recno_exists(recno, **kwargs):
    recnos = get_all_recnos(**kwargs)
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
        #
        if model is None and self.model is None:
            raise ValueError("Must specify model")
        if model is None and self.model is not None:
            model = self.model
        if row.empty:
            logger.warning("Empty row sent to make model")
        column_mapping = model.Config.schema_extra.get("ispec_column_mapping")
        #
        if column_mapping is None:
            raise ValueError("column_mapping must be set in model config schema_extra")
        for dbcol, col in column_mapping.items():
            if bool(dbcol) == False or col not in row:
                continue
            kwargs[dbcol] = row[col]
            # raise ValueError(f"no data mapped correctly")
        # import ipdb; ipdb.set_trace()
        # if bool(kwargs) == False:
        #     return None
        # if db is None:
        #     db = get_db(self.engine)

        model_instance = model(
            **kwargs
        )  # sqlmodel does not care about misc. kwargs (by default(
        # import ipdb; ipdb.set_trace()
        model_instance = self.post_make_model(model_instance)
        # with get_db(self.engine) as db:
        #     db.add(model_instance)

        return model_instance

    def post_make_model(self, model):
        return model

    def make_models(self, data=None, db=None, **kwargs):
        if data is None:
            logger.warning(f"No data")
            return
        logger.info(f"{self.__class__} : making models")
        tqdm.pandas(desc="model making progress")
        model_instances = data.progress_apply(
            self.make_model, axis=1, model=self.model, db=db, **kwargs
        )
        # models = data.apply(self.make_model, axis=1, model=self.model)
        model_instances = list(filter(None, model_instances))
        if model_instances is None:
            return
        # import ipdb; ipdb.set_trace()
        logger.info(f"Made {len(model_instances)} models")
        return model_instances

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
            logger.info(f"{self} : No data to load")
            return

        # import ipdb; ipdb.set_trace()
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
            # import ipdb; ipdb.set_trace()
            model_instances = self.make_models(data=data, db=db)
            if model_instances is None:
                return
            for model in tqdm(model_instances, desc=f"writing to database"):
                # pass
                # crud.add_and_commit(db, model)
                db.add(model)
            #     #crud.add_and_commit(db, model)
            db.commit()

        if before_db_close_func is not None:
            pass
            # before_db_close_func(db)

        return model_instances


class ImporterWithChecker(Importer):
    def post_get_data(self, data: pd.DataFrame):
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


from tackle.utils import GeneMapper


class ImporterCreateMissingGenes(Importer):

    genemapper = GeneMapper()

    def make_model(
        self,
        row: pd.Series,
        model: sqlmodel.SQLModel = None,
        db: sqlmodel.orm.session.Session = None,
        **kwargs,
    ):
        if db is None:  # not good
            raise ValueError(f"{self.__class__} : must pass open db")
        # kwargs = dict()
        # add all values
        # logger.info(f"{self.__class__} : making models")

        geneid = row["GeneID"]
        # import ipdb; ipdb.set_trace()

        generecord = crud.get_gene_by_id(db, geneid)
        if generecord is not None:
            kwargs["geneid"] = generecord
        else:
            logger.warning(
                f"{self.__class__} : {geneid} not found in genes database, creating a new record"
            )
            genesymbol = None
            if "GeneSymbol" in row:
                genesymbol = row["GeneSymbol"]
            elif "Symbol" in row:
                genesymbol = row["Symbol"]
            if (
                isinstance(genesymbol, float)
                and not genesymbol is None
                and not np.isfinite(genesymbol)
            ):
                genesymbol = self.genemapper.symbol.get(str(geneid))
                logger.warning(
                    f"{self.__class__} : retrieved symbol {genesymbol} from tackle util "
                )
            ##
            logger.info(f"{self.__class__} : got {genesymbol} for {geneid}")
            try:
                taxonid = row["TaxonID"]
            except KeyError:
                taxonid = None
            generecord = models.Gene(geneid=geneid, symbol=genesymbol, taxonid=taxonid)
            # import ipdb; ipdb.set_trace()
            # db.add(generecord)
            # db.commit()
            # crud.add_and_commit(db, generecord)
            # create it
            # return

        if model is None:
            model = models.Gene
        model = super().make_model(row=row, db=db, model=model, **kwargs)
        # model = self.model(**kwargs)
        db.commit()
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
        # _res = crud.get_all_experiments(get_db(self.engine))
        # _existing_recnos = [x.recno for x in _res]
        _existing_recnos = get_all_recnos(engine=self.engine)
        return data[
            (~data.exp_EXPRecNo.isna()) & (~data.exp_EXPRecNo.isin(_existing_recnos))
        ].sort_values(by="exp_EXPRecNo", ascending=False)

    def post_make_model(self, model):
        return model
        # # check if already exists
        # recno = model.recno
        # # check again
        # _res = crud.get_all_experiments(get_db(self.engine))
        # existing_recnos = [x.recno for x in _res]
        # #
        # if recno in existing_recnos == True:
        #     logger.warning(f"{recno} already exists")
        #     return
        # return model


@dataclass(frozen=True)
class ExperimentRuns_Importer(Importer):
    model: models.ExperimentRun = models.ExperimentRun

    def post_get_data(self, data):
        if "exprun_EXPRecNo" not in data.columns:
            logger.warning(f"could not find `exprun_EXPRecNo` column. Check data file")
            logger.warning(f"exiting")
            return
        _existing_recnos = get_all_recnos(engine=self.engine)
        data = data[data.exprun_EXPRecNo.isin(_existing_recnos)]

        # _existing_expruns = crud.get_all_experimentruns(engine=self.engine)
        # crud.ge
        # crud.get_exprun_by_recrunsearch
        return data

    def make_model(self, row, **kwargs):

        # column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        # kwargs = dict()
        # # add all values
        # for dbcol, col in column_mapping.items():
        #     if col == "":
        #         continue
        #     if col not in row:
        #         continue
        #     kwargs[dbcol] = row[col]

        #
        # check
        _recno = row["exprun_EXPRecNo"]
        _runno = row["exprun_EXPRunNo"]
        _searchno = row["exprun_EXPSearchNo"]
        #
        experiment = crud.get_experiment_by_recno(get_db(), _recno)
        if experiment is None:
            logger.warning(f"{_recno} does not exist")
            return
        # =======
        # this is new
        kwargs["experiment_id"] = experiment.id
        exprun = crud.get_exprun_by_recrunsearch(get_db(), _recno, _runno, _searchno)
        if exprun is not None:
            print(f"{_recno}:{_runno} already exists")
            return
        return super().make_model(row=row, **kwargs)
        # exprun = models.ExperimentRun(**kwargs)
        # return exprun


@dataclass(frozen=True)
class E2G_QUAL_Importer(Importer):
    model: models.E2GQual = models.E2GQual

    def post_get_data(self, data):
        assert data.EXPRecNo.nunique() == 1
        assert data.EXPRunNo.nunique() == 1
        assert data.EXPSearchNo.nunique() == 1
        return data

    def make_models(self, data: pd.DataFrame, import_type="e2g", db=None, **kwargs):
        #
        if import_type not in ("e2g", "psm"):
            return ValueError("must be one of e2g | psm")
        if import_type == "e2g":
            attribute_flag = "is_imported_e2g"
        elif import_type == "psm":
            attribute_flag = "is_imported_psm"
        #
        recno = int(data.iloc[0]["EXPRecNo"])
        runno = int(data.iloc[0]["EXPRunNo"])
        searchno = int(data.iloc[0]["EXPSearchNo"])
        #
        gene_identifiers = data["GeneID"]
        #
        exprun = crud.get_exprun_by_recrunsearch(db, recno, runno, searchno)
        # import ipdb; ipdb.set_trace()
        if exprun is None:
            logger.warning(f"{self.__class__} : {recno}_{runno}_{searchno} not found")
            return
        if getattr(exprun, attribute_flag) == True:
            logger.warning(
                f"{self.__class__} : {recno}_{runno}_{searchno} has already been imported"
            )
            return
        exprun  # , exp = res
        kwargs["experimentrun"] = exprun
        kwargs["experiment"] = exprun.experiment
        # kwargs = dict(
        #     experimentrun=exprun,
        #     # experimentrun_id=exprun.id,
        #     experiment=exprun.experiment
        # )
        model_instances = super().make_models(data, db=db, **kwargs)
        # import ipdb; ipdb.set_trace()

        # exprun.is_imported = True
        # crud.add_and_commit(db, exprun)

        return model_instances

    def make_model(self, row, db=None, **kwargs):
        geneid = row["GeneID"]
        generecord = crud.get_gene_by_id(db, geneid)
        if generecord is None:
            generecord = ImporterCreateMissingGenes().make_model(
                row=row,
                model=models.Gene,
                db=db,
            )
        kwargs["geneid_id"] = generecord.geneid
        return super().make_model(row=row, db=db, **kwargs)

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
        # column_mapping = self.model.Config.schema_extra.get("ispec_column_mapping")
        # model = super().make_model(row=row, db=db, **kwargs)

        exprun = kwargs["experimentrun"]
        geneid = row["GeneID"]
        generecord = crud.get_gene_by_id(db, geneid)
        #
        if generecord is None:
            generecord = ImporterCreateMissingGenes().make_model(
                row=row,
                model=models.Gene,
                db=db,
            )
        kwargs["geneid_id"] = generecord.geneid
        # if generecord is None:
        #     ImporterCreateMissingGenes.make_model(
        #         row: pd.Series,
        #         model: sqlmodel.SQLModel = None,
        #         db: sqlmodel.orm.session.Session = None,
        #         **kwargs)
        # if generecord is not None:
        #     kwargs["geneid"] = generecord
        # else:
        #     logger.warning(f"{self} : {geneid} not found in database")
        #     return
        #

        # import ipdb; ipdb.set_trace()

        # perform two separate joins so we know who is on the left (when multiple e2gquals appear, not supposed to happen but we want to know about it)
        statement = (
            db.query(models.E2GQual)
            .join(models.ExperimentRun)
            .where(models.ExperimentRun.id == exprun.id)
            .join(models.Gene)
            .where(models.Gene.geneid == generecord.geneid)
        )

        if statement.count() > 1:
            logger.warning(
                f"{exprun.experiment.recno} {exprun.runno} {exprun.searchno}, {generecord}"
                f"{self} : more than one e2gqual found for {generecord.geneid}"
                " check for duplicate import"
            )
            # logger.warning('more than one e2gqual found, check for duplicate import')

        e2gqual = statement.first()
        if e2gqual is None:
            logger.warning(
                f"{exprun.experiment.recno} {exprun.runno} {exprun.searchno}, {generecord}"
                f"{self} : e2gqual not found for {geneid}"
            )
            return
        kwargs["e2gqual"] = e2gqual

        statement = (
            db.query(models.PSMQual)
            .join(models.ExperimentRun)
            .where(models.ExperimentRun.id == exprun.id)
            .join(models.Gene)
            .where(models.Gene.geneid == generecord.geneid)
        )
        psmqual = statement.first()
        kwargs["psmqual"] = psmqual
        # import ipdb; ipdb.set_trace()
        # qq = db.exec(statement)
        # _res = qq.first()
        # this is not enforced yet but it will be
        # assert len(e2gqual) == 1 :

        # model = self.model(**kwargs)
        model = super().make_model(row=row, db=db, **kwargs)
        return model

    def post_make_model(self, model):
        return model

    def make_models(self, db=None, import_type="e2g", **kwargs):
        # import ipdb; ipdb.set_trace()
        # def make_models(self, data: pd.DataFrame, db=None):
        if import_type not in ("e2g", "psm"):
            return ValueError("must be one of e2g | psm")
        if import_type == "e2g":
            attribute_flag = "is_imported_e2g"
        elif import_type == "psm":
            attribute_flag = "is_imported_psm"

        # import ipdb; ipdb.set_trace()
        model_instances = super().make_models(db=db, import_type=import_type, **kwargs)

        #
        if model_instances is None:
            return
        expruns = [model.experimentrun for model in model_instances]
        exprun0 = expruns[0]
        # grab exprun
        if not all(exprun == exprun0 for exprun in expruns):
            raise ValueError("Got more than one exprun after making models")
        else:
            exprun = exprun0

        logger.info(
            f"Setting {exprun.experiment.recno}_{exprun.runno}_{exprun.searchno} {attribute_flag} to True"
        )
        # import ipdb; ipdb.set_trace()
        setattr(exprun, attribute_flag, True)
        # exprun.is_imported_e2g = True
        return model_instances
        # crud.add_and_commit(db, exprun)


@dataclass(frozen=True)
class PSM_QUAL_Importer(E2G_QUAL_Importer):
    model: models.PSMQual = models.PSMQual

    def make_models(self, data: pd.DataFrame, import_type="psm", db=None):
        return super().make_models(data=data, import_type=import_type, db=db)


@dataclass(frozen=True)
class PSM_QUANT_Importer(E2G_QUANT_Importer):
    model: models.PSMQuant = models.PSMQuant

    def make_models(self, data: pd.DataFrame, import_type="psm", db=None):
        # import ipdb; ipdb.set_trace()
        return super().make_models(data=data, import_type="psm", db=db)


@dataclass(frozen=True)
class GenesImporter(ImporterCreateMissingGenes):
    model: models.Gene = models.Gene

    def post_get_data(self, data) -> pd.DataFrame:
        """
        omit any geneids already present in the database
        """
        # data = super().post_get_data(data)
        data = data.drop_duplicates(
            subset=["GeneID", "GeneSymbol", "TaxonID", "GeneDescription"], keep="first"
        )
        genes = crud.get_all_genes(get_db(self.engine))
        existing_geneids = [g.geneid for g in genes]
        if not data.GeneID.is_unique:
            logger.warning("Input genes table is not unique on geneid")
            # import ipdb; ipdb.set_trace()
        data = data.drop_duplicates("GeneID")
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
