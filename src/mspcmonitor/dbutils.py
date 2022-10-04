# dbutils.py
from dataclasses import dataclass
from collections import defaultdict
import sys
import re
from typing import List

from functools import lru_cache
from matplotlib.pyplot import get
import typer
import logging
import pandas as pd
from pathlib import Path
import sqlmodel
from sqlmodel import Session

from . import crud, models, schemas
from mspcmonitor.database import engine, get_engine
from mspcmonitor import importers

app = typer.Typer(name="db", result_callback=None)


def get_db(engine=engine):
    db = Session(engine)
    try:
        return db
    finally:
        db.close()


from .logging_config import get_logger

logger = get_logger(__name__)


@lru_cache()
def _get_recnos(key="default"):
    exps = crud.get_all_experiments(get_db())
    _recnos = [x.recno for x in exps]
    return _recnos


def check_if_recno_exists(recno):
    recnos = _get_recnos()
    return recno in recnos

@app.command('autoimport')
def autoimport(
    dry: bool = typer.Option(
        False, "--dry", '-d', help="Dry run, do not actually execute commands"
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive", "-r"
    ),
    path: Path = typer.Argument(
        default=Path('.'),
        help="Root path to monitor",
    ),
    ):
    """
    this is an autoupload script for bulk insert of ispec tables
    new functionality will be added later
    """
    logger.info(f"searching {path} for new tables")
    globstr = "*[e2g|psm]*tsv"
    if recursive is True:
        _attr  = 'rglob'
    else:
        _attr = 'glob'
    globstr = f"**/{globstr}"
    files = list(getattr(path, _attr)(globstr))
    file_groups = get_groups("\d{5,}_\d{1,}_\d{1,}", files)
    # import ipdb; ipdb.set_trace()

    if len(file_groups) == 0:
        logger.warning(f"No filegroups identified")

    list(map(lambda fg: logger.info(f"Found filegroup {fg}"), file_groups))

    files_grouped_dict = group_files(file_groups, files)

    _fgd = files_grouped_dict
    for group in _fgd.keys():
        for filetype in _fgd[group].keys():
            for qtype in _fgd[group][filetype].keys():
                logger.info(f"{group}")
                logger.info("-"*4+f" filetype: {filetype}")
                qtype_val = _fgd[group][filetype][qtype]
                logger.info("-"*6+f" qtype: {qtype}")
                logger.info('-'*8+f" files: {qtype_val}")
            #logger.info(f"{group}")
    # list(map(lambda fg: logger.info(f"Grouped files {fg}\n\n"), files_grouped_dict.items()))

    # for fg in filegroups:
    #     logger.info(f"Found filegroup {fg}")
    #     print(fg)


    return

def get_groups(pat, inputs):
    """
    inputs : List[Path]
    Get all the groups we have.
    """
    p = re.compile(pat)
    groups = set(m.group() for m in
                 [p.search(i.name) for i in inputs]
                 if m)
    return groups

def get_file_datatype(p:Path, qtype=False):
    if 'psm' in p.name and not qtype:
        return 'psm'
    elif 'e2g' in p.name and not qtype:
        return 'e2g'
    elif not qtype:
        return p.name
    if qtype:
        if "QUANT" in p.name:
            return 'QUANT'
        elif "QUAL" in p.name:
            return 'QUAL'
        else:
            return p.name

def group_files(groups:set, files:list) -> dict:
    """
    {group : [e2g|psm][QUAL|QUANT] = files }
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list) )) # nested x3

    for group in groups:
        pat = re.compile(f".*{group}.*")
        #import ipdb; ipdb.set_trace()
        targets = [x for x in files if pat.search(x.name)]
        for target in targets:
            psm_or_e2g = get_file_datatype(target)
            qual_or_quant = get_file_datatype(target, qtype=True)
            out[group][psm_or_e2g][qual_or_quant].append(target)

        #out[group] = targets
    return out




@app.command(
    "create",
    help="executes SQLModel.metadata.create_all(engine) for ispec.db"
    "\nNote does not delete anything",
)
def create(
    force: bool = typer.Option(False),
    # path: Path = ".",
):
    """
    create database tables
    """
    #
    import ipdb

    if Path("ispec.db").exists() and not force:
        logger.info("ispec.db already exists")
        return

    models.create_db_and_tables(engine)
    logger.info("created new database")
    #




@app.command("add-metadata")
def add_metadata(ispec_export_file: Path):
    import_ispec_experiments(ispec_export_file)
    return


# 20221904_ispec_cur_Experiments_metadata_export.xlsx
@app.command("add-experiments-table")
def add_exp_table(ispec_export_file: Path):
    # import_ispec_experiments(ispec_export_file)
    importers.Experiments_Importer(ispec_export_file).insert_data(data_kwargs=dict())
    return


@app.command("add-experimentruns-table")
def add_exprun_table(ispec_export_file: Path):
    # import_ispec_expruns(ispec_export_file)
    importers.ExperimentRuns_Importer(ispec_export_file).insert_data(data_kwargs=dict())
    return


@app.command("add-experimentruns-table")
def add_exprun_table(ispec_export_file: Path):
    # import_ispec_expruns(ispec_export_file)
    importers.ExperimentRuns_Importer(ispec_export_file).insert_data(data_kwargs=dict())
    return


@app.command("add-e2g-table")
def add_e2g_table(e2g_qual_file: Path, e2g_quant_file: Path):
    # these importers does not make any foreign keys yet
    importers.E2G_QUAL_Importer(e2g_qual_file).insert_data(data_kwargs=dict())
    importers.E2G_QUANT_Importer(e2g_quant_file).insert_data(data_kwargs=dict())
    # import_e2g(e2g_qual_file=e2g_qual, e2g_quant_file=e2g_quant_file)
    return


@app.command("add-psm-table")
def add_psm_table(psm_qual_file: Path, psm_quant_file: Path):
    # these importers does not make any foreign keys yet
    importers.PSM_QUAL_Importer(psm_qual_file).insert_data(data_kwargs=dict())
    importers.PSM_QUANT_Importer(psm_quant_file).insert_data(data_kwargs=dict())
    # import_e2g(e2g_qual_file=e2g_qual, e2g_quant_file=e2g_quant_file)
    return


@app.command("add-genes-table")
def add_genes_table(gene_table_file: Path):
    importers.GenesImporter(gene_table_file).insert_data(data_kwargs=dict())
    return


from datetime import datetime


@app.command("monitor-for-rawfiles")
def monitor_for_rawfiles(
    path: Path, engine, last_time=None, maxsize=10, inst_id=99995
) -> list:
    if last_time is None:
        last_time = datetime(1971, 1, 1, 1, 1, 1)
    new_rawfiles = list()

    nfound = 0
    for rawfile in path.glob("**/*raw"):
        #
        logger.debug("=" * 80)
        logger.debug(f"{rawfile}")

        statres = rawfile.stat()

        MIN_RAWFILE_SIZE = -1
        MIN_WAIT_TIME = -1
        logger.debug(f"{rawfile.name} {statres}")
        logger.debug(
            f"{statres.st_mtime} gt {last_time.timestamp()} \
                   and {statres.st_size} gt {MIN_RAWFILE_SIZE} \
                   and {statres.st_mtime} sub {statres.st_ctime} / 60 gt {MIN_WAIT_TIME} \
                   "
        )

        if not (
            statres.st_mtime
            > last_time.timestamp()  # change to some min timestamp so we
            and statres.st_size
            > MIN_RAWFILE_SIZE  # don't process all rawfiles in existance
            and (statres.st_mtime - statres.st_ctime) / 60 > MIN_WAIT_TIME
        ):
            logger.debug(f"Skipping {rawfile}")
            continue

        #
        # put in another function
        try:
            _rec, _run, *res = rawfile.name.split("_")
        except ValueError:
            logger.debug(f"Skipping {rawfile} because cannot parse")
            continue
        #
        exprun = crud.get_exprun_by_recrun(get_db(engine), _rec, _run)

        if exprun is None:
            logger.debug(f"No exprun {_rec}_{_run} found for {rawfile}")
            continue
        else:
            exprun = exprun[0]
        rawfile_rec = crud.get_rawfile_by_name(get_db(engine), rawfile.name)
        new_rawfile = models.RawFile(
            name=rawfile.name,
            ctime=datetime.fromtimestamp(statres.st_ctime),
            mtime=datetime.fromtimestamp(statres.st_mtime),
            size=statres.st_size,
            # exprun=exprun, # don't do this, can break things, also not needed
            exprun_id=exprun.id,
        )
        if rawfile_rec == new_rawfile:
            logger.debug(f"\n\tAlready IN DB\n{rawfile_rec}")
            # already in database, skip
            # see models.RawFile.__eq__ for details
            continue

        if rawfile_rec is not None:
            exprun = rawfile_rec.exprun
            if exprun is None:
                # missing
                # try to find exprun
                exprun = None
            # import ipdb; ipdb.set_trace()
            logger.debug(f"record linked to rawfile is {exprun}")
            if (
                rawfile_rec is not None
                and exprun is not None
                and exprun.is_imported == True
            ):
                continue  # already exists and has been processed

        logger.debug(f"Found {rawfile}")

        crud.create_rawfile(get_db(engine), new_rawfile)
        logger.info(f"Saved {rawfile} record in db")

        new_rawfiles.append(rawfile)
        nfound += 1

    return new_rawfiles


def import_ispec_expruns(ispec_export_file: Path):
    """ """

    column_mapping = {
        "": "dev_exprun_u2g_check",
        "runno": "exprun_AddedBy",
        "date": "exprun_CreationTS",
        "": "exprun_E2GFile0_expFileName",
        "recno": "exprun_EXPRecNo",
        "runno": "exprun_EXPRunNo",
        "searchno": "exprun_EXPSearchNo",
        "": "exprun_Fraction_10090",
        "": "exprun_Fraction_9031",
        "": "exprun_Fraction_9606",
        "": "exprun_Fraction_9606 Average",
        "": "exprun_Fraction_9606 Standard Deviation",
        "": "exprun_Fraction_test",
        "is_grouped": "exprun_Grouper_EndFLAG",
        "": "exprun_Grouper_FailedFLAG",
        "": "exprun_Grouper_Filter_modiMax",
        "": "exprun_Grouper_FilterStamp",
        "": "exprun_Grouper_RefDatabase",
        "": "exprun_Grouper_StartFLAG",
        "": "exprun_Grouper_Version",
        "is_imported": "exprun_Import_EndFLAG",
        "": "exprun_Import_FixFLAG",
        "": "exprun_Import_StartFLAG",
        "": "exprun_ImportTS",
        "": "exprun_InputFileName",
        "": "exprun_LabelType",
        "": "exprun_ModificationTS",
        "": "exprun_MS_Experimenter",
        "": "exprun_MS_Instrument",
        "": "exprun_MSFFile_expFileName",
        "": "exprun_nGeneCount",
        "": "exprun_nGPGroupCount",
        "": "exprun_niBAQ_0_Total",
        "": "exprun_niBAQ_1_Total",
        "": "exprun_nMSFilesCount",
        "": "exprun_nMSFilesCount Total",
        "": "exprun_nMShrs",
        "": "exprun_nMShrs Total",
        "": "exprun_nPSMCount",
        "": "exprun_nTechRepeats",
        "": "exprun_PSMCount_unmatched",
        "": "exprun_PSMsFile_expFileName",
        "": "exprun_Purpose",
        "": "exprun_Search_Comments",
        "": "exprun_Search_EnzymeSetting",
        "": "exprun_Search_RefDatabase",
        "": "exprun_TaxonID",
        "": "exprun_Vali",
        "": "iSPEC_Experiments::exp_Extract_CellTissue",
    }

    #
    df = pd.read_excel(ispec_export_file, nrows=20000)
    df = df[~df.exprun_EXPRecNo.isna()]
    df = df.sample(490)
    # ==================================================

    def make_model(row):

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
        #
        if check_if_recno_exists(_recno) == False:
            logger.warning(f"{_recno} does not exist")
            return
        experiment = crud.get_experiment_by_recno(get_db(), _recno)
        if experiment is None:
            logger.warning(f"{_recno} does not exist")
            return
        #
        #
        kwargs["experiment_id"] = experiment.id
        exprun = crud.get_exprun_by_recrun(get_db(), _recno, _runno)
        if exprun is not None:
            logger.warning(f"{_recno}:{_runno} already exists")
            return
        exprun = models.ExperimentRun(**kwargs)
        return exprun

    expruns = df.apply(make_model, axis=1)
    expruns = filter(None, expruns)
    with get_db() as db:
        mapper = map(lambda x: crud.create_exprun(db, x), expruns)
        # walk
        [_ for _ in mapper]


# old, not using
def import_ispec_experiments(ispec_export_file: Path):
    """

    exp_EXPRecNo
    exp_EXPLabelFLAG
    exp_Extract_Genotype
    exp_EXPLabelType
    exp_ExpType
    exp_Extract_No
    exp_Extract_Adjustments
    exp_Extract_Amount
    exp_Extract_CellTissue
    exp_Extract_Fractions
    exp_Extract_Genotype
    exp_Digest_Enzyme
    exp_Digest_Experimenter
    exp_Digest_Type
    exp_Exp_Date
    exp_Separation_1
    exp_Separation_1Detail
    exp_Separation_2
    exp_Separation_2Detail
    exp_Separation_3
    exp_Separation_3Detail

    iSPEC_ExperimentRuns::exprun_EXPRunNo
    iSPEC_ExperimentRuns::exprun_MS_Instrument



    """

    ispec_export_file = "./data/20221904_ispec_cur_Experiments_metadata_export.xlsx"
    df = pd.read_excel(ispec_export_file)
    df = df[~df.exp_EXPRecNo.isna()]
    df = df.sample(490)

    column_mapping = {
        "recno": "exp_EXPRecNo",
        # "label": "exp_EXPLabelFLAG",
        "geno": "exp_Extract_Genotype",
        "label": "exp_EXPLabelType",
        # "": "exp_ExpType",
        "extractno": "exp_Extract_No",
        # "": "exp_Extract_Adjustments",
        # "": "exp_Extract_Amount",
        "cell_tissue": "exp_Extract_CellTissue",
        # "": "exp_Extract_Fractions",
        # "": "exp_Digest_Enzyme",
        # "": "exp_Digest_Experimenter",
        # "": "exp_Digest_Type",
        "date": "exp_Exp_Date",
        # "": "exp_Separation_1",
        # "": "exp_Separation_1Detail",
        # "": "exp_Separation_2,
        # "": "exp_Separation_2Detail,
        # "": "exp_Separation_3,
        # "": "exp_Separation_3Detail,
    }
    _recnos = None

    @lru_cache()
    def _get_recnos(key="default"):
        exps = crud.get_all_experiments(get_db())
        _recnos = [x.recno for x in exps]
        return _recnos

    def check_if_recno_exists(recno):
        recnos = _get_recnos()
        return recno in recnos

    def make_model(row):

        kwargs = dict()
        # add all values
        for dbcol, col in column_mapping.items():
            if col not in row:
                continue
            kwargs[dbcol] = row[col]
        #
        # check
        if check_if_recno_exists(row.exp_EXPRecNo) == True:
            logger.warning(f"{row.exp_EXPRecNo} already exists")
            return
        exp = models.Experiment(**kwargs)
        print(exp)
        return exp

    exps = df.apply(make_model, axis=1)
    exps = filter(None, exps)
    with get_db() as db:
        mapper = map(lambda exp: crud.create_exp(db, exp), exps)
        # walk
        [_ for _ in mapper]


# old not using
def import_e2g(e2g_qual_file: Path, e2g_quant_file: Path):
    """Import a single e2g file (one rec-run)"""
    column_mapping = {
        "recno": "EXPRecNo",
        "runno": "EXPRunNo",
        "searchno": "EXPSearchNo",
        # "gene": "GeneID",
        "": "LabelFLAG",
        "": "ProteinRef_GIDGroupCount",
        "": "TaxonID",
        "": "SRA",
        "gpgroup": "GPGroup",
        "": "GPGroups_All",
        "": "IDGroup",
        "": "IDGroup_u2g",
        "": "ProteinGI_GIDGroupCount",
        "": "HIDs",
        "psms_s": "PSMs_S",
        "psms_s_u2g": "PSMs_S_u2g",
        "peptidecount": "PeptideCount",
        "peptidecount_s": "PeptideCount_S",
        "peptidecount_s_u2g": "PeptideCount_S_u2g",
        "peptideprint": "PeptidePrint",
        "": "IDSet",
        "": "Coverage_u2g",
        "": "Symbol",
        "": "Coverage",
        "": "ProteinGIs",
        "": "Description",
        "": "PSMs",
        "": "ProteinRefs",
        "": "PSMs_S",
        "": "HomologeneID",
        "": "PeptideCount_u2g",
        "": "GeneSymbol",
        "": "PeptidePrint",
        "": "GeneCapacity",
        "areasum_u2g_0": "AreaSum_u2g_0",
        "areasum_u2g_all": "AreaSum_u2g_all",
        "areasum_max": "AreaSum_max",
        "areasum_dstrAdj": "AreaSum_dstrAdj",
        "ibaq_dstradj": "iBAQ_dstrAdj",
        # "proteingi_gidgroups": "ProteinGI_GIDGroups",
        # "proteinref_gidgroups": "ProteinRef_GIDGroups",
    }
    e2g_quant_cols = [
        "EXPRecNo",
        "EXPRunNo",
        "EXPSearchNo",
        "LabelFLAG",
        "GeneID",
        "SRA",
        "AreaSum_u2g_0",
        "AreaSum_u2g_all",
        "AreaSum_max",
        "AreaSum_dstrAdj",
        "iBAQ_dstrAdj",
    ]

    df = pd.read_table(e2g_qual_file)
    dfquant = pd.read_table(e2g_quant_file)
    if len(df) == 0:
        raise ValueError("No data")
    recno = df.iloc[0]["EXPRecNo"]
    runno = df.iloc[0]["EXPRunNo"]

    if check_if_recno_exists(recno) == False:
        logger.warning(f"{recno} does not exist")
        return
    experimentrun = crud.get_exprun_by_recrun(
        get_db(), recno=int(recno), runno=int(runno)
    )

    if experimentrun is None:
        logger.warning(f"{recno}:{runno} does not exist")
        return
    experimentrun = experimentrun[0]

    # ================================
    def make_model(row, model):
        kwargs = dict()
        for dbcol, col in column_mapping.items():
            if dbcol == "":
                continue
            if col not in row:
                continue
            kwargs[dbcol] = row[col]

        recno = row["EXPRecNo"]
        runno = row["EXPRunNo"]

        # do we need this here?
        experimentrun = crud.get_exprun_by_recrun(get_db(), recno=recno, runno=runno)
        if experimentrun is None:  # should have been checked before
            return
        experimentrun = experimentrun[0]
        #
        kwargs["experimentrun_id"] = experimentrun.id
        kwargs["experiment_id"] = experimentrun.experiment_id
        return model(**kwargs)
        # return models.E2GQual(**kwargs)
        #
        #

    e2gquals = df.apply(make_model, axis=1, model=models.E2GQual)
    e2gquals = [x for x in filter(None, e2gquals)]

    ## e2g quant
    e2gquants = dfquant.apply(make_model, axis=1, model=models.E2GQuant)
    e2gquants = [x for x in filter(None, e2gquants)]
    import ipdb

    ipdb.set_trace()

    with get_db() as db:
        db.add_all(e2gquals)
        db.add_all(e2gquants)
        db.commit()
        # mapper = map(lambda x: crud.create_exp(db, exp), exps)
        ## walk
        # [_ for _ in mapper]


#
