import sys
import os
import re
from enum import Enum
from pathlib import Path
import configparser
from glob import iglob, glob
import time
import subprocess
import shutil
from datetime import datetime
from typing import Optional, List, Tuple

import schedule

import click
import typer

import logging


from . import crud, models, schemas
from .database import SessionLocal, engine

MSFileInfoScanner = "~/tools/MSFileInfoScanner_Program/MSFileInfoScanner.exe"

models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


MIN_RAWFILE_SIZE = 5 * 10**8
MIN_WAIT_TIME = 100  # minutes
MAX_BATCH_SIZE = 4


def get_logger(name=__name__):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("MSPCRunner.log")

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = get_logger()

APPDIR = Path(typer.get_app_dir("mspcmonitor"))
TIMESHEET = APPDIR / "timesheet.txt"
if not TIMESHEET.exists():
    if not TIMESHEET.parent.exists():
        TIMESHEET.parent.mkdir(parents=True)
    TIMESHEET.touch()
    TIMESHEET.write_text(
        """
    [log]\n
    [workdir]\n
    """
    )

WORKDIR = None

config = configparser.RawConfigParser(delimiters=("?", "="))
config.optionxform = lambda option: option
config.read(TIMESHEET)

app = typer.Typer(chain=True)
BASEDIR = Path(os.path.split(__file__)[0])


@app.command()
def set_workdir(path: Path, config=config):
    config["workdir"]["processing"] = str(path.resolve())
    with open(TIMESHEET, "w") as f:
        config.write(f)
    logger.info(f"Updated ['workdir']['processing'] to {path} ")


@app.command()
def get_workdir(config=config):
    res = config["workdir"].get("processing")
    logger.info(f"['workdir']['processing'] : {res} ")

    return res


def get_last_time(path, config=config):
    res = config["log"].get(str(path))
    if res is not None:
        return datetime.strptime(res, "%Y-%m-%d %H:%M:%S")


def update_time(path, config=config, time=None):
    if time is None:
        time = datetime.now()
    config["log"][path] = time.isoformat(timespec="seconds", sep=" ")
    return


def monitor_files(path, last_time=None, maxsize=10, inst_id=99995) -> list:
    if last_time is None:
        last_time = datetime(1, 1, 1, 0, 0, 0)
    new_rawfiles = list()

    nfound = 0
    for rawfile in path.glob("*raw"):

        statres = rawfile.stat()

        # logger.debug(f"{rawfile.name} {statres}")
        # logger.debug(f"{statres.st_mtime} gt {last_time.timestamp()} \
        #            and {statres.st_size} gt {MIN_RAWFILE_SIZE} \
        #            and {statres.st_mtime} sub {statres.st_ctime} / 60 gt {MIN_WAIT_TIME} \
        #            ")

        if not (
            statres.st_mtime
            > last_time.timestamp()  # change to some min timestamp so we
            and statres.st_size
            > MIN_RAWFILE_SIZE  # don't process all rawfiles in existance
            and (statres.st_mtime - statres.st_ctime) / 60 > MIN_WAIT_TIME
        ):
            continue

        rawfile_rec = crud.get_rawfile_by_name(get_db(), rawfile.name)
        # import ipdb; ipdb.set_trace()
        if rawfile_rec is not None and rawfile_rec.processed == True:
            continue  # already exists and has been processed

        logger.info(f"Found {rawfile}")

        if rawfile_rec is None:  # need to add

            inst = crud.get_instrument_by_qc_recno(get_db(), qc_recno=inst_id)

            rawfile_rec = schemas.RawFileCreate(
                name=rawfile.name,
                # ctime=datetime.strptime(statres.st_ctime, "%Y-%m-%d %H:%M:%S"),
                # mtime=datetime.strptime(statres.st_mtime, "%Y-%m-%d %H:%M:%S"),
                ctime=datetime.fromtimestamp(statres.st_ctime),
                mtime=datetime.fromtimestamp(statres.st_mtime),
                # ctime=statres.st_ctime,
                # mtime=statres.st_mtime,
            )
            crud.create_rawfile(get_db(), rawfile_rec, instrument_id=inst.id)
            logger.info(f"Saved {rawfile} record in db")

        new_rawfiles.append(rawfile)
        nfound += 1

    return new_rawfiles


def move_files(files: List[Path], workdir: Path, dry=False):
    new_files = list()
    for file in files:
        target = workdir / Path(file.name)

        if not dry:
            if not target.exists():
                logger.info(f"Copying {file} to {target}")
                shutil.copy2(file, target)
            elif target.exists():
                logger.info(f"{target} already exists. Not moving")
        elif dry:
            logger.info(f"DRY RUN - Copying {file} to {target}")
        new_files.append(target)
    return new_files


def summarize_results(p):

    outfile = p.parent.glob(f"{p.name[:10]}*MSPCRunner*txt")
    outfile = list(outfile)
    if len(outfile) != 1:
        logger.error(f"Could not find MSPCRunner output for {p}")
        return

    import pandas as pd

    df = pd.read_table(outfile[0])
    p.name[:10]

    n_psms = len(df)

    rawfile_rec = crud.get_rawfile_by_name(get_db(), p.name)
    rawfile_rec.psms = n_psms
    rawfile_rec.commit()


def work(path, dry=False):
    # TODO: look for existance of fragger and percolator outputs
    # save completion in db
    # do not re-run if data already exists
    last_time = get_last_time(path) or datetime(1990, 1, 1, 1, 0, 0)

    new_rawfiles = monitor_files(path, last_time, maxsize=MAX_BATCH_SIZE)

    if len(new_rawfiles) == 0:
        return

    workdir = get_workdir()
    staged_rawfiles = move_files(new_rawfiles, workdir, dry=dry)
    # now that they are staged, let's add to db
    # for staged_rawfile in staged_rawfiles:

    QC_CMD = [
        "mspcrunner",
        *[
            b
            for a in [("--rawfile", str(f.resolve())) for f in staged_rawfiles]
            for b in a
        ],
        "search",
        "--preset",
        "OTIT-hs",
        "--ramalloc",
        "25",
        "percolate",
        #'quant'
    ]

    logger.info(f"Running {QC_CMD}")
    if dry:
        logger.info(f"Dry run, exiting")
        return

    CMD = QC_CMD
    result = subprocess.run(
        CMD,
        capture_output=True,
        universal_newlines=True,
        # stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    # print(result.returncode, result.stdout, result.stderr)

    # TODO make nicer
    from mspcrunner import psm_merge

    psm_merge.main(workdir)

    for f in staged_rawfiles:
        summarize_results(f)
        crud.update_rawfile(f.name, processed=True)
    # summarize_results(f)
    # subprocess.run
    # import ipdb; ipdb.set_trace()

    # capture_output=True) need py3.7
    # *args, **kwargs,    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


@app.command()
def watch(
    dry: bool = typer.Option(
        False, "--dry", help="Dry run, do not actually execute commands"
    ),
    path: Path = typer.Argument(
        default=None,
        help="Path with raw files to process. Will process all raw files in path.",
    ),
):
    ## checking inputs
    if path is None:
        raise ValueError("[path] is required")

    if APPDIR.exists():
        print(f"Monitoring {path}")

    ## do work
    # schedule.run_pending()
    work(path=path, dry=dry)

    schedule.every(30).minutes.do(work, path=path, dry=dry)

    while True:
        schedule.run_pending()
        time.sleep(1)
    # worker(path.resolve())


def scan(f):
    CMD = [
        MSFileInfoScanner,
        "/I:{f.'",
    ]


# @app.command()
# def run_api():
#     from .api import app as api_app
#     return api_app


@app.command()
def archive(
    path: Path = typer.Argument(
        default=None,
        help="Path with raw files to archive. Will process all raw files in path.",
    ),
):
    pass


@app.callback()
def main():
    pass
