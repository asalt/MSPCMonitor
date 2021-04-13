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

models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



MIN_RAWFILE_SIZE = 5 * 10 ** 8
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


def monitor_files(path, last_time=None, maxsize=10) -> list:
    if last_time is None:
        last_time = datetime(1, 1, 1, 0, 0, 0)
    new_rawfiles = list()

    nfound = 0
    for rawfile in path.glob("*raw"):
        statres = rawfile.stat()

        #logger.debug(f"{rawfile.name} {statres}")
        #logger.debug(f"{statres.st_mtime} gt {last_time.timestamp()} \
        #            and {statres.st_size} gt {MIN_RAWFILE_SIZE} \
        #            and {statres.st_mtime} sub {statres.st_ctime} / 60 gt {MIN_WAIT_TIME} \
        #            ")
                    
        if (
            statres.st_mtime > last_time.timestamp()
            and statres.st_size > MIN_RAWFILE_SIZE
            and (statres.st_mtime - statres.st_ctime) / 60 > MIN_WAIT_TIME
        ):
            logger.info(f"Found {rawfile}")
            new_rawfiles.append(rawfile)
            nfound += 1

    return new_rawfiles


def move_files(files: List[Path], workdir: Path, dry=False):
    new_files = list()
    for file in files:
        target = workdir / Path(file.name)

        if not dry:
            logger.info(f"Copying {file} to {target}")
            if not target.exists():
                shutil.copy2(file, target)
            elif target.exists():
                logger.info(f"{target} already exists. Not moving")
        elif dry:
            logger.info(f"DRY RUN - Copying {file} to {target}")
        new_files.append(target)
    return new_files


def work(path, dry=False):

    last_time = get_last_time(path) or datetime(1990, 1, 1, 1, 0, 0)

    new_rawfiles = monitor_files(path, last_time, maxsize=MAX_BATCH_SIZE)

    if len(new_rawfiles) == 0:
        return

    workdir = get_workdir()
    staged_rawfiles = move_files(new_rawfiles, workdir, dry=dry)

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
        #'percolate',
        #'quant'
    ]

    logger.info(f"Running {QC_CMD}")
    if dry:
        logger.info(f"Dry run, exiting")
        return

    CMD = QC_CMD
    result = subprocess.run(
        CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    print(result.returncode, result.stdout, result.stderr)

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
    if APPDIR.exists():
        print(f"Monitoring {path}")

    dry = True
    schedule.every(2).seconds.do(work, path=path, dry=dry)

    while True:
        schedule.run_pending()
        time.sleep(1)
    # worker(path.resolve())


@app.callback()
def main():
    pass