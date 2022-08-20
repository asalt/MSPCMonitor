# from sqlalchemy.orm import Session
from typing import List
from tqdm import tqdm
import sqlmodel
from sqlmodel import Session

from . import models, schemas


def add_and_commit(db: Session, obj):
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def add_multiple_and_commit(db: Session, objs):
    # this is not useful actually
    for obj in tqdm(objs, 'adding objects'):
        db.add(obj) # is this better than calling commit after adding each object?
    db.commit()
    db.refresh


def get_rawfile_by_name(db: Session, name: str):
    return db.query(models.RawFile).filter(models.RawFile.name == name).first()


def get_all_instruments(db: Session):
    print(db.query(models.Instrument).all())
    return db.query(models.Instrument).all()


def get_instrument_by_name(db: Session, name: str):
    return db.query(models.Instrument).filter(models.Instrument.name == name).first()


def get_instrument_by_qc_recno(db: Session, qc_recno: int):
    return (
        db.query(models.Instrument)
        .filter(models.Instrument.qc_recno == qc_recno)
        .first()
    )


def get_experiment_by_recno(db: Session, recno: int):
    return db.query(models.Experiment).filter(models.Experiment.recno == recno).first()


def get_all_experiments(db: Session):
    return db.query(models.Experiment).all()

def get_gene_by_id(db: Session, geneid):
    return db.query(models.Gene).filter(models.Gene.geneid == geneid).first()

def get_all_genes(db:Session):
    return db.query(models.Gene).all()

def get_all_experimentruns(db: Session):
    return db.query(models.ExperimentRun).join(models.Experiment).all()


def get_exprun_by_recrunsearch(db: Session, recno: int, runno: int, searchno:int):

    # if not isinstance(recno, int):
    #     raise TypeError(f"recno should be of type int")
    # if not isinstance(runno, int):
    #     raise TypeError(f"recno should be of type int")

    statement = (
        sqlmodel.select(models.ExperimentRun, models.Experiment)
        .where(models.Experiment.recno == recno and models.ExperimentRun.runno == runno and
                models.ExperimentRun.searchno == searchno
            )
            .join(models.ExperimentRun)
    )
        #.filter(models.ExperimentRun.runno == runno)

    qq = db.exec(statement)
    return qq.first()
    # return (
    #     db.query(models.ExperimentRun)
    #     .filter(models.ExperimentRun.recno == recno)
    #     .join(models.Experiment)
    #     .filter(
    #         models.Experiment.recno == recno and models.ExperimentRun.runno == runno
    #     )
    # ).first()


def get_unprocessed_data(db: Session):
    return (
        db.query(models.ExperimentRun)
        .filter(models.ExperimentRun.is_plotted == False)
        .join(models.Experiment)
    )


def create_instrument(db: Session, instrument: schemas.InstrumentCreate):
    db_instrument = models.Instrument(
        name=instrument.name, qc_recno=instrument.qc_recno
    )
    db.add(db_instrument)
    db.commit()
    db.refresh(db_instrument)
    return db_instrument


# def create_experiment(db: Session, experiment: schemas.ExperimentCreate):
#     db_exp = models.Experiment(recno=experiment.recno, label=experiment.label)
#     return add_and_commit(db, db_exp)


def create_experimentrun(
    db: Session, experimentrun: schemas.ExperimentRunCreate, recno: int
):
    db_exp = get_experiment_by_recno(db, recno)
    if db_exp is None:
        raise AttributeError("Experiment not found")
    db_exprun = models.ExperimentRun(
        runno=experimentrun.runno,
        searchno=experimentrun.searchno,
        # taxon=experimentrun.taxon,
        refdb=experimentrun.refdb,
        experiment_id=db_exp.id,
    )
    # db.add(db_exprun)
    # db.commit()
    # db.refresh(db_exprun)
    # return db_exprun
    return add_and_commit(db, db_exprun)


# def create_rawfile(db: Session, rawfile: schemas.RawFileCreate, instrument_id: int):
#     db_rawfile = models.RawFile(**rawfile.dict(), instrument_id=instrument_id)
#     db.add(db_rawfile)
#     db.commit()
#     db.refresh(db_rawfile)
#     return db_rawfile


def create_rawfile(db: Session, rawfile: models.RawFile):
    return add_and_commit(db, rawfile)


def create_experiment(db: Session, experiment: models.Experiment):
    return add_and_commit(db, experiment)


def create_exp(db: Session, exp: models.Experiment):
    return create_experiment(db, exp)


# def update_rawfile(db: Session, rawfile_name: str, processed=False, psms=None):
#     rawfile = get_rawfile_by_name(db, rawfile_name)
#     rawfile.processed = processed
#     rawfile.psms = psms
#     db.commit()


def get_rawfiles(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.RawFile).offset(skip).limit(limit).all()


def create_tables(db: Session):
    models.create_db_and_tables()

    # db_exp = models.Experiment(recno=exp.recno, label=exp.label)


# def create_exprun(db: Session, exprun: models.ExperimentRun):
#     return add_and_commit(db, exprun)
