from sqlalchemy.orm import Session

from . import models, schemas

def add_and_commit(db, obj):
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_rawfile_by_name(db: Session, name: str):
    return db.query(models.RawFile).filter(models.RawFile.name == name).first()


def get_instrument_by_name(db: Session, name: str):
    return db.query(models.Instrument).filter(models.Instrument.name == name).first()

def get_instrument_by_qc_recno(db: Session, qc_recno: int):
    return db.query(models.Instrument).filter(models.Instrument.qc_recno == qc_recno).first()

def get_experiment_by_recno(db: Session, recno: int):
    return db.query(models.Experiment).filter(models.Experiment.recno == recno).first()


def create_instrument(db: Session, instrument: schemas.InstrumentCreate):
    db_instrument = models.Instrument(
        name=instrument.name, qc_recno=instrument.qc_recno
    )
    db.add(db_instrument)
    db.commit()
    db.refresh(db_instrument)
    return db_instrument

def create_experiment(db: Session, experiment: schemas.ExperimentCreate):
    db_exp = models.Experiment(
        recno = experiment.recno,
        label = experiment.label
    )
    return add_and_commit(db, db_exp)
    
def create_experimentrun(db: Session, experimentrun: schemas.ExperimentRunCreate,
    recno: int):
    db_exp = get_experiment_by_recno(db, recno)
    db_exprun = models.ExperimentRun(
        runno = experimentrun.runno,
        searchno = experimentrun.searchno,
        taxon = experimentrun.taxon,
        refdb = experimentrun.taxon,
        recno_id = db_exp.id
    )
    return add_and_commit(db, db_exprun)


def create_rawfile(db: Session, rawfile: schemas.RawFileCreate, instrument_id: int):
    db_rawfile = models.RawFile(**rawfile.dict(), instrument_id=instrument_id)
    db.add(db_rawfile)
    db.commit()
    db.refresh(db_rawfile)
    return db_rawfile

def update_rawfile(db: Session, rawfile_name: str, processed=False, psms=None):
    rawfile = get_rawfile_by_name(db, rawfile_name)
    rawfile.processed = processed
    rawfile.psms = psms
    db.commit()

def get_rawfiles(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.RawFile).offset(skip).limit(limit).all()
