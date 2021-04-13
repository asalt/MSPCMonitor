from sqlalchemy.orm import Session

from . import models, schemas



def get_rawfile_by_name(db: Session, name: str):
    return db.query(models.RawFile).filter(models.RawFile.name == name).first()

def get_instrument_by_name(db: Session, name: str):
    return db.query(models.Instrument).filter(models.Instrument.name == name).first()

def create_instrument(db: Session, instrument: schemas.InstrumentCreate):
    db_instrument = models.Instrument(name=instrument.name, qc_recno = instrument.qc_recno)
    db.add(db_instrument)
    db.commit()
    db.refresh(db_instrument)
    return db_instrument


def create_rawfile(db: Session, rawfile: schemas.RawFileCreate, instrument_id: int):
    db_rawfile = models.RawFile(**rawfile.dict(), instrument_id=instrument_id)
    db.add(db_rawfile)
    db.commit()
    db.refresh(db_rawfile)
    return db_rawfile

def get_rawfiles(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.RawFile).offset(skip).limit(limit).all()