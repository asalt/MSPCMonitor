from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

from fastapi import FastAPI

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "MSPCMonitor"}



@app.get("/instruments/{instrument_id}")
def get_instrument(instrument_id: str):
    instrument = {'instrument_id': instrument_id}
    return instrument

@app.post("/instruments/", response_model=schemas.Instrument)
def create_instrument(instrument: schemas.InstrumentCreate, db: Session = Depends(get_db)):
    db_instrument = crud.get_instrument_by_name(db, name=instrument.name)
    if db_instrument:
        raise HTTPException(status_code=400, detail="Instrument already exists")
    return crud.create_instrument(db=db, instrument=instrument)

@app.post("/instrument/{instrument_id}/rawfiles", response_model=schemas.RawFile)
def create_rawfile(instrument_id: int, rawfile: schemas.RawFileCreate, db: Session = Depends(get_db)):
    return crud.create_rawfile(db=db, rawfile=rawfile, instrument_id=instrument_id)

@app.get("/rawfiles/", response_model=List[schemas.RawFile])
def get_rawfile(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    rawfiles = crud.get_rawfiles(db, skip=skip, limit=limit)
    return rawfiles