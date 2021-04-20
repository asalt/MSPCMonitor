# test
from mspcmonitor import models, schemas, database, crud
#from mspcmonitor.database import SessionLocal, engine

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# =============================


models.Base.metadata.create_all(bind=engine)

db = SessionLocal()

#inst = schemas.InstrumentCreate(name='test', qc_recno=12345)
inst = schemas.InstrumentCreate(name='amfusion', qc_recno=99999)
crud.create_instrument(db, inst)

inst = schemas.InstrumentCreate(name='LumosETD', qc_recno=99995)
crud.create_instrument(db, inst)


exp = schemas.ExperimentCreate(recno=12345,
                        label='none',
)
crud.create_experiment(db, exp)

exprun = schemas.ExperimentRunCreate(
    runno=1,
    searchno=1,
    taxon='9606',
    refdb = '.'
)
crud.create_experimentrun(db, exprun, recno=12345)