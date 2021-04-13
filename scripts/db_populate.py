# test
from mspcmonitor import models, schemas, database, crud
from mspcmonitor.database import SessionLocal, engine



models.Base.metadata.create_all(bind=engine)

db = SessionLocal()

inst = schemas.InstrumentCreate(name='test', qc_recno=12345)
crud.create_instrument(db, inst)