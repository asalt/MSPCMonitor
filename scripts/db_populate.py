# test
from mspcmonitor import models, schemas, database, crud
from mspcmonitor.models import (
    Experiment,
    ExperimentRun,
    Instrument,
    # create_database_and_data,
)
from mspcmonitor import crud
from mspcmonitor.database import engine


# from mspcmonitor.database import SessionLocal, engine

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlmodel import create_engine
from sqlmodel import Field, Session, SQLModel, Relationship


def create_data():
    with Session(engine) as session:
        inst = Instrument(name="amfusion", qc_recno=99999, id=1)
        session.add(inst)

        #  inst = Instrument(name="LumosETD", qc_recno=99995)
        #  session.add(inst)

        Experiment(id=1, recno=12345, label="none")
        exp = Experiment(id=1, recno=12345, label="none")
        session.add(exp)
        session.commit()

        print(exp.id)
        exprun = ExperimentRun(
            runno=1, searchno=1, taxon="9606", refdb=".", experiment_id=exp.id
        )
        session.add(exprun)

        session.commit()


def read_data():
    with Session(engine) as session:
        res = crud.get_all_instruments(session)
        print(res)
        import ipdb

        # ipdb.set_trace()


if __name__ == "__main__":
    create_data()
    read_data()


# =============================
# =============================

# # old not needed anymore
#
# models.Base.metadata.create_all(bind=engine)
#
# db = SessionLocal()
#
# # inst = schemas.InstrumentCreate(name='test', qc_recno=12345)
# inst = schemas.InstrumentCreate(name="amfusion", qc_recno=99999)
# crud.create_instrument(db, inst)
#
# inst = schemas.InstrumentCreate(name="LumosETD", qc_recno=99995)
# crud.create_instrument(db, inst)
#
#
# exp = schemas.ExperimentCreate(
#     recno=12345,
#     label="none",
# )
# crud.create_experiment(db, exp)
#
# exprun = schemas.ExperimentRunCreate(runno=1, searchno=1, taxon="9606", refdb=".")
# crud.create_experimentrun(db, exprun, recno=12345)
#
