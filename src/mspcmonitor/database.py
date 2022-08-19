"""
database.py
this is where we define what database (including connection) we are using
"""
import os
# from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlmodel import create_engine

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
SQLALCHEMY_DATABASE_URL = "sqlite:///ispec.db"

# INHERIT_CACHE= True

connect_args={"check_same_thread": True,
}
# this doesn't work when engine is already created it seems
def get_engine(**kws):
    return create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args, **kws)


ECHO = os.environ.get("ISPEC_DB_ECHO", False)
ECHO = True if ECHO in ('T', 'True', 'TRUE') else False

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args, echo=ECHO,
)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()
