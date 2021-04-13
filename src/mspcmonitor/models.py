from sqlalchemy import Column, String, DateTime, Boolean, Integer, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base

class Instrument(Base):
    __tablename__ = 'instruments'

    id = Column(Integer, primary_key=True)
    name           = Column(String)
    qc_recno       = Column(BigInteger)

    rawfiles = relationship("RawFile", back_populates="instrument")

class RawFile(Base):

    __tablename__ = 'rawfiles'

    id = Column(Integer, primary_key=True)
    #fullpath = CharField(max_length=255, unique=True)
    name = Column(String)
    ctime = Column(DateTime)
    mtime = Column(DateTime)
    processed = Column(Boolean)
    psms = Column(Integer)
    instrument_id = Column(Integer, ForeignKey('instruments.id'))

    instrument = relationship("Instrument", back_populates='rawfiles')

    # filename = CharField(max_length=255)
    # exprec = ForeignKeyField(Experiment, backref='rawfiles')
    # exprun     = ForeignKeyField(ExpRun, backref='rawfiles')
    # birth = DateTimeField(null=True)
    # size = FloatField(null=True)
    # instrument = CharField(null=True)