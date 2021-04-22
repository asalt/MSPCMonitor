from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    BigInteger,
    ForeignKey,
    Float,
    SmallInteger
)
from sqlalchemy.orm import relationship

from .database import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    qc_recno = Column(BigInteger)

    #rawfiles = relationship("RawFile", back_populates="instrument")
    rawfiles = relationship("RawFile")


class RawFile(Base):

    __tablename__ = "rawfiles"

    id = Column(Integer, primary_key=True)
    # fullpath = CharField(max_length=255, unique=True)
    name = Column(String)
    ctime = Column(DateTime)
    mtime = Column(DateTime)
    processed = Column(Boolean)
    psms = Column(Integer)
    #instrument_id = Column(Integer, ForeignKey("instruments.id"))

    #instrument = relationship("Instrument", back_populates="rawfiles")
    instrument_id = Column(Integer, ForeignKey('instruments.id'))
    runno_id = Column(Integer, ForeignKey('experimentruns.id'))
    #instrument = relationship("Instrument")

    # filename = CharField(max_length=255)
    # exprec = ForeignKeyField(Experiment, backref='rawfiles')
    # exprun     = ForeignKeyField(ExpRun, backref='rawfiles')
    # birth = DateTimeField(null=True)
    # size = FloatField(null=True)
    # instrument = CharField(null=True)

class Experiment(Base):

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    recno = Column(BigInteger)
    label = Column(String)

    expruns = relationship("ExperimentRun", backref='experiment')
    #, back_populates="experiments")

class ExperimentRun(Base):

    __tablename__ = "experimentruns"

    id = Column(Integer, primary_key=True)
    runno = Column(Integer)
    searchno = Column(Integer)
    is_plotted = Column(Boolean, default=False)
    is_searched = Column(Boolean, default=False)  # make more?
    is_validated = Column(Boolean, default=False)  # make more?
    is_grouped = Column(Boolean, default=False)  # make more?

    taxon = Column(String)
    refdb = Column(String)
   
    #recno = relationship("Experiment", back_populates="experimentruns")
    recno_id = Column(Integer, ForeignKey('experiments.id'))
    rawfiles = relationship("RawFile")
    #, back_populates="experimentruns")


class MSFraggerParams(Base):

    __tablename__ = "msfraggerparams"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    precursor_mass_lower = Column(Float)                  # Lower bound of the precursor mass window.
    precursor_mass_upper = Column(Float)                   # Upper bound of the precursor mass window.
    precursor_mass_units = Column(SmallInteger)                    # Precursor mass tolerance units (0 for Da, 1 for ppm, 2 for DIA-MS1, 3 for DIA-all).
    precursor_true_tolerance = Column(Float)               # True precursor mass tolerance (window is +/- this value).
    precursor_true_units = Column(SmallInteger)                    # True precursor mass tolerance units (0 for Da, 1 for ppm).
    fragment_mass_tolerance = Column(Float)               # Fragment mass tolerance (window is +/- this value).
    fragment_mass_units = Column(SmallInteger)                     # Fragment mass tolerance units (0 for Da, 1 for ppm).
    calibrate_mass = Column(SmallInteger)                          # Perform mass calibration (0 for OFF, 1 for ON, 2 for ON and find optimal parameters).

    modi01 = Column(Integer, ForeignKey("modifications.id"))
    modi02 = Column(Integer, ForeignKey("modifications.id"))
    modi03 = Column(Integer, ForeignKey("modifications.id"))
    modi04 = Column(Integer, ForeignKey("modifications.id"))
    modi05 = Column(Integer, ForeignKey("modifications.id"))
    modi06 = Column(Integer, ForeignKey("modifications.id"))
    modi07 = Column(Integer, ForeignKey("modifications.id"))


class Modification(Base):

    __tablename__ = "modifications"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    mass_shift = Column(Float)
    position_str = Column(String)
    ntotal = Column(SmallInteger)