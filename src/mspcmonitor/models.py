from sqlite3 import Date
from matplotlib.pyplot import table
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    BigInteger,
    ForeignKey,
    Float,
    SmallInteger,
)
from sqlalchemy.orm import relationship
from typing import List, Optional, Dict, Any, Union, Tuple
from sqlmodel import create_engine
from sqlmodel import Field, Session, SQLModel, Relationship


# ========================================

# import crud
# from .database import Base, engine


# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
#
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, echo=True
# )
from .database import engine

# engine = engine
# ========================================


# these are the slots for PyDantic FieldInfo that can be added as kwargs to Field
#
#         'default',
#         'default_factory',
#         'alias',
#         'alias_priority',
#         'title',
#         'description',
#         'exclude',
#         'include',
#         'const',
#         'gt',
#         'ge',
#         'lt',
#         'le',
#         'multiple_of',
#         'max_digits',
#         'decimal_places',
#         'min_items',
#         'max_items',
#         'unique_items',
#         'min_length',
#         'max_length',
#         'allow_mutation',
#         'repr',
#         'regex',
#         'discriminator',
#         'extra',


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# class Instrument(Base):
class Instrument(SQLModel, table=True):
    __tablename__ = "instrument"

    class Meta:
        load_instance = True

    # id = Column(Integer, primary_key=True)
    # name = Column(String)
    # qc_recno = Column(BigInteger)
    id: int = Field(primary_key=True)
    name: str = Field(default=None)
    qc_recno: int = Field(default=None)

    # rawfiles = relationship("RawFile", back_populates="instrument")
    # rawfiles = relationship("RawFile")
    rawfiles: List["RawFile"] = Relationship(back_populates="instrument")


# class RawFile(Base):
class RawFile(SQLModel, table=True):
    __tablename__ = "rawfile"

    class Meta:
        load_instance = True

    id: int = Field(primary_key=True)
    name: str
    # TODO datetime
    ctime: str
    mtime: str

    instrument_id: int = Field(default=None, foreign_key="instrument.id")
    instrument: Instrument = Relationship(back_populates="rawfiles")

    exprun_id: int = Field(default=None, foreign_key="experimentrun.id")
    exprun: "ExperimentRun" = Relationship(back_populates="rawfiles")
    ## =========================================

    # id = Column(Integer, primary_key=True)
    # name = Column(String)
    # ctime = Column(DateTime)
    # mtime = Column(DateTime)
    # processed = Column(Boolean)
    # psms = Column(Integer)

    # # instrument = relationship("Instrument", back_populates="rawfiles")
    # # instrument_id = Column(Integer, ForeignKey("instruments.id"))
    # instrument_id = Column(Integer, ForeignKey("instruments.id"))
    # runno_id = Column(Integer, ForeignKey("experimentruns.id"))
    # # instrument = relationship("Instrument")

    ## =========================================
    ## =========================================
    # filename = CharField(max_length=255)
    # exprec = ForeignKeyField(Experiment, backref='rawfiles')
    # exprun     = ForeignKeyField(ExpRun, backref='rawfiles')
    # birth = DateTimeField(null=True)
    # size = FloatField(null=True)
    # instrument = CharField(null=True)


# class Experiment(Base):
class Experiment(SQLModel, table=True):
    __tablename__ = "experiment"

    class Meta:
        load_instance = True
        include_relationships = True
        # model = SQLModel

    id: int = Field(primary_key=True)
    recno: int
    label: str = Field(default=None)
    experimentruns: Optional[List["ExperimentRun"]] = Relationship(
        back_populates="experiment"
    )
    e2gquals: Optional[List["E2GQual"]] = Relationship(back_populates="experiment")
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="experiment")
    #
    # =======================================================================
    # many to many relationship, do it later or not at all
    # genes: List["Gene"] = Relationship(
    #    back_populates="experiment"
    # )  # will this cause a performance issue?
    #
    # =======================================================================

    # id = Column(Integer, primary_key=True)
    # recno = Column(BigInteger)
    # label = Column(String)

    # expruns = relationship("ExperimentRun", backref="experiment")
    # , back_populates="experiments")


# class ExperimentRun(Base):
class ExperimentRun(SQLModel, table=True):

    __tablename__ = "experimentrun"
    id: int = Field(primary_key=True)
    runno: int = Field(default=None)
    searchno: int = Field(default=6)
    tic: bool = Field(default=False)
    # is_searched: bool = False
    # is_validated: bool = False
    # is_grouped: bool

    # taxon: str
    # refdb: str

    experiment_id: int = Field(default=None, foreign_key="experiment.id")
    experiment: Experiment = Relationship(back_populates="experimentruns")
    rawfiles: Optional[List["RawFile"]] = Relationship()
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="experimentrun")
    e2gquals: Optional[List["E2GQual"]] = Relationship(back_populates="experimentrun")

    # id = Column(Integer, primary_key=True)
    # runno = Column(Integer)
    # searchno = Column(Integer)
    # is_plotted = Column(Boolean, default=False)
    # is_searched = Column(Boolean, default=False)  # make more?
    # is_validated = Column(Boolean, default=False)  # make more?
    # is_grouped = Column(Boolean, default=False)  # make more?

    # taxon = Column(String)
    # refdb = Column(String)

    # # recno = relationship("Experiment", back_populates="experimentruns")
    # recno_id = Column(Integer, ForeignKey("experiments.id"))
    # rawfiles = relationship("RawFile")
    # # , back_populates="experimentruns")


class Gene(SQLModel, table=True):
    class Meta:
        load_instance = True

    id: int = Field(primary_key=True)
    symbol: str = Field()
    funcats: str = Field()
    # =======================================================================
    # many to many relationship, do it later or not at all
    # experiments: List[Experiment] = Relationship(back_populates="genes")
    # =======================================================================
    gene_capacity: int = Field(default=0)
    e2gquals: Optional[List["E2GQual"]] = Relationship(back_populates="geneid")
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="geneid")


class E2GQual(SQLModel, table=True):
    class Meta:
        load_instance = True

    id: int = Field(primary_key=True)
    #
    experiment: Optional[Experiment] = Relationship(back_populates="e2gquals")
    experiment_id: Optional[int] = Field(default=None, foreign_key="experiment.id")
    experimentrun: Optional[ExperimentRun] = Relationship(back_populates="e2gquals")
    experimentrun_id: Optional[int] = Field(
        default=None, foreign_key="experimentrun.id"
    )
    #
    geneid: Optional[Gene] = Relationship(back_populates="e2gquals")
    geneid_id: Optional[int] = Field(foreign_key="gene.id")
    #
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="e2gqual")
    #

    #
    # label: Optional[str] = Relationship(back_populates="label")
    # label_id: Optional[int] = Field(default=None, foreign_key="label.id")
    #
    ##
    # ProteinRef_GIDGroupCount: str = Field()
    # # TaxonID: str = Field()
    # SRA: str = Field()
    # gpgroups_all: str = Field()
    # idgroup: str = Field()
    # idgroup_u2g: str = Field()
    # proteingi_gidgroupcount: str = Field()
    # hids: str = Field()
    # peptidecount: str = Field()
    # idset: int = Field()
    # coverage_u2g: str = Field()
    # # Symbol: str = Field()
    # coverage: float = Field()
    # proteingis: str = Field()
    # description: str = Field()
    # psms: int = Field()
    # peptidecount_s: int = Field()
    # proteinrefs: int = Field()
    # psms_s: int = Field()
    # psms_s_u2g: int = Field()

    # TODO
    # homologeneid: str = Field()

    peptidecount_u2g: Optional[int] = Field()
    # GeneSymbol: str = Field()
    gpgroup: Optional[int] = Field()
    peptidecount_s_u2g: Optional[int] = Field()
    peptideprint: Optional[str] = Field()
    psms_u2g: Optional[int] = Field()
    # GeneCapacity: str = Field()
    proteingi_gidgroups: Optional[str] = Field()
    proteinref_gidgroups: Optional[str] = Field()
    # EXPRecNo	EXPRunNo	EXPSearchNo	GeneID	LabelFLAG	ProteinRef_GIDGroupCount	TaxonID	SRA	GPGroups_All	IDGroup	IDGroup_u2g	ProteinGI_GIDGroupCount	HIDs	PeptideCount	IDSet	Coverage_u2g	Symbol	Coverage	PSMs_S_u2g	ProteinGIs	Description	PSMs	PeptideCount_S	ProteinRefs	PSMs_S	HomologeneID	PeptideCount_u2g	GeneSymbol	GPGroup	PeptideCount_S_u2g	PeptidePrint	PSMs_u2g	GeneCapacity	ProteinGI_GIDGroups	ProteinRef_GIDGroups


class E2GQuant(SQLModel, table=True):
    class Meta:
        load_instance = True

    id: int = Field(primary_key=True)
    #
    experiment: Experiment = Relationship(back_populates="e2gquants")
    experiment_id: int = Field(default=None, foreign_key="experiment.id")
    experimentrun: ExperimentRun = Relationship(back_populates="e2gquants")
    experimentrun_id: int = Field(default=None, foreign_key="experimentrun.id")
    #
    geneid: Optional[Gene] = Relationship(back_populates="e2gquants")
    geneid_id: Optional[int] = Field(foreign_key="gene.id")
    #
    e2gqual_id: Optional[int] = Field(default=None, foreign_key="e2gqual.id")
    e2gqual: Optional["E2GQual"] = Relationship(back_populates="e2gquants")

    #
    # label: Optional[str] = Relationship(back_populates="label")
    # label_id: Optional[int] = Field(default=None, foreign_key="label.id")
    #
    areasum_u2g_0: float
    areasum_u2g_all: float
    areasum_max: float
    areasum_dstradj: float
    ibaq_dstradj: float


class Label(SQLModel, table=True):
    class Meta:
        load_instance = True

    id: int = Field(primary_key=True)
    name: Optional[str] = Field(default="none")
    primary_mass_shift: float = Field(default=0.0)


# EXPRecNo	EXPRunNo	EXPSearchNo	LabelFLAG	GeneID	SRA


# class MSFraggerParams(Base):
#
#     __tablename__ = "msfraggerparams"
#
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#
#     precursor_mass_lower = Column(Float)  # Lower bound of the precursor mass window.
#     precursor_mass_upper = Column(Float)  # Upper bound of the precursor mass window.
#     precursor_mass_units = Column(
#         SmallInteger
#     )  # Precursor mass tolerance units (0 for Da, 1 for ppm, 2 for DIA-MS1, 3 for DIA-all).
#     precursor_true_tolerance = Column(
#         Float
#     )  # True precursor mass tolerance (window is +/- this value).
#     precursor_true_units = Column(
#         SmallInteger
#     )  # True precursor mass tolerance units (0 for Da, 1 for ppm).
#     fragment_mass_tolerance = Column(
#         Float
#     )  # Fragment mass tolerance (window is +/- this value).
#     fragment_mass_units = Column(
#         SmallInteger
#     )  # Fragment mass tolerance units (0 for Da, 1 for ppm).
#     calibrate_mass = Column(
#         SmallInteger
#     )  # Perform mass calibration (0 for OFF, 1 for ON, 2 for ON and find optimal parameters).
#
#     modi01 = Column(Integer, ForeignKey("modifications.id"))
#     modi02 = Column(Integer, ForeignKey("modifications.id"))
#     modi03 = Column(Integer, ForeignKey("modifications.id"))
#     modi04 = Column(Integer, ForeignKey("modifications.id"))
#     modi05 = Column(Integer, ForeignKey("modifications.id"))
#     modi06 = Column(Integer, ForeignKey("modifications.id"))
#     modi07 = Column(Integer, ForeignKey("modifications.id"))


# class Modification(Base):
class Modification(SQLModel, table=True):

    __tablename__ = "modifications"

    id: int = Field(primary_key=True)
    name: str
    mass_shift: float
    position_str: str
    ntotal: int

    # id = Column(Integer, primary_key=True)
    # name = Column(String)
    # mass_shift = Column(Float)
    # position_str = Column(String)
    # ntotal = Column(SmallInteger)


def create_data():
    with Session(engine) as session:
        # inst = Instrument(name="amfusion", qc_recno=99999, id=1)
        # session.add(inst)

        #  inst = Instrument(name="LumosETD", qc_recno=99995)
        #  session.add(inst)

        import ipdb

        ipdb.set_trace()
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


if __name__ == "__main__":
    create_db_and_tables()
    create_data()
