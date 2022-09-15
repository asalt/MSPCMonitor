from sqlite3 import Date
from datetime import datetime
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

from .database import engine


def create_db_and_tables(engine):
    SQLModel.metadata.create_all(engine)


# class Instrument(SQLModel, table=True):
#     # class Meta:
#     #     load_instance = True
#
#     # id = Column(Integer, primary_key=True)
#     # name = Column(String)
#     # qc_recno = Column(BigInteger)
#     id: Optional[int] = Field(primary_key=True)
#     name: str = Field(default=None)
#     qc_recno: int = Field(default=None)
#
#     # rawfiles = relationship("RawFile", back_populates="instrument")
#     # rawfiles = relationship("RawFile")
#     rawfiles: List["RawFile"] = Relationship(back_populates="instrument")
#     experimentruns: List["ExperimentRun"] = Relationship(back_populates="instrument")
#
#     # class Config:
#     #     immutable = True


# class RawFile(Base):
class RawFile(SQLModel, table=True):
    __tablename__ = "rawfile"

    class Meta:
        load_instance = True

    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "RawFile:": "rawfile",
                "Instrument:": "instrument",
                "ExperimentMsOrder:": "",
                "MS1Analyzer:": "MS1Analyzer",
                "MS2Analyzer:": "MS2Analyzer",
                "MS3Analyzer:": "MS3Analyzer",
                "TotalAnalysisTime(min):": "TotalAnalysisTime",
                "TotalScans:": "TotalScans",
                "NumMs1Scans:": "NumMs1Scans",
                "NumMs2Scans:": "NumMs2Scans",
                "NumMs3Scans:": "NumMs3Scans",
                "MeanMs2TriggerRate(/Ms1Scan):": "MeanMs2TriggerRate",
                "Ms1ScanRate(/sec):": "Ms1ScanRate",
                "Ms2ScanRate(/sec):": "Ms2ScanRate",
                "MeanDutyCycle(s):": "MeanDutyCycle",
                "MedianMs1FillTime(ms):": "MedianMs1FillTime",
                "MedianMs2FillTime(ms):": "MedianMs2FillTime",
                "MedianMs3FillTime(ms):": "MedianMs3FillTime",
                "Ms2MedianSummedIntensity:": "Ms2MedianSummedIntensity",
                "MedianMS1IsolationInterference:": "MedianMS1IsolationInterference",
                "MedianPeakWidthAt10%H(s):": "MedianPeakWidthAt10",
                "MedianPeakWidthAt50%H(s):": "MedianPeakWidthAt50",
                "MedianAsymmetryFactor:": "MedianAsymmetryFactor",
                "PeakCapacity:": "PeakCapacity",
                "NumEsiInstabilityFlags:": "NumEsiInstabilityFlags",
            }
        }

    id: Optional[int] = Field(primary_key=True)
    name: str
    # TODO datetime
    ctime: datetime
    mtime: datetime
    size: int
    # instrument: str = Field(default=None)
    # instrument_id: Optional[int] = Field(default=None, foreign_key="instrument.id")
    instrument: str = Field(default=None)

    # flags
    # we can use the flags from the linked exprun

    # instrument_id: Optional[int] = Field(default=None, foreign_key="instrument.id")
    # instrument: Instrument = Relationship(back_populates="rawfiles")

    experimentrun_id: Optional[int] = Field(
        default=None, foreign_key="experimentrun.id"
    )
    experimentrun: "ExperimentRun" = Relationship(back_populates="rawfiles")
    ## =========================================
    def __eq__(self, other):
        if other is None:
            return
        return (
            self.name
            == other.name
            # and self.ctime == other.ctime
            # and self.mtime == other.mtime
        )

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


class Project(SQLModel, table=True):
    __tablename__ = "project"

    class Config:
        schema_extra = dict()  # TODO fill

    id: Optional[int] = Field(primary_key=True)


# class Experiment(Base):
class Experiment(SQLModel, table=True):
    __tablename__ = "experiment"

    class Meta:
        load_instance = True
        include_relationships = True
        # model = SQLModel

    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "recno": "exp_EXPRecNo",
                # "label": "exp_EXPLabelFLAG",
                "geno": "exp_Extract_Genotype",
                "label": "exp_EXPLabelType",
                # "": "exp_ExpType",
                "extractno": "exp_Extract_No",
                # "": "exp_Extract_Adjustments",
                # "": "exp_Extract_Amount",
                "cell_tissue": "exp_Extract_CellTissue",
                # "": "exp_Extract_Fractions",
                # "": "exp_Digest_Enzyme",
                # "": "exp_Digest_Experimenter",
                # "": "exp_Digest_Type",
                "date": "exp_Exp_Date",
                # "": "exp_Separation_1",
                # "": "exp_Separation_1Detail",
                # "": "exp_Separation_2,
                # "": "exp_Separation_2Detail,
                # "": "exp_Separation_3,
                # "": "exp_Separation_3Detail,
            }
        }

    id: Optional[int] = Field(primary_key=True)
    recno: int
    label: str = Field(default=None)  # TMT, LF,
    extractno: int = Field(default=None)
    date: str = Field(default=None)
    digest_enzyme: str = Field(default=None)
    cell_tissue: str = Field(default=None)
    geno: str = Field(default=None)
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
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                # "": "dev_exprun_u2g_check",
                "runno": "exprun_AddedBy",
                "date": "exprun_CreationTS",
                # "": "exprun_E2GFile0_expFileName",
                "recno": "exprun_EXPRecNo",
                "runno": "exprun_EXPRunNo",
                "searchno": "exprun_EXPSearchNo",
                # "": "exprun_Fraction_10090",
                # "": "exprun_Fraction_9031",
                # "": "exprun_Fraction_9606",
                "is_grouped": "exprun_Grouper_EndFLAG",
                # "": "exprun_Grouper_FailedFLAG",
                # "": "exprun_Grouper_Filter_modiMax",
                # "": "exprun_Grouper_FilterStamp",
                # "": "exprun_Grouper_RefDatabase",
                # "": "exprun_Grouper_StartFLAG",
                # "": "exprun_Grouper_Version",
                "is_imported": "exprun_Import_EndFLAG",
                # "": "exprun_Import_FixFLAG",
                # "": "exprun_Import_StartFLAG",
                # "": "exprun_ImportTS",
                # "": "exprun_InputFileName",
                # "": "exprun_LabelType",
                # "": "exprun_ModificationTS",
                # "": "exprun_MS_Experimenter",
                "instrument": "exprun_MS_Instrument",
                # "": "exprun_MSFFile_expFileName",
                # "": "exprun_nGeneCount",
                # "": "exprun_nGPGroupCount",
                # "": "exprun_niBAQ_0_Total",
                # "": "exprun_niBAQ_1_Total",
                # "": "exprun_nMSFilesCount",
                # "": "exprun_nMSFilesCount Total",
                # "": "exprun_nMShrs",
                # "": "exprun_nMShrs Total",
                # "": "exprun_nPSMCount",
                # "": "exprun_nTechRepeats",
                # "": "exprun_PSMCount_unmatched",
                # "": "exprun_PSMsFile_expFileName",
                # "": "exprun_Purpose",
                # "": "exprun_Search_Comments",
                # "": "exprun_Search_EnzymeSetting",
                # "": "exprun_Search_RefDatabase",
                # "": "exprun_TaxonID",
                # "": "exprun_Vali",
                # "": "iSPEC_Experiments::exp_Extract_CellTissue",
            }
        }

    __tablename__ = "experimentrun"
    id: Optional[int] = Field(primary_key=True)
    runno: int = Field(default=None)
    searchno: int = Field(default=6)
    tic: bool = Field(default=False)
    is_searched: bool = Field(default=False)
    is_grouped: bool = Field(default=False)
    is_imported: bool = Field(default=False)
    refdb: Optional[str] = Field(default=None)
    # is_validated: bool = False
    date: str = Field(default=None)
    # instrument: str = Field(default=None)

    # instrument_id: Optional[int] = Field(default=None, foreign_key="instrument.id")
    # instrument: Instrument = Relationship(back_populates="experimentruns")
    instrument: Optional[str] = Field(default=None)
    # taxon: str
    # refdb: str

    experiment_id: Optional[int] = Field(default=None, foreign_key="experiment.id")
    experiment: Experiment = Relationship(back_populates="experimentruns")
    rawfiles: Optional[List["RawFile"]] = Relationship(back_populates="experimentrun")
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="experimentrun")
    e2gquals: Optional[List["E2GQual"]] = Relationship(back_populates="experimentrun")


class Gene(SQLModel, table=True):
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "geneid": "GeneID",
                "symbol": "GeneSymbol",
                "funcats": "FunCats",
                "taxonid": "TaxonID",
                "synonyms": "Synonyms",
                "description": "GeneDescription",
                "chromosome": "GeneChromosome",
                "gi": "gi",
                "sequence": "sequence",
                "AAlength": "AAlength",
                "MW": "MW",
                "maxMW": "maxMW",
            }
        }

    class Meta:
        load_instance = True

    # id: Optional[int] = Field(primary_key=True)
    geneid: Optional[int] = Field(primary_key=True)
    # geneid: str = Field()
    taxonid: Optional[int] = Field()
    symbol: str = Field()
    funcats: Optional[str] = Field()
    synonyms: Optional[str] = Field()
    description: Optional[str] = Field()
    chromosome: Optional[str] = Field()
    gi: Optional[str] = Field()
    sequence: Optional[str] = Field()
    AAlength: Optional[str] = Field()
    MW: Optional[str] = Field()
    maxMW: Optional[str] = Field()
    # =======================================================================
    # many to many relationship, do it later or not at all
    # experiments: List[Experiment] = Relationship(back_populates="genes")
    # =======================================================================
    # gene_capacity: int = Field(default=0)
    e2gquals: Optional[List["E2GQual"]] = Relationship(back_populates="geneid")
    e2gquants: Optional[List["E2GQuant"]] = Relationship(back_populates="geneid")


class E2GQual(SQLModel, table=True):
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "peptidecount": "PeptideCount",
                "peptidecount_u2g": "PeptideCount_u2g",
                "peptidecount_s_u2g": "PeptideCount_S_u2g",
                "peptidecount_s": "PeptideCount_S",
                "peptideprint": "PeptidePrint",
                "coverage": "Coverage",
                "gpgroup": "GPGroup",
                "gpgroups_all": "GPGroups_All",
                "coverage": "Converage",
                "coverage_u2g": "Converage_u2g",
                "psms_u2g": "",
                "proteingi_gpdgroups": "",
                "proteinref_gidgroups": "",
            }
        }

    class Meta:
        load_instance = True

    id: Optional[int] = Field(primary_key=True)
    #
    experiment: Optional[Experiment] = Relationship(back_populates="e2gquals")
    experiment_id: Optional[int] = Field(default=None, foreign_key="experiment.id")
    experimentrun: Optional[ExperimentRun] = Relationship(back_populates="e2gquals")
    experimentrun_id: Optional[int] = Field(
        default=None, foreign_key="experimentrun.id"
    )
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
    # peptidecount_s: int = Field()
    # proteinrefs: int = Field()

    geneid: Optional[Gene] = Relationship(back_populates="e2gquals")
    geneid_id: Optional[int] = Field(foreign_key="gene.geneid")
    #
    coverage: Optional[float] = Field()
    coverage_u2g: Optional[float] = Field()
    idset: Optional[int] = Field()
    SRA: Optional[str] = Field()
    #
    psms: Optional[int] = Field()
    psms_s: Optional[int] = Field()
    psms_s_u2g: Optional[int] = Field()

    peptidecount: Optional[int] = Field()
    peptidecount_u2g: Optional[int] = Field()
    peptidecount_s: Optional[int] = Field()
    # GeneSymbol: str = Field()
    peptidecount_s_u2g: Optional[int] = Field()
    gpgroup: Optional[int] = Field()
    peptideprint: Optional[str] = Field()
    psms_u2g: Optional[int] = Field()
    # GeneCapacity: str = Field()
    proteingi_gidgroups: Optional[str] = Field()
    proteinref_gidgroups: Optional[str] = Field()


class PSMQuant(SQLModel, table=True):
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "label": "LabelFLAG",
                "reporter_ion_intensity": "ReporterIntensity",
                "precursor_ion_auc": "PrecursorArea",
            }
        }

    id: Optional[int] = Field(primary_key=True)
    label: str = Field(default=None)
    reporter_ion_intensity: Optional[float] = Field()
    precursor_ion_auc: Optional[float] = Field()


class PSMQual(SQLModel, table=True):
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "sequence": "Sequence",
                "rt": "RTmin",
                "charge": "Charge",
                "num_matched_ions": "num_matched_ions",
                "precursor_neutral_mass": "precursor_neutral_mass",
                "scan": "FirstScan",
            }
        }

    id: Optional[int] = Field(primary_key=True)
    sequence: str = Field(default=None)
    rt: float = Field()
    charge: int = Field()
    num_matched_ions: int = Field()
    # num_matched_ions: int: Field()
    scan: int = Field()
    precursor_neutral_mass: float = Field()
    label: str = Field(default=None)
    quality: Optional[int] = Field()
    quantity: Optional[int] = Field()


class E2GQuant(SQLModel, table=True):
    class Config:
        schema_extra = {
            "ispec_column_mapping": {
                "label": "LabelFLAG",
                "areasum_u2g_0": "AreaSum_u2g_0",
                "areasum_u2g_all": "AreaSum_u2g_all",
                "areasum_max": "AreaSum_max",
                "areasum_dstradj": "AreaSum_dstrAdj",
                "ibaq_dstradj": "iBAQ_dstrAdj",
            }
        }

    class Meta:
        load_instance = True

    id: Optional[int] = Field(primary_key=True)
    label: str = Field(default=None)
    #
    experiment: Experiment = Relationship(back_populates="e2gquants")
    experiment_id: int = Field(default=None, foreign_key="experiment.id")
    experimentrun: ExperimentRun = Relationship(back_populates="e2gquants")
    experimentrun_id: int = Field(default=None, foreign_key="experimentrun.id")
    #
    geneid: Optional[Gene] = Relationship(back_populates="e2gquants")
    geneid_id: Optional[int] = Field(foreign_key="gene.geneid")
    #
    e2gqual_id: Optional[int] = Field(default=None, foreign_key="e2gqual.id")
    e2gqual: Optional["E2GQual"] = Relationship(back_populates="e2gquants")

    #
    # label: Optional[str] = Relationship(back_populates="label")
    # label_id: Optional[int] = Field(default=None, foreign_key="label.id")
    #
    areasum_u2g_0: Optional[float] = Field(default=0)
    areasum_u2g_all: Optional[float] = Field(default=0)
    areasum_max: Optional[float] = Field(default=0)
    areasum_dstradj: Optional[float] = Field(default=0)
    ibaq_dstradj: Optional[float] = Field(default=0)


# class Label(SQLModel, table=True):
#     class Meta:
#         load_instance = True
#
#     id: Optional[int] = Field(primary_key=True)
#     name: Optional[str] = Field(default="none")
#     primary_mass_shift: float = Field(default=0.0)


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


def _example_create_data():
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


# create_db_and_tables()  # ?? bad?? yes

if __name__ == "__main__":
    pass
    # create_db_and_tables()
    # _example_create_data()
