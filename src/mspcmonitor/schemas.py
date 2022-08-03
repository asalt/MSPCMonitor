from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

# overhaul this using unified sqlmodel library


class RawFileBase(BaseModel):
    name: str
    ctime: Optional[datetime] = None
    mtime: Optional[datetime] = None
    processed: Optional[bool] = False
    psms: Optional[int] = None
    # processed: Optional[bool] = False


class RawFileCreate(RawFileBase):
    pass


class RawFile(RawFileBase):
    id: int
    instrument_id: int

    class Config:
        orm_mode = True


# =======================================


class InstrumentBase(BaseModel):
    name: str
    qc_recno: Optional[int] = None


class InstrumentCreate(InstrumentBase):
    pass


class Instrument(InstrumentBase):
    id: int
    rawfiles: List[RawFile]

    class Config:
        orm_mode = True


# =======================================


class ExperimentBase(BaseModel):
    recno: int
    label: str


class ExperimentCreate(ExperimentBase):
    pass


class Experiment(ExperimentBase):
    id: int
    # rawfiles: List[RawFile]

    class Config:
        orm_mode = True


class ExperimentRunBase(BaseModel):
    runno: int
    searchno: str

    taxon: str
    refdb: str


class ExperimentRunCreate(ExperimentRunBase):
    pass


class ExperimentRun(ExperimentRunBase):
    id: int

    class Config:
        orm_mode = True
