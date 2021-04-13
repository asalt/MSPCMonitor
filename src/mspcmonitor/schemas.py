from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class RawFileBase(BaseModel):
    name: str
    ctime: Optional[float] = None
    mtime: Optional[float] = None
    processed: Optional[bool] = False
    psms: Optional[int] = None
    #processed: Optional[bool] = False

class RawFileCreate(RawFileBase):
    pass

class RawFile(RawFileBase):
    id: int
    instrument_id: int
    class Config:
        orm_mode = True

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