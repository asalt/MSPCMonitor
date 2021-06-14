from typing import List

from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse

from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

from fastapi import FastAPI

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    #return {"message": "MSPCMonitor"}
    content = """
<body>
<form action="/upload/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/upload/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.get("/instruments/{instrument_id}")
def get_instrument(instrument_id: str):
    instrument = {"instrument_id": instrument_id}
    return instrument

@app.get("/instruments/", response_model=List[schemas.Instrument])
def get_all_instruments(db: Session = Depends(get_db)):
    return crud.get_all_instruments(db=db)


@app.post("/instruments/", response_model=schemas.Instrument)
def create_instrument(
    instrument: schemas.InstrumentCreate, db: Session = Depends(get_db)
):
    db_instrument = crud.get_instrument_by_name(db, name=instrument.name)
    if db_instrument:
        raise HTTPException(status_code=400, detail="Instrument already exists")
    return crud.create_instrument(db=db, instrument=instrument)

@app.post("/upload/")
async def upload_raw(files: List[UploadFile] = File(...)):
    return {"filename": file.filename for file in files}

@app.post("/instrument/{instrument_id}/rawfiles", response_model=schemas.RawFile)
def create_rawfile(
    instrument_id: int, rawfile: schemas.RawFileCreate, db: Session = Depends(get_db)
):
    return crud.create_rawfile(db=db, rawfile=rawfile, instrument_id=instrument_id)



@app.get("/rawfiles/", response_model=List[schemas.RawFile])
def get_rawfile(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    rawfiles = crud.get_rawfiles(db, skip=skip, limit=limit)
    return rawfiles


#import json
#import asyncio
#import aiopg
#from starlette.endpoints import WebSocketEndpoint
#from fastapi import Depends, FastAPI, HTTPException
#from starlette.websockets import WebSocket
#from pydantic import BaseModel

#@app.websocket_route("/status")
#class WebSocketOrders(WebSocketEndpoint):
#
#    encoding = "json"
#
#    def __init__(self, scope, receive, send):
#        super().__init__(scope, receive, send)
#        self.connected: bool = False
#        self.loop = asyncio.get_event_loop()
#        self.websocket: WebSocket = {}
#
#    @asyncio.coroutine
#    async def listen(self, conn, channel):
#        async with conn.cursor() as cur:
#            await cur.execute("LISTEN {0}".format(channel))
#            while self.connected:
#                msg = await conn.notifies.get()
#                payload: dict = json.loads(msg.payload)
#                if payload.get("action") == "INSERT":
#                    insert_data: Order = payload.get("data")
#                    await self.websocket.send_json(
#                        {"message": "New order", "data": insert_data}
#                    )
#                elif payload.get("action") == "UPDATE":
#                    update_data: Order = payload.get("data")
#                    await self.websocket.send_json(
#                        {"message": "Order update", "data": update_data}
#                    )
#
#    @asyncio.coroutine
#    async def db_events(self, data: dict, channel: str):
#        async with aiopg.create_pool(dsn) as pool:
#            async with pool.acquire() as conn:
#                try:
#                    await asyncio.gather(
#                        self.listen(conn, channel), return_exceptions=False
#                    )
#                except:
#                    print("releasing connection")
#
#    async def on_receive(self, websocket: WebSocket, data: dict):
#        channel: str = data.get("channel")
#        asyncio.ensure_future(self.db_events(data, channel), loop=self.loop)
#
#    async def on_connect(self, websocket: WebSocket):
#        await websocket.accept()
#        self.connected = True
#        self.websocket = websocket
#        await self.websocket.send_json({"message": "Welcome"})
#
#    async def on_close(self, websocket):
#        self.connected = False
#        self.loop.close()
#        self.websocket.close()