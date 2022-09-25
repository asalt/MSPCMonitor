# test_fixtures.py
import pytest


@pytest.fixture
def sqlengine():

    from sqlmodel import create_engine, SQLModel

    SQLALCHEMY_DATABASE_URL = "sqlite://"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": True}, echo=False
    )
    SQLModel.metadata.create_all(engine)
    yield engine
