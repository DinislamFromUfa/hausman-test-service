from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI

from infrastructure.di import HausmanProvider
from presentation.hausman.router import router


app = FastAPI()

container = make_async_container(HausmanProvider())
setup_dishka(container, app)

app.include_router(router)