from pydantic import BaseModel


class PingResponse(BaseModel):
    alive: bool
