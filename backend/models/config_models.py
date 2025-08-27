from pydantic import BaseModel, Field
from typing import Tuple
from datetime import datetime

class Config(BaseModel):
    ticker: str
    
    moneyness: Tuple[float, float] = (0.75, 1.25)
    