from typing import List, Dict
from pydantic import BaseModel


class LincDetectionResponse(BaseModel):
    bounding_box_coords: Dict[str, List[float]]

