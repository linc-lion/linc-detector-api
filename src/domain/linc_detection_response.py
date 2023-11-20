from typing import List, Dict
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class BoundingBoxCoords(BaseModel):
    bounding_boxes: Dict[str, List[float]]


class LincDetectionResponse(BaseModel):
    bounding_box_coords: BoundingBoxCoords
