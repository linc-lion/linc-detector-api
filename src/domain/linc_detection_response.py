from typing import List, Dict, Optional
from pydantic import BaseModel

class LincDetectionResponse(BaseModel):
    bounding_box_coords: Optional[Dict[str, List[float]]] = None
    error_message: Optional[str] = None
    message: Optional[str] = None
