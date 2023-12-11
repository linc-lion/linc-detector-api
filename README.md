# Linc Detection BentoML Service

## Overview

This BentoML service utilizes the Linc Detection model for image annotation. It exposes an API endpoint for annotating images with bounding box coordinates.

## Prerequisites

- Python 3.6 or later
- packages specified in `requirements.txt`

Install the required dependencies:

```bash
pip install bentoml Pillow torchvision boto3 torch
```

## Getting Started

1. Clone the repository
```bash
git clone <repository-url>
```
2. Navigate to the src directory:
```bash
cd <repository-path>/src
```
3. Build the BentoML service:
```bash
bentoml build
```
4. Run the BentoML service:
```bash
bentoml serve linc_detection:svc --reload
```
This will start the BentoML service, and you can access the API at http://0.0.0.0:3000/v1/annotate. Make sure to check the documentation or code for any additional configuration options or API endpoints.
curl request:
```
curl --location --request POST 'http://127.0.0.1:5000/v1/annotate?vert_size=250' \
--header 'Authorization: Bearer 1e620008-745c-4e84-be74-81042ab71b1e' \
--form 'file=@"{FILE_PATH_TO_IMAGE}"'
```

