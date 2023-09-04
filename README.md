### Linc Detector Webapp

### Requirements
- Python v3.8+

### Install
To install, clone this repo and cd into the root of the project.

1) Create a virtual environment:

```conda create --name linc-detector-api python=3.8```

2) Activate the virtual environment. If you're using a Unix-based OS, run:

```conda activate linc-detector-api```

3) Install the needed python dependencies:

```pip install -r requirements.txt```

### Running Locally

1. Configure Environment Variables

    Add a .env file in the root directory of the project add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as show below will

    ```buildoutcfg
    AWS_ACCESS_KEY_ID = "KEY_ID"
    AWS_SECRET_ACCESS_KEY = "ACCESS_KEY"
    ```

2. Execute ```flask --app application run``` at the root of the project

You should be able to view the REST endpoint at 127.0.0.1:5000/v1/annotate

### Using POSTMAN

To make a request using POSTMAN:

```
curl --location --request POST 'http://127.0.0.1:5000/v1/annotate?vert_size=250' \
--header 'Authorization: Bearer 1e620008-745c-4e84-be74-81042ab71b1e' \
--form 'file=@"{FILE_PATH_TO_IMAGE}"'
```

Replace YOUR_API_KEY with your actual API key and provide the path to the image file in {FILE_PATH_TO_IMAGE}.

The vert_size is configured as a query parameter, and the default value is 500.

note: file path needs the picture name and file extension ie: `/Users/habibamohamed/Downloads/habiba/evaluation/Amboga/PJB_2359.JPG`

This endpoint allows you to perform object detection on an image by uploading it to the server. The API will return an image with bounding boxes drawn around detected objects and the coordinates of these bounding boxes.

### Endpoint: `/v1/annotate`

This endpoint allows you to perform object detection on an image by uploading it to the server. The API will return an image with bounding boxes drawn around detected objects and the coordinates of these bounding boxes.

- HTTP Method: POST
- Content Type: `multipart/form-data`

**Form Data Parameters:**
- `file`: The image file to be uploaded for object detection.

**Query Parameter:**
- `vert_size` (optional): The vertical size for drawing bounding boxes. Default value is 500.

### Output

The API will return a JSON object with the following fields:

- `input_image`: The path to the uploaded input image.
- `bounding_box_coords`: A list of dictionaries containing the coordinates of the bounding boxes.


### Running Tests

To run the integration tests for the API, follow these steps:

1. Ensure you have dependencies installed before running tests
2. Run the tests using the command `python test_app.py`.

The tests cover various scenarios, including successful image annotation and cases where exceptions are thrown due to missing or invalid data.

