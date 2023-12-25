# Lion Detection Service

## Overview

This BentoML service utilizes the lion detection model for image annotation. It exposes an API endpoint for annotating images with bounding box coordinates.

## Installation
1) Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation).
2) Execute the following on a terminal:
   - ```
      > conda create --name <insert_name> python=3.9 --y
      > conda activate <insert_name>  # instruction for Mac. See conda cheatsheet below for other OS.
      > pip install -r requirements.txt
     ``` 
 
3) Conda [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).
4) Recommended IDE is [Pycharm](https://www.jetbrains.com/pycharm/download/).
   * Right mouse click on `src` to mark directory as source root.

## Run the service
1) Before running the service, we need to download the artifacts from s3:
   - ```
     > ./fetch-artifacts.sh
     ```
2) From command line:
   - ```commandline
     export BEARER_TOKEN=insert_bearer_token_of_your_choice
     ```
   - ```commandline
     cd src
     bentoml serve linc_detection:svc --reload
     ```
   - Alternatively, you can start up a production server:
     ```commandline
     bentoml serve linc_detection:svc --production
     ```

## Usage
1) Swagger url: http://localhost:3000

2) Example curl request:
    ```bash
    curl --location 'http://localhost:3000/v1/annotate' \
    --header 'Authorization: Bearer insert_bearer_token_of_your_choice' \
    --form 'file=@"path/to/lion_image.jpg"'
   ```

## Resources
* [BentoML](https://docs.bentoml.org/en/latest/index.html)
* [Pytest](https://docs.pytest.org/en/stable/contents.html)