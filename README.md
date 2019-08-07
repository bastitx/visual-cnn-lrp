# Visual CNN LRP
Add layer-wise relevance propagation method to http://scs.ryerson.ca/~aharley/vis/

## Installation
For the backend install Python 3 and Flask and then from the backend directory start the Flask server by entering `FLASK_APP=backend.py flask run`. Currently the frontend expects the backend to be on port 5000.
For the frontend you should be able to use any web server, I chose nginx. Point the webserver at the webui folder and start it. 

## Run Docker Container
In order to run the visualization in docker containers (one for frontend and one for backend) you can execute the docker_build.sh and docker_start.sh files. 