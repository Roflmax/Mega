#!/bin/bash
# gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 
#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload