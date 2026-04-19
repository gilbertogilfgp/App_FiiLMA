# gunicorn.conf.py
workers = 1
threads = 4
worker_class = "gthread"
timeout = 120
