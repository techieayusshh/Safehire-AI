import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 2
worker_class = "gthread"
timeout = 180
graceful_timeout = 30
keepalive = 5
max_requests = 200
max_requests_jitter = 50
preload_app = True
