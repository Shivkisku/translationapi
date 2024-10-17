# config/gunicorn_config.py

bind = '127.0.0.1:5000'
workers = 4
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'
