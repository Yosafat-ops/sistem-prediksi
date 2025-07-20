from app import app
from flask import request

def handler(req):
    with app.request_context(req.environ):
        return app.full_dispatch_request()