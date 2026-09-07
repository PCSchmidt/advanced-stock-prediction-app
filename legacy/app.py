from flask import Flask, send_from_directory
from main import app as fastapi_app
from fastapi.middleware.wsgi import WSGIMiddleware

app = Flask(__name__)

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

app.mount("/api", WSGIMiddleware(fastapi_app))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))