from flask import Flask, jsonify
from flask_cors import CORS
from route.pdf_scanner import pdf_scanner
from route.chat_route import chat_route
from route.history_route import history_route

from flasgger import Swagger


app = Flask(__name__)
swagger = Swagger(app)
CORS(app)
app.register_blueprint(pdf_scanner)

chat_route(app)
history_route(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
