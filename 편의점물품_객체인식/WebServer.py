from flask import Flask, jsonify, request, render_template, make_response, send_file
from flask_cors import CORS
import Product_Detect

app = Flask(__name__, static_url_path="/static")
CORS(app)

@app.route("/")
def test1():                           
    print("들어옴")
    return render_template("product_detect.html")

if __name__ == "__main__":
    app.register_blueprint(Product_Detect.bp)              
    app.run(host="127.0.0.1", port="8123")