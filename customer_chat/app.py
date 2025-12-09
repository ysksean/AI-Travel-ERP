from flask import Flask, render_template, jsonify
import os
from routes.main_route import bp as main_bp
app = Flask(__name__)

# 라우트 등록
app.register_blueprint(main_bp)

if __name__ == '__main__':
    print("🚀 Travel AI Server (Member B Ver.) Started!")
    print("👉 Test URL: http://127.0.0.1:5000/chat")
    app.run(host='0.0.0.0', port=7879, debug=True)
