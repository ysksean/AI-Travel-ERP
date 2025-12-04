from flask import Flask, render_template
from routes import product, reservation, ops, finance
import os

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(product.bp)
app.register_blueprint(reservation.bp)
app.register_blueprint(ops.bp)
app.register_blueprint(finance.bp)

@app.route('/')
def index():
    return render_template('index.html', active_page='dashboard')

if __name__ == '__main__':
    # Ensure models directory exists to avoid startup errors if empty
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created missing model directory: {model_dir}")

    app.run(host='0.0.0.0', port=7878, debug=True)
