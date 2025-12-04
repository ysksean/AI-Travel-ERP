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

@app.route('/products')
def product_list():
    return render_template('product_list.html', active_page='products')

@app.route('/products/new')
def product_create():
    return render_template('product_create.html', active_page='products')

@app.route('/products/<id>')
def product_detail(id):
    return render_template('product_detail.html', active_page='products')

@app.route('/reservations')
def reservation_list():
    return render_template('reservation_list.html', active_page='reservations')

@app.route('/reservations/<id>')
def reservation_detail(id):
    return render_template('reservation_detail.html', active_page='reservations')

@app.route('/quotations')
def quotation_list():
    return render_template('quotation_list.html', active_page='quotations')

@app.route('/quotations/new')
def quotation_create():
    return render_template('quotation_create.html', active_page='quotations')

@app.route('/quotations/<id>')
def quotation_detail(id):
    return render_template('quotation_detail.html', active_page='quotations')

@app.route('/payments')
def payment_page():
    return render_template('payment.html', active_page='payments')

@app.route('/finance')
def finance_page():
    return render_template('finance.html', active_page='finance')

@app.route('/flights')
def flight_list():
    return render_template('flight_list.html', active_page='flights')

@app.route('/hotels')
def hotel_list():
    return render_template('hotel_list.html', active_page='hotels')

@app.route('/attractions')
def attraction_list():
    return render_template('attraction_list.html', active_page='attractions')

@app.route('/partners')
def partner_list():
    return render_template('partner_list.html', active_page='partners')

@app.route('/partners/new')
def partner_create():
    return render_template('partner_create.html', active_page='partners')

@app.route('/customers')
def customer_list():
    return render_template('customer_list.html', active_page='customers')

@app.route('/settings')
def settings_page():
    return render_template('settings.html', active_page='settings')

# API Routes (Legacy/AI Features)
@app.route('/api/product/analyze', methods=['POST'])
def analyze_product_text():
    # ... implementation ...
    return jsonify({}) # Placeholder if needed, or import from routes

# Keep original routes for backward compatibility if needed, but mapped to new templates if possible
# or just keep them as API endpoints if they were used for that.
# The original code imported blueprints. I should keep using blueprints if I want to keep the logic clean,
# but for now I am replacing the route definitions.
# Actually, the original code registered blueprints. I should probably keep that structure if possible,
# but the user wants "design exactly" which implies the frontend navigation works.
# The blueprints were for /api/product etc.
# I will keep the blueprint registration and just add the page routes.

if __name__ == '__main__':
    # Ensure models directory exists to avoid startup errors if empty
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created missing model directory: {model_dir}")

    app.run(host='0.0.0.0', port=7878, debug=True)
