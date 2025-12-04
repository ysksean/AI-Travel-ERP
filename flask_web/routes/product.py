from flask import Blueprint, jsonify, request
from services.ai_service import ai_service

bp = Blueprint('product', __name__, url_prefix='/api/product')

@bp.route('/analyze', methods=['POST'])
def analyze_product_text():
    data = request.json
    text = data.get('text', '')
    result = ai_service.extract_entities(text)
    return jsonify(result)
