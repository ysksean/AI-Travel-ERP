from flask import Blueprint, jsonify, request
from services.ai_service import ai_service

bp = Blueprint('ops', __name__, url_prefix='/api/ops')

@bp.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    result = ai_service.analyze_sentiment(text)
    return jsonify(result)
