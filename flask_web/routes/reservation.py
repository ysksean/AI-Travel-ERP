from flask import Blueprint, jsonify, request
from services.ai_service import ai_service

bp = Blueprint('reservation', __name__, url_prefix='/api/reservation')

@bp.route('/summarize', methods=['POST'])
def summarize_request():
    data = request.json
    text = data.get('text', '')
    result = ai_service.summarize_request(text)
    return jsonify(result)
