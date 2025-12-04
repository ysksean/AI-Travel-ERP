from flask import Blueprint, jsonify, request
from services.ai_service import ai_service

bp = Blueprint('finance', __name__, url_prefix='/api/finance')

@bp.route('/forecast', methods=['POST'])
def forecast_price():
    data = request.json
    date_range = data.get('date_range', [])
    result = ai_service.forecast_price(date_range)
    return jsonify(result)
