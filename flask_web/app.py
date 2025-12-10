import sys
from flask import Flask, render_template, jsonify, request
from routes import product, reservation, ops, finance
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Add sibling directory to sys.path to access 'travel' modules
current_dir = os.path.dirname(os.path.abspath(__file__))
travel_dir = os.path.abspath(os.path.join(current_dir, '../travel'))
if travel_dir not in sys.path:
    sys.path.append(travel_dir)

# Try importing RAG engine
rag_engine = None
try:
    from services.rag_service import rag_engine
    print("✅ RAG Engine imported successfully.")
    
    # Initialize Gemini
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        print("⚠️  Warning: GOOGLE_API_KEY not found in .env")
        
except ImportError as e:
    print(f"⚠️  Warning: Could not import rag_service: {e}")
    print("   Chat functionality will be limited to mock responses.")

app = Flask(__name__)

# 데이터베이스 초기화 (서버 시작 시 테이블 생성)
try:
    from services.models import init_db
    init_db()
except Exception as e:
    print(f"⚠️  Warning: Database initialization failed: {e}")
    print("   Make sure MySQL is running and travel_erp database exists.")

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

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages using RAG + Gemini with DB Persistence.
    """
    from services.models import ChatLog
    from services.db_connect import SessionLocal
    
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id')
    user_type = data.get('user_type', 'guest')
    user_name = data.get('user_name', 'Guest')
    
    if not user_message:
        return jsonify({'reply': '메시지를 입력해주세요.'}), 400

    db = SessionLocal()
    try:
        # 1. Save User Message
        user_log = ChatLog(
            session_id=session_id,
            user_type=user_type,
            user_name=user_name,
            sender='user',
            message=user_message
        )
        db.add(user_log)
        db.commit()

        # 2. RAG Search (Get Context)
        context = ""
        if rag_engine:
            try:
                # Assuming rag_engine.search returns a list of strings or similar
                context_results = rag_engine.search(user_message)
                if context_results:
                    context = "\n".join(context_results)
            except Exception as e:
                print(f"RAG Search Error: {e}")
                # Continue without context if RAG fails

        # 3. Generate Response with Gemini
        reply_text = ""
        if not os.getenv('GOOGLE_API_KEY'):
             reply_text = '[Mock] API Key missing. Simulating response: ' + user_message[::-1]
        else:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            You are a helpful travel agency assistant. Use the following context to answer the customer's question.
            If the answer is not in the context, use your general knowledge but mention that you are not sure.
            
            Context:
            {context}
            
            Question:
            {user_message}
            
            Answer (in Korean):
            """
            response = model.generate_content(prompt)
            reply_text = response.text

        # 4. Save Bot Response
        bot_log = ChatLog(
            session_id=session_id,
            user_type=user_type,
            user_name=user_name,
            sender='bot',
            message=reply_text
        )
        db.add(bot_log)
        db.commit()
        
        return jsonify({'reply': reply_text})

    except Exception as e:
        print(f"Chat Error: {e}")
        db.rollback()
        return jsonify({'reply': '죄송합니다. 오류가 발생했습니다.'}), 500
    finally:
        db.close()

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """
    Fetch chat sessions grouped by session_id, ordered by latest message.
    """
    from services.models import ChatLog
    from services.db_connect import SessionLocal
    from sqlalchemy import func, desc
    
    db = SessionLocal()
    try:
        # Subquery to find the latest message time for each session
        subquery = db.query(
            ChatLog.session_id,
            func.max(ChatLog.created_at).label('max_created_at')
        ).group_by(ChatLog.session_id).subquery()
        
        # Join with main table to get details of the latest message
        latest_msgs = db.query(ChatLog).join(
            subquery,
            (ChatLog.session_id == subquery.c.session_id) & 
            (ChatLog.created_at == subquery.c.max_created_at)
        ).order_by(desc(subquery.c.max_created_at)).limit(20).all()
        
        sessions = []
        for msg in latest_msgs:
            sessions.append({
                'session_id': msg.session_id,
                'user_name': msg.user_name,
                'user_type': msg.user_type,
                'last_message': msg.message,
                'updated_at': msg.created_at.strftime('%Y-%m-%d %H:%M')
            })
            
        return jsonify(sessions)
    except Exception as e:
        print(f"Session Fetch Error: {e}")
        return jsonify([])
    finally:
        db.close()

if __name__ == '__main__':
    # Ensure models directory exists to avoid startup errors if empty
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created missing model directory: {model_dir}")

    app.run(host='0.0.0.0', port=7878, debug=True)
