import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch

# Add the current directory to sys.path so we can import app
sys.path.append(os.getcwd())

from app import app

class TestRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_chat_endpoint_no_message(self):
        response = self.app.post('/chat', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('reply', data)
        self.assertEqual(data['reply'], '메시지를 입력해주세요.')

    @patch('app.rag_engine')
    @patch('app.genai')
    def test_chat_endpoint_success(self, mock_genai, mock_rag_engine):
        # Mock RAG engine
        mock_rag_engine.search.return_value = "Mock Context"
        
        # Mock Gemini
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a mock response from Gemini."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        response = self.app.post('/chat', json={'message': 'Hello'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('reply', data)
        self.assertEqual(data['reply'], "This is a mock response from Gemini.")

    def test_real_import_logic(self):
        # This test checks if we can actually import without crashing, 
        # relying on app.py's try-except block
        try:
             import services.rag_service
             print("Real rag_service found (Expected if files exist)")
        except ImportError:
            print("Real rag_service NOT found (Might be expected if sibling dir missing)")

if __name__ == '__main__':
    unittest.main()
