web: streamlit run code_final.py --server.port $PORT --server.enableCORS false
api: gunicorn api:app --bind 0.0.0.0:$API_PORT
