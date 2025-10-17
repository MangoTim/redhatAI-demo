from flask import request, jsonify
from modules import get_model, get_context_from_keywords, log_message, clean_reply, get_db_connection

from flasgger import swag_from
from api_docs.history_route_docs import history_doc

def history_route(app):
    @app.route("/history", methods=["GET"])
    @swag_from(history_doc)
    def get_history():
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, message FROM chat_history
            WHERE role IN ('user', 'assistant')
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([{"role": r, "message": m} for r, m in reversed(rows)])

