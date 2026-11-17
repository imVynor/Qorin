"""
Tests for QORIN WebSocket Chat Streaming Endpoint (/ws/chat)
"""

import json
import pytest
from fastapi.testclient import TestClient
from backend.app import app, generate_session_token, COMPUTE_TOKEN_VALID

client = TestClient(app)


def test_websocket_connection_ack():
    """Verify WebSocket connection succeeds and receives connection_ack."""
    with client.websocket_connect("/ws/chat") as websocket:
        ack = websocket.receive_json()
        assert ack["type"] == "connection_ack"
        assert ack["device"] == "cpu"
        assert "timestamp" in ack


def test_websocket_heartbeat():
    """Verify WebSocket responds to client ping/heartbeat."""
    with client.websocket_connect("/ws/chat") as websocket:
        _ack = websocket.receive_json()
        websocket.send_json({"type": "ping", "request_id": "req_ping_001"})
        res = websocket.receive_json()
        assert res["type"] == "heartbeat"
        assert res["request_id"] == "req_ping_001"


def test_websocket_unauthorized_chat():
    """Verify chat request without compute token returns AUTH_REQUIRED error."""
    with client.websocket_connect("/ws/chat") as websocket:
        _ack = websocket.receive_json()
        websocket.send_json({
            "type": "chat",
            "request_id": "req_test_001",
            "message": "Hello QORIN",
            "model_id": "qorin-100m-sft-v3",
        })
        res = websocket.receive_json()
        assert res["type"] == "error"
        assert res["error_code"] == "AUTH_REQUIRED"
        assert "Protected compute access required" in res["error"]


def test_websocket_authorized_chat_streaming():
    """Verify authorized chat request streams tokens and completes."""
    session_tok = generate_session_token("prof_udit_bhatia", expiry_hours=1)

    with client.websocket_connect("/ws/chat") as websocket:
        _ack = websocket.receive_json()
        websocket.send_json({
            "type": "chat",
            "request_id": "req_stream_test",
            "conversation_id": "conv_test_01",
            "message": "Hi",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": session_tok,
            "max_new_tokens": 5,
            "temperature": 0.0,
        })

        events = []
        tokens = []
        while True:
            evt = websocket.receive_json()
            events.append(evt)
            if evt["type"] == "token":
                tokens.append(evt.get("text", ""))
            elif evt["type"] == "generation_completed":
                break
            elif evt["type"] == "error":
                pytest.fail(f"Received unexpected error: {evt}")

        event_types = [e["type"] for e in events]
        assert "request_accepted" in event_types
        assert "generation_started" in event_types
        assert "generation_completed" in event_types
        # Verify tokens were streamed
        assert len(tokens) > 0


def test_websocket_cancellation():
    """Verify request cancellation protocol."""
    with client.websocket_connect("/ws/chat") as websocket:
        _ack = websocket.receive_json()
        websocket.send_json({
            "type": "cancel",
            "request_id": "req_cancel_001",
            "conversation_id": "conv_cancel_001",
        })
        res = websocket.receive_json()
        assert res["type"] == "generation_cancelled"
        assert res["request_id"] == "req_cancel_001"

