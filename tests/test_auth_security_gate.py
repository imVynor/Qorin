"""
QORIN Comprehensive Security Gate Authentication Test Suite
Tests all 10 security verification gates under strict non-disclosure of credentials.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Enforce CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi.testclient import TestClient
from backend.app import (
    app,
    COMPUTE_ACCESS_PASSWORD,
    COMPUTE_TOKEN_SECRET,
    generate_session_token,
    verify_session_token,
    check_compute_auth,
    get_client_ip,
    FAILED_AUTH_ATTEMPTS,
    REVOKED_SESSION_JTI,
)

client = TestClient(app)


def test_1_correct_new_password():
    print("[Gate 1/10] Testing Correct New Evaluator Credential...")
    # Read active secret without printing it
    res = client.post("/api/auth/verify-compute", json={"password": COMPUTE_ACCESS_PASSWORD})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["success"] is True
    assert "token" in data and data["token"].startswith("qorin_tok_")
    assert "legacy_token" not in data, "Legacy token must not be issued"
    print("           PASS: Valid session token issued with zero credential disclosure.")
    return data["token"]


def test_2_incorrect_password():
    print("[Gate 2/10] Testing Incorrect Password Rejection...")
    res = client.post("/api/auth/verify-compute", json={"password": "TotallyBogusPassword123!"})
    assert res.status_code == 401, f"Expected 401, got {res.status_code}"
    data = res.json()
    assert "Invalid evaluator authorization key" in data["detail"]
    assert "UDIT" not in data["detail"].upper(), "Detail disclosed old name"
    print("           PASS: Incorrect password rejected with generic 401 error.")


def test_3_old_password_rejection():
    print("[Gate 3/10] Testing Old Disclosed Password & Legacy String Rejection...")
    old_disclosed_passwords = [
        "UDIT BHATIA",
        "udit bhatia",
        "UDIT_BHATIA_AUTHORIZED",
        "PROF_UDIT_BHATIA",
    ]
    for old_pwd in old_disclosed_passwords:
        # Endpoint login test
        res = client.post("/api/auth/verify-compute", json={"password": old_pwd})
        assert res.status_code == 401, f"Old password '{old_pwd}' was accepted at login!"

        # Direct token header test
        res_header = client.post("/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, headers={"X-Compute-Token": old_pwd})
        assert res_header.status_code == 401, f"Old string '{old_pwd}' was accepted as token header!"
        assert check_compute_auth(old_pwd) is False, f"check_compute_auth accepted old string: {old_pwd}"

    print("           PASS: All old disclosed credentials and legacy strings rejected across all auth paths.")


def test_4_invalid_signature():
    print("[Gate 4/10] Testing Invalid Cryptographic Signature...")
    valid_token = generate_session_token("evaluator", expiry_hours=24)
    # Flip the last character of the signature
    prefix, sig = valid_token.rsplit(".", 1)
    tampered_sig = ("0" if sig[-1] != "0" else "1") + sig[:-1]
    tampered_token = f"{prefix}.{tampered_sig}"

    assert verify_session_token(tampered_token) is False
    assert check_compute_auth(tampered_token) is False

    res = client.post("/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, headers={"X-Compute-Token": tampered_token})
    assert res.status_code == 401
    print("           PASS: Invalid signature rejected cryptographically.")


def test_5_expired_token():
    print("[Gate 5/10] Testing Expired Session Token...")
    # Generate token expired 1 hour ago
    expired_token = generate_session_token("evaluator", expiry_hours=-1)
    assert verify_session_token(expired_token) is False
    assert check_compute_auth(expired_token) is False

    res = client.post("/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, headers={"X-Compute-Token": expired_token})
    assert res.status_code == 401
    print("           PASS: Expired token rejected automatically.")


def test_6_logged_out_session():
    print("[Gate 6/10] Testing Logged-Out / Revoked Session...")
    token = generate_session_token("evaluator", expiry_hours=24)
    assert verify_session_token(token) is True

    # Call logout endpoint
    res_logout = client.post("/api/auth/logout", headers={"X-Compute-Token": token})
    assert res_logout.status_code == 200
    assert res_logout.json()["success"] is True

    # Verify session is now revoked and rejected
    assert verify_session_token(token) is False
    assert check_compute_auth(token) is False

    res_post_logout = client.post("/api/models/select", json={"model_id": "qorin-100m-sft-v3"}, headers={"X-Compute-Token": token})
    assert res_post_logout.status_code == 401
    print("           PASS: Session invalidation / logout successfully revokes token.")


def test_7_session_tampering():
    print("[Gate 7/10] Testing Session Payload Tampering...")
    token = generate_session_token("evaluator", expiry_hours=24)
    body = token[len("qorin_tok_"):]
    payload_b64, sig = body.split(".")

    padding = 4 - (len(payload_b64) % 4)
    payload_b64_padded = payload_b64 + ("=" * padding) if padding < 4 else payload_b64
    payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded.encode("utf-8")).decode("utf-8"))

    # Attacker tries to escalate privileges or extend expiration
    payload["sub"] = "superuser_root"
    payload["exp"] = payload["exp"] + 1000000
    tampered_payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    tampered_token = f"qorin_tok_{tampered_payload_b64}.{sig}"

    assert verify_session_token(tampered_token) is False
    assert check_compute_auth(tampered_token) is False
    print("           PASS: Payload tampering detected; signature mismatch enforced.")


def test_8_brute_force_rate_limiting():
    print("[Gate 8/10] Testing Brute-Force Rate Limiting (5 Attempts / 60s)...")
    test_ip = "198.51.100.42"
    FAILED_AUTH_ATTEMPTS[test_ip] = []

    # Send 5 incorrect attempts
    for i in range(1, 6):
        res = client.post(
            "/api/auth/verify-compute",
            json={"password": f"wrong_guess_{i}"},
            headers={"X-Forwarded-For": test_ip},
        )
        assert res.status_code == 401, f"Attempt {i} expected 401, got {res.status_code}"

    # 6th attempt should be blocked by rate limiter with 429
    res_blocked = client.post(
        "/api/auth/verify-compute",
        json={"password": "wrong_guess_6"},
        headers={"X-Forwarded-For": test_ip},
    )
    assert res_blocked.status_code == 429, f"Expected 429 Too Many Requests, got {res_blocked.status_code}"
    assert "Too many failed compute authorization attempts" in res_blocked.json()["detail"]
    print("           PASS: 6th brute-force attempt triggered HTTP 429 rate limit.")


def test_9_independent_sessions():
    print("[Gate 9/10] Testing Session Independence...")
    tok_a = generate_session_token("evaluator_a", expiry_hours=24)
    tok_b = generate_session_token("evaluator_b", expiry_hours=24)
    assert tok_a != tok_b, "Two independently generated tokens must be distinct"

    # Revoke session A
    client.post("/api/auth/logout", headers={"X-Compute-Token": tok_a})
    assert verify_session_token(tok_a) is False, "Session A should be revoked"
    assert verify_session_token(tok_b) is True, "Session B must remain valid and independent"
    print("           PASS: Sessions are cryptographically isolated with independent jti values.")


def test_10_direct_unauthenticated_api_access():
    print("[Gate 10/10] Testing Direct Unauthenticated API Access Rejection...")
    endpoints = [
        ("POST", "/api/models/select", {"model_id": "qorin-100m-sft-v3"}),
        ("POST", "/api/models/unload", {}),
        ("POST", "/api/chat/stream", {"model_id": "qorin-100m-sft-v3", "messages": [{"role": "user", "content": "hi"}]}),
    ]
    for method, path, payload in endpoints:
        res = client.post(path, json=payload)
        assert res.status_code == 401, f"{path} expected 401, got {res.status_code}"
        assert "Evaluator compute access protected" in res.json()["detail"]

    print("           PASS: All protected inference endpoints reject unauthenticated access.")


def test_reverse_proxy_ip_sanitization():
    print("[Bonus] Testing Reverse-Proxy IP Header Sanitization...")
    class MockReq:
        def __init__(self, headers, host="127.0.0.1"):
            self.headers = headers
            class Client:
                host = "127.0.0.1"
            self.client = Client()

    # Normal public IP
    req1 = MockReq({"x-forwarded-for": "203.0.113.195"})
    assert get_client_ip(req1) == "203.0.113.195"

    # Multi-hop proxy chain: first IP is edge client
    req2 = MockReq({"x-forwarded-for": "203.0.113.195, 10.0.0.1, 172.16.0.2"})
    assert get_client_ip(req2) == "203.0.113.195"

    # Malicious injection attempt in X-Forwarded-For
    req3 = MockReq({"x-forwarded-for": "malicious; DROP TABLE; 10.0.0.1"})
    # Should reject the invalid IP and fall back to request.client.host
    assert get_client_ip(req3) == "127.0.0.1"
    print("           PASS: Reverse-proxy edge client resolution and injection mitigation verified.")


if __name__ == "__main__":
    print("=" * 70)
    print("QORIN AUTHENTICATION SECURITY GATE VERIFICATION")
    print("=" * 70)
    test_1_correct_new_password()
    test_2_incorrect_password()
    test_3_old_password_rejection()
    test_4_invalid_signature()
    test_5_expired_token()
    test_6_logged_out_session()
    test_7_session_tampering()
    test_8_brute_force_rate_limiting()
    test_9_independent_sessions()
    test_10_direct_unauthenticated_api_access()
    test_reverse_proxy_ip_sanitization()
    print("=" * 70)
    print("ALL 10 SECURITY GATE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)
