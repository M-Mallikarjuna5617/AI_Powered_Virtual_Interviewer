"""
Unit tests for GD and HR interview API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestGDEndpoints:
    """Test Group Discussion endpoints"""

    def test_gd_start(self):
        """Test starting a GD session"""
        payload = {
            "topic": "Should AI replace human jobs?",
            "participants": ["Alice", "Bob", "Charlie"],
            "time_limit": 90
        }

        response = client.post("/api/gd/start", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert data["topic"] == payload["topic"]
        assert data["participants"] == payload["participants"]
        assert data["current_turn"] == 0
        assert data["time_limit"] == payload["time_limit"]

    def test_gd_turn(self):
        """Test processing a GD turn"""
        # First start a session
        start_payload = {
            "topic": "Remote work benefits",
            "participants": ["Alice", "Bob"]
        }
        start_response = client.post("/api/gd/start", json=start_payload)
        session_id = start_response.json()["session_id"]

        # Process a turn
        turn_payload = {
            "session_id": session_id,
            "participant": "Alice",
            "transcript": "I believe remote work offers great flexibility and work-life balance.",
            "duration": 45.5
        }

        response = client.post("/api/gd/turn", json=turn_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["participant"] == "Alice"
        assert "scores" in data
        assert "feedback" in data
        assert "next_participant" in data
        assert "round_complete" in data

    def test_gd_end(self):
        """Test ending a GD session"""
        # Start and complete a minimal session
        start_payload = {
            "topic": "Technology in education",
            "participants": ["Alice"]
        }
        start_response = client.post("/api/gd/start", json=start_payload)
        session_id = start_response.json()["session_id"]

        # Add a turn
        turn_payload = {
            "session_id": session_id,
            "participant": "Alice",
            "transcript": "Technology can enhance education through interactive learning.",
            "duration": 30.0
        }
        client.post("/api/gd/turn", json=turn_payload)

        # End session
        end_payload = {"session_id": session_id}
        response = client.post("/api/gd/end", json=end_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert "transcript" in data
        assert "participants" in data
        assert "summary" in data
        assert len(data["participants"]) == 1


class TestHREndpoints:
    """Test HR Interview endpoints"""

    def test_hr_start(self):
        """Test starting an HR interview"""
        payload = {
            "candidate_name": "John Doe",
            "skills": ["Python", "JavaScript", "SQL"]
        }

        response = client.post("/api/hr/start", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "session_id" in data
        assert data["candidate_name"] == payload["candidate_name"]
        assert "question" in data
        assert "question_type" in data

    def test_hr_answer(self):
        """Test processing an HR answer"""
        # Start interview
        start_payload = {
            "candidate_name": "Jane Smith",
            "skills": ["Python", "Machine Learning"]
        }
        start_response = client.post("/api/hr/start", json=start_payload)
        session_id = start_response.json()["session_id"]

        # Submit answer
        answer_payload = {
            "session_id": session_id,
            "answer": "I have 3 years of experience in Python development, focusing on web applications and data processing."}

        response = client.post("/api/hr/answer", json=answer_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert "scores" in data
        assert "feedback" in data
        assert "next_question" in data or data.get("interview_complete")

    def test_hr_end(self):
        """Test ending an HR interview"""
        # Start interview
        start_payload = {
            "candidate_name": "Bob Wilson",
            "skills": ["Java", "Spring Boot"]
        }
        start_response = client.post("/api/hr/start", json=start_payload)
        session_id = start_response.json()["session_id"]

        # Submit multiple answers to complete interview
        for i in range(3):
            answer_payload = {
                "session_id": session_id,
                "answer": f"This is my answer to question {i+1}. I have relevant experience and skills."
            }
            client.post("/api/hr/answer", json=answer_payload)

        # End interview
        end_payload = {"session_id": session_id}
        response = client.post("/api/hr/end", json=end_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert "overall_scores" in data
        assert "feedback_report" in data


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_session_gd(self):
        """Test invalid session ID for GD"""
        payload = {
            "session_id": "invalid-session-id",
            "participant": "Alice",
            "transcript": "Test",
            "duration": 30.0
        }

        response = client.post("/api/gd/turn", json=payload)
        assert response.status_code == 404

    def test_invalid_session_hr(self):
        """Test invalid session ID for HR"""
        payload = {
            "session_id": "invalid-session-id",
            "answer": "Test answer"
        }

        response = client.post("/api/hr/answer", json=payload)
        assert response.status_code == 404

    def test_missing_fields_gd_start(self):
        """Test missing required fields in GD start"""
        payload = {"topic": "Test topic"}  # Missing participants

        response = client.post("/api/gd/start", json=payload)
        # Should handle gracefully or return validation error
        assert response.status_code in [200, 422]  # 200 if defaults used, 422 if validation


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__])
