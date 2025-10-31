from flask import Blueprint, request, jsonify
import subprocess
import tempfile
import os

technical_bp = Blueprint("technical", __name__)

# ✅ Mock question bank (replace with DB fetch if you have it)
technical_questions = [
    {
        "id": 1,
        "question": "Write a program to print the sum of two numbers.",
        "language": "python",
        "test_cases": [
            {"input": "2 3", "expected_output": "5"},
            {"input": "10 20", "expected_output": "30"}
        ]
    }
]

@technical_bp.route("/technical/questions", methods=["GET"])
def get_questions():
    """Fetch available technical questions"""
    return jsonify({"questions": technical_questions})


@technical_bp.route("/technical/run_code", methods=["POST"])
def run_code():
    """
    Execute user-submitted code securely for coding assessment.
    Validates code before execution, runs against all test cases,
    and returns detailed pass/fail results.
    """
    data = request.get_json()
    code = data.get("code", "")
    language = data.get("language", "python")
    question_id = data.get("question_id")

    # ✅ Validate inputs
    if not code or len(code.strip()) < 5:
        return jsonify({
            "success": True,
            "result": {
                "status": "completed",
                "test_results": [{
                    "test_case_number": 1,
                    "input": "",
                    "expected_output": "",
                    "actual_output": "⚠️ No code submitted.",
                    "passed": False
                }],
                "passed_count": 0,
                "total_count": 1
            }
        }), 200

    if not any(x in code for x in ["print", "return", "def", "System.out", "cout"]):
        return jsonify({
            "success": True,
            "result": {
                "status": "completed",
                "test_results": [{
                    "test_case_number": 1,
                    "input": "",
                    "expected_output": "",
                    "actual_output": "⚠️ Code does not contain any executable statements.",
                    "passed": False
                }],
                "passed_count": 0,
                "total_count": 1
            }
        }), 200

    # ✅ Get test cases for the question
    question = next((q for q in technical_questions if q["id"] == question_id), None)
    if not question:
        return jsonify({
            "success": False,
            "error": f"No question found with ID {question_id}"
        }), 404

    test_cases = question.get("test_cases", [])
    if not test_cases:
        return jsonify({
            "success": False,
            "error": "No test cases found for this question."
        }), 400

    # ✅ Create temp file to safely execute code
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        temp_path = temp_file.name

    results = []
    passed_count = 0

    # ✅ Run code for each test case
    for i, tc in enumerate(test_cases, start=1):
        try:
            input_data = tc["input"]
            expected_output = tc["expected_output"]

            process = subprocess.run(
                ["python", temp_path],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=3
            )

            actual_output = process.stdout.strip()
            passed = actual_output == expected_output
            if passed:
                passed_count += 1

            results.append({
                "test_case_number": i,
                "input": input_data,
                "expected_output": expected_output,
                "actual_output": actual_output or "⚠️ No output",
                "passed": passed
            })

        except subprocess.TimeoutExpired:
            results.append({
                "test_case_number": i,
                "input": tc["input"],
                "expected_output": tc["expected_output"],
                "actual_output": "⏱️ Execution timed out",
                "passed": False
            })
        except Exception as e:
            results.append({
                "test_case_number": i,
                "input": tc["input"],
                "expected_output": tc["expected_output"],
                "actual_output": f"💥 Error: {str(e)}",
                "passed": False
            })

    os.remove(temp_path)  # ✅ Clean up temp file

    return jsonify({
        "success": True,
        "result": {
            "status": "completed",
            "test_results": results,
            "passed_count": passed_count,
            "total_count": len(test_cases)
        }
    }), 200
