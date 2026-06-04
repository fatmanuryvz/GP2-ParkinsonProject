import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("Testing Endpoints...")
    
    # 1. Login test
    print("\n--- Testing Login ---")
    login_data = {"username": "dr_ahmet", "password": "doctor123"}
    r = requests.post(f"{BASE_URL}/api/login", json=login_data)
    print(f"Doc Login status: {r.status_code}")
    print(f"Doc Login response: {r.json()}")
    assert r.status_code == 200
    assert r.json()["role"] == "doctor"
    
    patient_login = {"username": "fatma_demir", "password": "patient123"}
    r_pat = requests.post(f"{BASE_URL}/api/login", json=patient_login)
    print(f"Pat Login status: {r_pat.status_code}")
    print(f"Pat Login response: {r_pat.json()}")
    assert r_pat.status_code == 200
    assert r_pat.json()["role"] == "patient"
    
    bad_login = {"username": "dr_ahmet", "password": "wrongpassword"}
    r_bad = requests.post(f"{BASE_URL}/api/login", json=bad_login)
    print(f"Bad Login status (should be 401): {r_bad.status_code}")
    assert r_bad.status_code == 401

    # 2. Get patients list
    print("\n--- Testing Patients List ---")
    r_list = requests.get(f"{BASE_URL}/api/patients")
    print(f"Patients list status: {r_list.status_code}")
    patients = r_list.json()
    print(f"Found {len(patients)} patients")
    for p in patients:
        print(f"Patient: {p['name']} (ID: {p['id']}), age: {p['age']}, gender: {p['gender']}, tests: {p['test_count']}")
    assert r_list.status_code == 200
    assert len(patients) >= 2

    # 3. Get specific patient detail (Fatma Demir: H001)
    print("\n--- Testing Patient Detail ---")
    r_det = requests.get(f"{BASE_URL}/api/patients/H001")
    print(f"Patient detail status: {r_det.status_code}")
    det = r_det.json()
    print(f"Name: {det['name']}, Notes: {det['doctor_notes']}, History items: {len(det['history'])}")
    assert r_det.status_code == 200
    assert det['id'] == 'H001'

    # 4. Save doctor notes
    print("\n--- Testing Doctor Notes Saving ---")
    notes_payload = {
        "doctor_notes": "Yeni klinik not: Hastanın durumunda stabil seyir izleniyor.",
        "doctor_advise": "Günde 3 kez egzersiz yapılması önerilir."
    }
    r_notes = requests.post(f"{BASE_URL}/api/patients/H001/notes", json=notes_payload)
    print(f"Save notes status: {r_notes.status_code}")
    assert r_notes.status_code == 200
    
    # Verify notes updated
    r_det2 = requests.get(f"{BASE_URL}/api/patients/H001")
    det2 = r_det2.json()
    print(f"Updated Notes: {det2['doctor_notes']}")
    assert det2['doctor_notes'] == notes_payload['doctor_notes']
    assert det2['doctor_advise'] == notes_payload['doctor_advise']

    # 5. Save test results
    print("\n--- Testing Save Test ---")
    test_payload = {
        "test_type": "tapping",
        "medication": True,
        "result": {
            "tap_rate": 3.5,
            "amp_decay_pct": 12.0,
            "rhythm_cov": 22.0,
            "tap_count": 35,
            "updrs_score": 1,
            "severity": "hafif"
        }
    }
    r_test = requests.post(f"{BASE_URL}/api/patients/H001/tests", json=test_payload)
    print(f"Save test status: {r_test.status_code}")
    assert r_test.status_code == 200
    
    # Verify test saved and appears in clinical report
    r_report = requests.get(f"{BASE_URL}/report/clinical?patient_id=H001")
    print(f"Clinical report status: {r_report.status_code}")
    report = r_report.json()
    print(f"Overall status: {report.get('overall_status')}, comparisons: {len(report.get('comparisons', []))}")
    assert r_report.status_code == 200

    print("\nAll integration tests passed successfully!")

if __name__ == "__main__":
    test_api()
