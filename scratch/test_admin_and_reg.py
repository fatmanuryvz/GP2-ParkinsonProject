import requests
import json
import random

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=== STARTING ADMIN & REGISTRATION INTEGRATION TESTS ===")
    
    # 1. Admin Login
    print("\n[Test 1] Testing Admin Login...")
    admin_credentials = {"username": "admin", "password": "admin123"}
    r = requests.post(f"{BASE_URL}/api/login", json=admin_credentials)
    print(f"Admin Login response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "Admin login failed!"
    assert r.json()["role"] == "admin"
    
    # 2. Get Admin Stats
    print("\n[Test 2] Testing Admin Stats...")
    r = requests.get(f"{BASE_URL}/api/admin/stats")
    print(f"Admin Stats response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "Getting admin stats failed!"
    stats = r.json()
    assert "doctor_count" in stats
    assert "patient_count" in stats
    assert "test_count" in stats
    assert "db_type" in stats
    
    # 3. Admin: Add Doctor
    print("\n[Test 3] Testing Admin Add Doctor...")
    rand_suffix = random.randint(1000, 9999)
    doc_username = f"dr_test_{rand_suffix}"
    doc_password = "doctorpassword"
    doc_name = f"Test Doctor {rand_suffix}"
    doc_specialty = "Nöroloji"
    
    doctor_payload = {
        "username": doc_username,
        "password": doc_password,
        "name": doc_name,
        "specialty": doc_specialty
    }
    r = requests.post(f"{BASE_URL}/api/admin/doctors", json=doctor_payload)
    print(f"Add Doctor response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "Admin adding doctor failed!"
    
    # 4. Get Doctors List
    print("\n[Test 4] Testing Admin Doctors List...")
    r = requests.get(f"{BASE_URL}/api/admin/doctors")
    print(f"Doctors list status: {r.status_code}")
    doctors = r.json()
    print(f"Found {len(doctors)} doctors")
    test_doc_found = False
    for d in doctors:
        if d["name"] == doc_name:
            test_doc_found = True
            doc_id = d["id"]
            break
    assert test_doc_found, "The newly created doctor was not found in the list!"
    
    # 5. Doctor Login
    print("\n[Test 5] Testing New Doctor Login...")
    r = requests.post(f"{BASE_URL}/api/login", json={"username": doc_username, "password": doc_password})
    print(f"New Doctor login response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "New doctor login failed!"
    
    # 6. Patient Registration (Needs approval)
    print("\n[Test 6] Testing Patient Registration (pending approval)...")
    pat_username = f"patient_{rand_suffix}"
    pat_password = "patientpassword"
    pat_name = f"Test Patient {rand_suffix}"
    
    reg_payload = {
        "username": pat_username,
        "password": pat_password,
        "name": pat_name,
        "age": 65,
        "gender": "Erkek"
    }
    r = requests.post(f"{BASE_URL}/api/register", json=reg_payload)
    print(f"Register response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "Patient registration endpoint failed!"
    assert "onay" in r.json()["message"].lower() or "talep" in r.json()["message"].lower()
    
    # 7. Try to Login as Unapproved Patient (should be 403)
    print("\n[Test 7] Testing Unapproved Patient Login (should fail with 403)...")
    r = requests.post(f"{BASE_URL}/api/login", json={"username": pat_username, "password": pat_password})
    print(f"Unapproved Patient login response: {r.status_code} - {r.json()}")
    assert r.status_code == 403, f"Expected 403 but got {r.status_code}"
    assert "onay" in r.json()["message"].lower() or "hekim" in r.json()["message"].lower()
    
    # 8. Check Admin Patients list to see the unapproved patient
    print("\n[Test 8] Checking Admin Patients list for the new patient...")
    r = requests.get(f"{BASE_URL}/api/admin/patients")
    print(f"Admin Patients list response: {r.status_code}")
    patients = r.json()
    new_patient_data = None
    for p in patients:
        if p["name"] == pat_name:
            new_patient_data = p
            break
    assert new_patient_data is not None, "New registration not found in admin patients list!"
    print(f"Found new patient in admin view: {new_patient_data}")
    assert new_patient_data["is_approved"] is False, "Patient should not be approved yet!"
    
    # 9. Log in as Doctor/Admin to Approve the Patient
    # Let's verify we have a pending list endpoint for the doctor
    print("\n[Test 9] Fetching Pending Patients for Doctor...")
    doc_session = requests.Session()
    # Log in doctor
    doc_session.post(f"{BASE_URL}/api/login", json={"username": "dr_ahmet", "password": "doctor123"})
    r = doc_session.get(f"{BASE_URL}/api/patients/pending")
    print(f"Pending patients for doctor response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    pending_list = r.json()
    patient_in_pending = False
    for p in pending_list:
        if p["name"] == pat_name:
            patient_in_pending = True
            break
    assert patient_in_pending, "Patient not found in doctor's pending list!"
    
    # Approve the patient
    print(f"\n[Test 10] Approving patient: {new_patient_data['id']}...")
    r = doc_session.post(f"{BASE_URL}/api/patients/approve/{new_patient_data['id']}")
    print(f"Approve response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    
    # 10. Login as newly approved patient (should succeed)
    print("\n[Test 11] Testing Approved Patient Login (should succeed)...")
    r = requests.post(f"{BASE_URL}/api/login", json={"username": pat_username, "password": pat_password})
    print(f"Approved Patient login response: {r.status_code} - {r.json()}")
    assert r.status_code == 200, "Approved patient login failed!"
    assert r.json()["role"] == "patient"
    
    # 11. Doctor: Add Patient Directly (should be approved immediately)
    print("\n[Test 12] Testing Doctor Add Patient Directly...")
    direct_pat_username = f"direct_pat_{rand_suffix}"
    direct_pat_password = "directpassword"
    direct_pat_name = f"Direct Patient {rand_suffix}"
    
    direct_payload = {
        "username": direct_pat_username,
        "password": direct_pat_password,
        "name": direct_pat_name,
        "age": 70,
        "gender": "Kadın"
    }
    r = doc_session.post(f"{BASE_URL}/api/patients", json=direct_payload)
    print(f"Doctor Direct Add Patient response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    direct_pat_id = r.json()["patient_id"]
    
    # 12. Test login of directly added patient (should succeed immediately)
    print("\n[Test 13] Testing Directly Added Patient Login (should succeed)...")
    r = requests.post(f"{BASE_URL}/api/login", json={"username": direct_pat_username, "password": direct_pat_password})
    print(f"Direct Patient login response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    
    # 13. Clean up: Delete doctor and patient
    print("\n[Test 14] Cleaning up test users...")
    # Admin delete doctor
    r = requests.delete(f"{BASE_URL}/api/admin/doctors/{doc_id}")
    print(f"Delete doctor response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    
    # Doctor delete directly added patient
    r = doc_session.delete(f"{BASE_URL}/api/patients/{direct_pat_id}")
    print(f"Delete direct patient response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    
    # Admin delete approved patient
    r = requests.delete(f"{BASE_URL}/api/admin/patients/{new_patient_data['id']}")
    print(f"Delete approved patient response: {r.status_code} - {r.json()}")
    assert r.status_code == 200
    
    print("\n=== ALL SYSTEM TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
