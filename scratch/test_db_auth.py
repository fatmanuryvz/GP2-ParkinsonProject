import requests

BASE_URL = "http://127.0.0.1:5000"

def test_login_doctor():
    print("Testing doctor login...")
    payload = {"username": "dr_ahmet", "password": "doctor123"}
    res = requests.post(f"{BASE_URL}/api/login", json=payload)
    print("Status:", res.status_code)
    print("Response:", res.json())
    assert res.status_code == 200
    assert res.json()["role"] == "doctor"
    print("Doctor login OK!\n")

def test_login_patient():
    print("Testing patient login...")
    payload = {"username": "fatma_demir", "password": "patient123"}
    res = requests.post(f"{BASE_URL}/api/login", json=payload)
    print("Status:", res.status_code)
    print("Response:", res.json())
    assert res.status_code == 200
    assert res.json()["role"] == "patient"
    print("Patient login OK!\n")

def test_login_invalid():
    print("Testing invalid credentials...")
    payload = {"username": "dr_ahmet", "password": "wrongpassword"}
    res = requests.post(f"{BASE_URL}/api/login", json=payload)
    print("Status:", res.status_code)
    print("Response:", res.json())
    assert res.status_code == 401
    print("Invalid login check OK!\n")

def test_get_patients():
    print("Testing get patients...")
    res = requests.get(f"{BASE_URL}/api/patients")
    print("Status:", res.status_code)
    print("Patients:", len(res.json()))
    assert res.status_code == 200
    print("Get patients OK!\n")

if __name__ == "__main__":
    try:
        test_login_doctor()
        test_login_patient()
        test_login_invalid()
        test_get_patients()
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    except Exception as e:
        print("Test failed:", e)
