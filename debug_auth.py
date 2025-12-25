"""Debug authentication issue."""
import requests
from jose import jwt

# Step 1: Login
print("="*50)
print("Step 1: Login")
resp = requests.post('http://localhost:8000/api/auth/login', json={
    'email': 'yenitest@test.com', 
    'password': 'Test1234!'
})
print(f"Login status: {resp.status_code}")
data = resp.json()
token = data['access_token']
print(f"Token: {token[:80]}...")

# Step 2: Check SECRET_KEY
print("\n" + "="*50)
print("Step 2: Check SECRET_KEY")
from src.api.utils.auth import SECRET_KEY, ALGORITHM
print(f"SECRET_KEY: {SECRET_KEY}")
print(f"ALGORITHM: {ALGORITHM}")

# Step 3: Decode token
print("\n" + "="*50)
print("Step 3: Decode token")
try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(f"Decoded successfully: {payload}")
except Exception as e:
    print(f"Decode error: {type(e).__name__}: {e}")

# Step 4: Try /me endpoint
print("\n" + "="*50)
print("Step 4: Test /me endpoint")
headers = {'Authorization': f'Bearer {token}'}
me_resp = requests.get('http://localhost:8000/api/auth/me', headers=headers)
print(f"/me status: {me_resp.status_code}")
print(f"/me body: {me_resp.text}")
