"""Direct decode test."""
from jose import jwt, JWTError
from src.api.utils.auth import SECRET_KEY, ALGORITHM, create_access_token, decode_token
from datetime import timedelta

# Create token with STRING sub
token = create_access_token(
    data={"sub": "5", "email": "test@test.com"},  # Changed to string!
    expires_delta=timedelta(hours=1)
)
print(f"Token: {token}")
print(f"SECRET_KEY: {SECRET_KEY}")
print(f"ALGORITHM: {ALGORITHM}")

# Try decode
try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(f"Decoded payload: {payload}")
    print(f"sub type: {type(payload.get('sub'))}")
    print(f"sub value: {payload.get('sub')}")
except JWTError as e:
    print(f"JWTError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Test decode_token function
print("\n--- Testing decode_token function ---")
result = decode_token(token)
print(f"decode_token result: {result}")
