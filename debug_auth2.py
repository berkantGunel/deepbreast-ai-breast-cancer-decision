"""Direct auth test."""
from src.api.database import SessionLocal
from src.api.utils.auth import (
    create_access_token, decode_token, get_user_by_id,
    SECRET_KEY, ALGORITHM
)
from datetime import timedelta

db = SessionLocal()

# Create a token manually
token = create_access_token(
    data={"sub": 5, "email": "yenitest@test.com"},
    expires_delta=timedelta(hours=1)
)
print(f"Created token: {token}")

# Decode it
result = decode_token(token)
print(f"Decoded result: {result}")

if result:
    user = get_user_by_id(db, result.user_id)
    print(f"User found: {user}")
    if user:
        print(f"User details: id={user.id}, email={user.email}, username={user.username}")

db.close()
