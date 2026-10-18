import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import config


bearer_scheme = HTTPBearer(auto_error=False)


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None or not config.JWT_SECRET_KEY:
        raise HTTPException(status_code=401, detail="缺少有效的Bearer Token")
    try:
        claims = jwt.decode(
            credentials.credentials,
            config.JWT_SECRET_KEY,
            algorithms=[config.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Token无效或已过期") from exc
    return claims
