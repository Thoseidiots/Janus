"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from core.database import get_db
from core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_referral_code,
)
from core.config import settings
from core.exceptions import UnauthorizedException, ConflictException, ValidationException
from models.user import User, UserSession
from models.social import Referral
from schemas.user import UserCreate, LoginRequest, TokenResponse, RefreshTokenRequest, UserProfile

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    referral_code: str = None,
    db: Session = Depends(get_db)
):
    """
    Register a new user.

    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Strong password (min 8 characters, uppercase, lowercase, digit)
    - **referral_code**: Optional referral code from existing user
    """
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise ConflictException("Username already taken")

    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise ConflictException("Email already registered")

    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
    )

    db.add(new_user)
    db.flush()  # Get user ID without committing

    # Handle referral
    referrer = None
    if referral_code:
        referral = db.query(Referral).filter(Referral.code == referral_code).first()
        if referral:
            referrer = db.query(User).filter(User.id == referral.referrer_id).first()

            # Create referral record
            new_referral = Referral(
                referrer_id=referrer.id,
                referred_id=new_user.id,
                code=referral_code,
            )
            db.add(new_referral)

            # Award credits
            if not new_referral.is_rewarded:
                referrer.balance += settings.REFERRER_REWARD
                new_user.balance += settings.REFERRED_REWARD
                new_referral.is_rewarded = True
                new_referral.rewarded_at = datetime.utcnow()

    db.commit()
    db.refresh(new_user)

    # Create tokens
    access_token = create_access_token(data={"sub": str(new_user.id)})
    refresh_token = create_refresh_token(data={"sub": str(new_user.id)})

    # Create session
    session = UserSession(
        user_id=new_user.id,
        token=access_token,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(session)
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Login with username/email and password.

    Returns access token and refresh token.
    """
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == login_data.username) | (User.email == login_data.username)
    ).first()

    if not user:
        raise UnauthorizedException("Invalid credentials")

    # Verify password
    if not verify_password(login_data.password, user.password_hash):
        raise UnauthorizedException("Invalid credentials")

    # Check if user is active
    if not user.is_active:
        raise UnauthorizedException("Account is disabled")

    # Update last login
    user.last_login = datetime.utcnow()

    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})

    # Create session
    session = UserSession(
        user_id=user.id,
        token=access_token,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(session)
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    try:
        payload = decode_token(refresh_data.refresh_token)
        user_id = payload.get("sub")
        token_type = payload.get("type")

        if token_type != "refresh":
            raise UnauthorizedException("Invalid token type")

        if not user_id:
            raise UnauthorizedException("Invalid token")

    except Exception:
        raise UnauthorizedException("Invalid or expired refresh token")

    # Find user
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise UnauthorizedException("User not found or inactive")

    # Verify session exists
    session = db.query(UserSession).filter(
        UserSession.user_id == user_id,
        UserSession.refresh_token == refresh_data.refresh_token
    ).first()

    if not session:
        raise UnauthorizedException("Session not found")

    # Create new tokens
    new_access_token = create_access_token(data={"sub": str(user.id)})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})

    # Update session
    session.token = new_access_token
    session.refresh_token = new_refresh_token
    session.expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    db.commit()

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Logout and invalidate session.
    """
    # Find and delete session
    session = db.query(UserSession).filter(
        UserSession.refresh_token == refresh_data.refresh_token
    ).first()

    if session:
        db.delete(session)
        db.commit()

    return None


@router.get("/referral-code")
async def get_referral_code(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get or generate referral code for current user.
    """
    # Check if user already has a referral code
    existing_referral = db.query(Referral).filter(
        Referral.referrer_id == current_user.id
    ).first()

    if existing_referral:
        code = existing_referral.code
    else:
        # Generate new code
        code = generate_referral_code(str(current_user.id))

        # Create placeholder referral (will be completed when someone uses it)
        referral = Referral(
            referrer_id=current_user.id,
            referred_id=current_user.id,  # Temporary
            code=code,
        )
        db.add(referral)
        db.commit()

    return {
        "code": code,
        "referral_url": f"{settings.FRONTEND_URL}/register?ref={code}",
        "referrer_reward": float(settings.REFERRER_REWARD),
        "referred_reward": float(settings.REFERRED_REWARD),
    }


# Import at end to avoid circular import
from core.security import get_current_user
