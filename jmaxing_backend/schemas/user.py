"""User schemas."""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
from uuid import UUID


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8, max_length=100)

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """User update schema."""
    avatar_url: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)
    location: Optional[str] = Field(None, max_length=100)
    website: Optional[str] = Field(None, max_length=255)
    github: Optional[str] = Field(None, max_length=100)
    twitter: Optional[str] = Field(None, max_length=100)


class UserProfile(BaseModel):
    """User profile response schema."""
    id: UUID
    username: str
    email: str
    avatar_url: Optional[str]
    bio: Optional[str]
    location: Optional[str]
    website: Optional[str]
    github: Optional[str]
    twitter: Optional[str]
    balance: float
    reputation: int
    rank: int
    tasks_completed: int
    followers_count: int = Field(alias="followers")
    following_count: int = Field(alias="following")
    is_following: Optional[bool] = None
    created_at: datetime = Field(alias="joinedAt")

    class Config:
        from_attributes = True
        populate_by_name = True


class UserPublic(BaseModel):
    """Public user information."""
    id: UUID
    username: str
    avatar_url: Optional[str] = None
    reputation: int
    rank: int

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str
