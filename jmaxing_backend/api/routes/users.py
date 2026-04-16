"""User management routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from core.security import get_current_user
from core.exceptions import NotFoundException
from models.user import User
from schemas.user import UserProfile, UserUpdate, UserPublic

router = APIRouter()


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user's profile."""
    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        avatar_url=current_user.avatar_url,
        bio=current_user.bio,
        location=current_user.location,
        website=current_user.website,
        github=current_user.github,
        twitter=current_user.twitter,
        balance=float(current_user.balance),
        reputation=current_user.reputation,
        rank=current_user.rank,
        tasks_completed=current_user.tasks_completed,
        followers=current_user.followers_count,
        following=current_user.following_count,
        joinedAt=current_user.created_at,
    )


@router.put("/me", response_model=UserProfile)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile."""
    update_data = user_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        setattr(current_user, field, value)

    db.commit()
    db.refresh(current_user)

    return UserProfile(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        avatar_url=current_user.avatar_url,
        bio=current_user.bio,
        location=current_user.location,
        website=current_user.website,
        github=current_user.github,
        twitter=current_user.twitter,
        balance=float(current_user.balance),
        reputation=current_user.reputation,
        rank=current_user.rank,
        tasks_completed=current_user.tasks_completed,
        followers=current_user.followers_count,
        following=current_user.following_count,
        joinedAt=current_user.created_at,
    )


@router.get("/{username}", response_model=UserProfile)
async def get_user_by_username(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user profile by username."""
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise NotFoundException("User not found")

    # Check if current user follows this user
    from models.social import Follow
    is_following = db.query(Follow).filter(
        Follow.follower_id == current_user.id,
        Follow.following_id == user.id
    ).first() is not None

    return UserProfile(
        id=user.id,
        username=user.username,
        email=user.email,
        avatar_url=user.avatar_url,
        bio=user.bio,
        location=user.location,
        website=user.website,
        github=user.github,
        twitter=user.twitter,
        balance=float(user.balance),
        reputation=user.reputation,
        rank=user.rank,
        tasks_completed=user.tasks_completed,
        followers=user.followers_count,
        following=user.following_count,
        is_following=is_following,
        joinedAt=user.created_at,
    )


@router.get("/search", response_model=List[UserPublic])
async def search_users(
    q: str,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Search users by username."""
    users = db.query(User).filter(
        User.username.ilike(f"%{q}%")
    ).limit(limit).offset(offset).all()

    return [
        UserPublic(
            id=user.id,
            username=user.username,
            avatar_url=user.avatar_url,
            reputation=user.reputation,
            rank=user.rank,
        )
        for user in users
    ]
