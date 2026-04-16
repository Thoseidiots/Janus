"""Social features routes - Feed, Posts, Comments, Follows, Referrals."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from core.database import get_db
from core.security import get_current_user
from core.exceptions import NotFoundException, ConflictException
from models.user import User
from models.social import Post, PostLike, Comment, Follow, Referral, PostType

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Feed & Posts
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/feed")
async def get_feed(
    tab: str = "following",  # following or explore
    limit: int = Query(20, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized feed.

    - **following**: Posts from users you follow
    - **explore**: Trending posts from everyone
    """
    query = db.query(Post).filter(Post.is_deleted == False)

    if tab == "following":
        # Get IDs of users current user follows
        following_ids = db.query(Follow.following_id).filter(
            Follow.follower_id == current_user.id
        ).subquery()

        query = query.filter(
            or_(
                Post.user_id.in_(following_ids),
                Post.user_id == current_user.id  # Include own posts
            )
        )

    # Order by creation time
    posts = query.order_by(desc(Post.created_at)).limit(limit).offset(offset).all()

    # Format response
    result = []
    for post in posts:
        # Check if current user liked
        is_liked = db.query(PostLike).filter(
            PostLike.post_id == post.id,
            PostLike.user_id == current_user.id
        ).first() is not None

        result.append({
            "id": str(post.id),
            "userId": str(post.user_id),
            "username": post.user.username,
            "avatar": post.user.avatar_url,
            "type": post.type.value,
            "content": post.content,
            "code": post.code,
            "language": post.language,
            "jobId": str(post.job_id) if post.job_id else None,
            "submissionId": str(post.submission_id) if post.submission_id else None,
            "score": float(post.score) if post.score else None,
            "payout": float(post.payout) if post.payout else None,
            "likes": post.likes_count,
            "comments": post.comments_count,
            "shares": post.shares_count,
            "timestamp": int(post.created_at.timestamp() * 1000),
            "isLiked": is_liked,
            "images": post.images or [],
            "tags": post.tags or [],
        })

    return result


@router.post("/posts", status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new post."""
    # Validate post type
    try:
        post_type = PostType(post_data.get("type", "status"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid post type")

    new_post = Post(
        user_id=current_user.id,
        type=post_type,
        content=post_data["content"],
        code=post_data.get("code"),
        language=post_data.get("language"),
        images=post_data.get("images", []),
        tags=post_data.get("tags", []),
    )

    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    return {
        "id": str(new_post.id),
        "userId": str(new_post.user_id),
        "username": current_user.username,
        "avatar": current_user.avatar_url,
        "type": new_post.type.value,
        "content": new_post.content,
        "code": new_post.code,
        "language": new_post.language,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "timestamp": int(new_post.created_at.timestamp() * 1000),
        "isLiked": False,
        "images": new_post.images or [],
        "tags": new_post.tags or [],
    }


@router.get("/posts/{post_id}")
async def get_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get post details."""
    post = db.query(Post).filter(Post.id == post_id).first()

    if not post:
        raise NotFoundException("Post not found")

    # Check if liked
    is_liked = db.query(PostLike).filter(
        PostLike.post_id == post.id,
        PostLike.user_id == current_user.id
    ).first() is not None

    return {
        "id": str(post.id),
        "userId": str(post.user_id),
        "username": post.user.username,
        "avatar": post.user.avatar_url,
        "type": post.type.value,
        "content": post.content,
        "code": post.code,
        "language": post.language,
        "likes": post.likes_count,
        "comments": post.comments_count,
        "shares": post.shares_count,
        "timestamp": int(post.created_at.timestamp() * 1000),
        "isLiked": is_liked,
        "images": post.images or [],
        "tags": post.tags or [],
    }


@router.post("/posts/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Like a post."""
    # Check if post exists
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise NotFoundException("Post not found")

    # Check if already liked
    existing_like = db.query(PostLike).filter(
        PostLike.post_id == post_id,
        PostLike.user_id == current_user.id
    ).first()

    if existing_like:
        raise ConflictException("Already liked this post")

    # Create like
    new_like = PostLike(
        post_id=post_id,
        user_id=current_user.id
    )
    db.add(new_like)
    db.commit()

    return None


@router.delete("/posts/{post_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unlike a post."""
    like = db.query(PostLike).filter(
        PostLike.post_id == post_id,
        PostLike.user_id == current_user.id
    ).first()

    if not like:
        raise NotFoundException("Like not found")

    db.delete(like)
    db.commit()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Comments
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/posts/{post_id}/comments")
async def get_comments(
    post_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comments for a post."""
    comments = db.query(Comment).filter(
        Comment.post_id == post_id,
        Comment.is_deleted == False,
        Comment.parent_id == None  # Only top-level comments
    ).order_by(Comment.created_at).limit(limit).offset(offset).all()

    result = []
    for comment in comments:
        result.append({
            "id": str(comment.id),
            "postId": str(comment.post_id),
            "userId": str(comment.user_id),
            "username": comment.user.username if hasattr(comment, 'user') else "Unknown",
            "avatar": comment.user.avatar_url if hasattr(comment, 'user') else None,
            "content": comment.content,
            "likes": comment.likes_count,
            "timestamp": int(comment.created_at.timestamp() * 1000),
        })

    return result


@router.post("/posts/{post_id}/comment", status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: UUID,
    comment_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a comment to a post."""
    # Check if post exists
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise NotFoundException("Post not found")

    new_comment = Comment(
        post_id=post_id,
        user_id=current_user.id,
        content=comment_data["content"],
        parent_id=comment_data.get("parentId")
    )

    db.add(new_comment)

    # Update post comment count
    post.comments_count += 1

    db.commit()
    db.refresh(new_comment)

    return {
        "id": str(new_comment.id),
        "postId": str(new_comment.post_id),
        "userId": str(new_comment.user_id),
        "username": current_user.username,
        "avatar": current_user.avatar_url,
        "content": new_comment.content,
        "likes": 0,
        "timestamp": int(new_comment.created_at.timestamp() * 1000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Follow System
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/users/{user_id}/follow", status_code=status.HTTP_204_NO_CONTENT)
async def follow_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Follow a user."""
    if str(user_id) == str(current_user.id):
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    # Check if user exists
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise NotFoundException("User not found")

    # Check if already following
    existing_follow = db.query(Follow).filter(
        Follow.follower_id == current_user.id,
        Follow.following_id == user_id
    ).first()

    if existing_follow:
        raise ConflictException("Already following this user")

    # Create follow
    new_follow = Follow(
        follower_id=current_user.id,
        following_id=user_id
    )
    db.add(new_follow)
    db.commit()

    return None


@router.delete("/users/{user_id}/follow", status_code=status.HTTP_204_NO_CONTENT)
async def unfollow_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unfollow a user."""
    follow = db.query(Follow).filter(
        Follow.follower_id == current_user.id,
        Follow.following_id == user_id
    ).first()

    if not follow:
        raise NotFoundException("Not following this user")

    db.delete(follow)
    db.commit()

    return None


@router.get("/users/{user_id}/followers")
async def get_followers(
    user_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get user's followers."""
    follows = db.query(Follow).filter(
        Follow.following_id == user_id
    ).limit(limit).offset(offset).all()

    result = []
    for follow in follows:
        user = follow.follower
        result.append({
            "id": str(user.id),
            "username": user.username,
            "avatar": user.avatar_url,
            "reputation": user.reputation,
            "rank": user.rank,
        })

    return result


@router.get("/users/{user_id}/following")
async def get_following(
    user_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get users that this user follows."""
    follows = db.query(Follow).filter(
        Follow.follower_id == user_id
    ).limit(limit).offset(offset).all()

    result = []
    for follow in follows:
        user = follow.following
        result.append({
            "id": str(user.id),
            "username": user.username,
            "avatar": user.avatar_url,
            "reputation": user.reputation,
            "rank": user.rank,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Referrals
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/referrals")
async def get_referral_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get referral statistics for current user."""
    referrals = db.query(Referral).filter(
        Referral.referrer_id == current_user.id,
        Referral.referred_id != current_user.id  # Exclude placeholder
    ).all()

    total_earned = sum(float(r.referrer_reward) for r in referrals if r.is_rewarded)

    return {
        "totalReferrals": len(referrals),
        "totalEarned": total_earned,
        "pendingRewards": sum(
            float(r.referrer_reward) for r in referrals if not r.is_rewarded
        ),
        "referrals": [
            {
                "id": str(r.id),
                "referredUsername": db.query(User).filter(User.id == r.referred_id).first().username if db.query(User).filter(User.id == r.referred_id).first() else "Unknown",
                "reward": float(r.referrer_reward),
                "isRewarded": r.is_rewarded,
                "createdAt": int(r.created_at.timestamp() * 1000),
            }
            for r in referrals
        ],
    }
