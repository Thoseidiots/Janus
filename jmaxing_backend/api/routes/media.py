"""Media platform routes."""

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime
from uuid import UUID
import os
import hashlib

from core.database import get_db
from core.security import get_current_user
from core.config import settings
from core.exceptions import NotFoundException, ValidationException, ConflictException
from models.user import User
from models.media import MediaCategory, MediaItem, MediaLike, MediaComment, MediaType

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/categories")
async def list_categories(
    db: Session = Depends(get_db)
):
    """List all media categories."""
    categories = db.query(MediaCategory).order_by(MediaCategory.name).all()

    return [
        {
            "id": str(cat.id),
            "name": cat.name,
            "slug": cat.slug,
            "description": cat.description,
            "icon": cat.icon,
            "color": cat.color,
            "itemCount": cat.item_count,
            "followers": cat.followers_count,
            "trending": cat.trending,
            "isAutoCreated": cat.is_auto_created,
        }
        for cat in categories
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Media Items
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
async def list_media(
    category: Optional[str] = None,
    type: Optional[str] = None,
    sort: str = "recent",  # recent, popular, trending
    limit: int = Query(20, le=100),
    offset: int = 0,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List media items."""
    query = db.query(MediaItem).filter(
        MediaItem.is_approved == True,
        MediaItem.is_processing == False
    )

    # Filter by category
    if category:
        cat = db.query(MediaCategory).filter(MediaCategory.slug == category).first()
        if cat:
            query = query.filter(MediaItem.category_id == cat.id)

    # Filter by type
    if type:
        try:
            media_type = MediaType(type)
            query = query.filter(MediaItem.type == media_type)
        except ValueError:
            pass

    # Sort
    if sort == "popular":
        query = query.order_by(desc(MediaItem.likes_count))
    elif sort == "trending":
        query = query.order_by(desc(MediaItem.views_count))
    else:  # recent
        query = query.order_by(desc(MediaItem.created_at))

    media_items = query.limit(limit).offset(offset).all()

    # Format response
    result = []
    for item in media_items:
        # Check if current user liked
        is_liked = False
        if current_user:
            is_liked = db.query(MediaLike).filter(
                MediaLike.media_id == item.id,
                MediaLike.user_id == current_user.id
            ).first() is not None

        result.append({
            "id": str(item.id),
            "userId": str(item.user_id),
            "username": item.user.username,
            "avatar": item.user.avatar_url,
            "url": item.url,
            "thumbnailUrl": item.thumbnail_url,
            "type": item.type.value,
            "title": item.title,
            "description": item.description,
            "category": item.category.slug,
            "suggestedCategory": item.suggested_category.slug if item.suggested_category else None,
            "aiConfidence": float(item.ai_confidence) if item.ai_confidence else None,
            "tags": item.tags or [],
            "width": item.width,
            "height": item.height,
            "duration": item.duration,
            "fileSize": item.file_size,
            "mimeType": item.mime_type,
            "views": item.views_count,
            "likes": item.likes_count,
            "shares": item.shares_count,
            "comments": item.comments_count,
            "downloads": item.downloads_count,
            "isLiked": is_liked,
            "isProcessing": item.is_processing,
            "isFlagged": item.is_flagged,
            "isApproved": item.is_approved,
            "createdAt": int(item.created_at.timestamp() * 1000),
        })

    return result


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_media(
    title: str,
    description: Optional[str] = None,
    category: str = "other",
    tags: Optional[str] = None,  # comma-separated
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload media file.

    This is a simplified implementation. In production:
    - Validate file type and size
    - Generate thumbnails for images/videos
    - Process with AI for categorization
    - Upload to S3/Cloudinary
    - Return processing status
    """
    # Validate file size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.MAX_UPLOAD_SIZE:
        raise ValidationException(f"File size exceeds {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB limit")

    # Determine media type
    content_type = file.content_type or ""
    if content_type.startswith("image/"):
        media_type = MediaType.IMAGE
    elif content_type.startswith("video/"):
        media_type = MediaType.VIDEO
    elif content_type.startswith("audio/"):
        media_type = MediaType.AUDIO
    elif content_type == "image/gif":
        media_type = MediaType.GIF
    else:
        raise ValidationException("Unsupported file type")

    # Find category
    cat = db.query(MediaCategory).filter(MediaCategory.slug == category).first()
    if not cat:
        # Default to "other"
        cat = db.query(MediaCategory).filter(MediaCategory.slug == "other").first()

    # Generate unique filename
    file_hash = hashlib.sha256(file_content).hexdigest()[:16]
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{file_hash}{file_ext}"

    # Save file (local storage)
    file_path = os.path.join(settings.UPLOAD_DIR, "media", filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)

    # Create media item
    file_url = f"/uploads/media/{filename}"

    new_media = MediaItem(
        user_id=current_user.id,
        url=file_url,
        type=media_type,
        title=title,
        description=description,
        category_id=cat.id,
        tags=tags.split(",") if tags else [],
        file_size=file_size,
        mime_type=content_type,
    )

    db.add(new_media)

    # Update category item count
    cat.item_count += 1

    db.commit()
    db.refresh(new_media)

    return {
        "id": str(new_media.id),
        "userId": str(new_media.user_id),
        "username": current_user.username,
        "avatar": current_user.avatar_url,
        "url": new_media.url,
        "type": new_media.type.value,
        "title": new_media.title,
        "description": new_media.description,
        "category": cat.slug,
        "tags": new_media.tags or [],
        "fileSize": new_media.file_size,
        "mimeType": new_media.mime_type,
        "views": 0,
        "likes": 0,
        "shares": 0,
        "comments": 0,
        "downloads": 0,
        "isLiked": False,
        "isProcessing": False,
        "createdAt": int(new_media.created_at.timestamp() * 1000),
    }


@router.get("/{media_id}")
async def get_media(
    media_id: UUID,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get media details."""
    media = db.query(MediaItem).filter(MediaItem.id == media_id).first()

    if not media:
        raise NotFoundException("Media not found")

    # Increment view count
    media.views_count += 1
    db.commit()

    # Check if liked
    is_liked = False
    if current_user:
        is_liked = db.query(MediaLike).filter(
            MediaLike.media_id == media_id,
            MediaLike.user_id == current_user.id
        ).first() is not None

    return {
        "id": str(media.id),
        "userId": str(media.user_id),
        "username": media.user.username,
        "avatar": media.user.avatar_url,
        "url": media.url,
        "thumbnailUrl": media.thumbnail_url,
        "type": media.type.value,
        "title": media.title,
        "description": media.description,
        "category": media.category.slug,
        "suggestedCategory": media.suggested_category.slug if media.suggested_category else None,
        "aiConfidence": float(media.ai_confidence) if media.ai_confidence else None,
        "tags": media.tags or [],
        "width": media.width,
        "height": media.height,
        "duration": media.duration,
        "fileSize": media.file_size,
        "mimeType": media.mime_type,
        "views": media.views_count,
        "likes": media.likes_count,
        "shares": media.shares_count,
        "comments": media.comments_count,
        "downloads": media.downloads_count,
        "isLiked": is_liked,
        "isProcessing": media.is_processing,
        "isFlagged": media.is_flagged,
        "isApproved": media.is_approved,
        "createdAt": int(media.created_at.timestamp() * 1000),
    }


@router.post("/{media_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def like_media(
    media_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Like media."""
    media = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not media:
        raise NotFoundException("Media not found")

    # Check if already liked
    existing_like = db.query(MediaLike).filter(
        MediaLike.media_id == media_id,
        MediaLike.user_id == current_user.id
    ).first()

    if existing_like:
        raise ConflictException("Already liked this media")

    # Create like
    new_like = MediaLike(
        media_id=media_id,
        user_id=current_user.id
    )
    db.add(new_like)
    db.commit()

    return None


@router.delete("/{media_id}/like", status_code=status.HTTP_204_NO_CONTENT)
async def unlike_media(
    media_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unlike media."""
    like = db.query(MediaLike).filter(
        MediaLike.media_id == media_id,
        MediaLike.user_id == current_user.id
    ).first()

    if not like:
        raise NotFoundException("Like not found")

    db.delete(like)
    db.commit()

    return None


@router.get("/{media_id}/comments")
async def get_media_comments(
    media_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get media comments."""
    comments = db.query(MediaComment).filter(
        MediaComment.media_id == media_id,
        MediaComment.is_deleted == False,
        MediaComment.parent_id == None
    ).order_by(MediaComment.created_at).limit(limit).offset(offset).all()

    result = []
    for comment in comments:
        user = db.query(User).filter(User.id == comment.user_id).first()
        result.append({
            "id": str(comment.id),
            "mediaId": str(comment.media_id),
            "userId": str(comment.user_id),
            "username": user.username if user else "Unknown",
            "avatar": user.avatar_url if user else None,
            "content": comment.content,
            "likes": comment.likes_count,
            "timestamp": int(comment.created_at.timestamp() * 1000),
        })

    return result


@router.post("/{media_id}/comment", status_code=status.HTTP_201_CREATED)
async def create_media_comment(
    media_id: UUID,
    comment_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add comment to media."""
    media = db.query(MediaItem).filter(MediaItem.id == media_id).first()
    if not media:
        raise NotFoundException("Media not found")

    new_comment = MediaComment(
        media_id=media_id,
        user_id=current_user.id,
        content=comment_data["content"],
        parent_id=comment_data.get("parentId")
    )

    db.add(new_comment)

    # Update comment count
    media.comments_count += 1

    db.commit()
    db.refresh(new_comment)

    return {
        "id": str(new_comment.id),
        "mediaId": str(new_comment.media_id),
        "userId": str(new_comment.user_id),
        "username": current_user.username,
        "avatar": current_user.avatar_url,
        "content": new_comment.content,
        "likes": 0,
        "timestamp": int(new_comment.created_at.timestamp() * 1000),
    }


@router.get("/trending")
async def get_trending_media(
    limit: int = Query(20, le=100),
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get trending media (most views/likes in last 7 days)."""
    from datetime import timedelta

    week_ago = datetime.utcnow() - timedelta(days=7)

    media_items = db.query(MediaItem).filter(
        MediaItem.is_approved == True,
        MediaItem.is_processing == False,
        MediaItem.created_at >= week_ago
    ).order_by(
        desc(MediaItem.views_count + MediaItem.likes_count * 2)
    ).limit(limit).all()

    result = []
    for item in media_items:
        is_liked = False
        if current_user:
            is_liked = db.query(MediaLike).filter(
                MediaLike.media_id == item.id,
                MediaLike.user_id == current_user.id
            ).first() is not None

        result.append({
            "id": str(item.id),
            "userId": str(item.user_id),
            "username": item.user.username,
            "avatar": item.user.avatar_url,
            "url": item.url,
            "thumbnailUrl": item.thumbnail_url,
            "type": item.type.value,
            "title": item.title,
            "category": item.category.slug,
            "views": item.views_count,
            "likes": item.likes_count,
            "isLiked": is_liked,
            "createdAt": int(item.created_at.timestamp() * 1000),
        })

    return result
