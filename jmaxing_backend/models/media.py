"""Media platform models."""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, ForeignKey, BigInteger, DECIMAL, Enum, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from core.database import Base


class MediaType(str, enum.Enum):
    """Media type enumeration."""
    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    AUDIO = "audio"


class MediaCategory(Base):
    """Media category model."""

    __tablename__ = "media_categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    icon = Column(String(50), nullable=True)
    color = Column(String(7), nullable=True)

    # Stats
    item_count = Column(Integer, default=0)
    followers_count = Column(Integer, default=0)
    trending = Column(Boolean, default=False, index=True)

    # Auto-creation
    is_auto_created = Column(Boolean, default=False)
    created_from_searches = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    media_items = relationship("MediaItem", back_populates="category")

    def __repr__(self):
        return f"<MediaCategory {self.name}>"


class MediaItem(Base):
    """Media item model."""

    __tablename__ = "media_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Content
    url = Column(Text, nullable=False)
    thumbnail_url = Column(Text, nullable=True)
    type = Column(Enum(MediaType), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Categorization
    category_id = Column(UUID(as_uuid=True), ForeignKey("media_categories.id"), nullable=False, index=True)
    suggested_category_id = Column(UUID(as_uuid=True), ForeignKey("media_categories.id"), nullable=True)
    ai_confidence = Column(DECIMAL(3, 2), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)

    # Metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration = Column(Integer, nullable=True)  # seconds
    file_size = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)

    # Engagement
    views_count = Column(Integer, default=0, index=True)
    likes_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    downloads_count = Column(Integer, default=0)

    # Status
    is_processing = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="media_items")
    category = relationship("MediaCategory", back_populates="media_items", foreign_keys=[category_id])
    likes = relationship("MediaLike", back_populates="media", cascade="all, delete-orphan")
    comments = relationship("MediaComment", back_populates="media", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MediaItem {self.title}>"


class MediaLike(Base):
    """Media like model."""

    __tablename__ = "media_likes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    media = relationship("MediaItem", back_populates="likes")

    def __repr__(self):
        return f"<MediaLike {self.media_id} by {self.user_id}>"


class MediaComment(Base):
    """Media comment model."""

    __tablename__ = "media_comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id = Column(UUID(as_uuid=True), ForeignKey("media_items.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("media_comments.id", ondelete="CASCADE"), nullable=True, index=True)
    content = Column(Text, nullable=False)
    likes_count = Column(Integer, default=0)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    media = relationship("MediaItem", back_populates="comments")
    parent = relationship("MediaComment", remote_side=[id], backref="replies")

    def __repr__(self):
        return f"<MediaComment {self.id} on {self.media_id}>"
