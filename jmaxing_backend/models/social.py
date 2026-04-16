"""Social features models."""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, ForeignKey, DECIMAL, Enum, CheckConstraint, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from core.database import Base


class PostType(str, enum.Enum):
    """Post type enumeration."""
    SUBMISSION = "submission"
    JOB = "job"
    ACHIEVEMENT = "achievement"
    STATUS = "status"
    SHOWCASE = "showcase"


class Post(Base):
    """Social post model."""

    __tablename__ = "posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(Enum(PostType), nullable=False, index=True)
    content = Column(Text, nullable=False)

    # Code content
    code = Column(Text, nullable=True)
    language = Column(String(50), nullable=True)

    # References
    job_id = Column(UUID(as_uuid=True), nullable=True)
    submission_id = Column(UUID(as_uuid=True), nullable=True)

    # Job/submission details
    score = Column(DECIMAL(5, 2), nullable=True)
    payout = Column(DECIMAL(10, 2), nullable=True)

    # Media
    images = Column(ARRAY(Text), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)

    # Engagement
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)

    # Status
    is_pinned = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="posts")
    likes = relationship("PostLike", back_populates="post", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Post {self.id} by {self.user_id}>"


class PostLike(Base):
    """Post like model."""

    __tablename__ = "post_likes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    post = relationship("Post", back_populates="likes")

    __table_args__ = (
        CheckConstraint("post_id IS NOT NULL AND user_id IS NOT NULL"),
    )

    def __repr__(self):
        return f"<PostLike {self.post_id} by {self.user_id}>"


class Comment(Base):
    """Comment model."""

    __tablename__ = "comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("comments.id", ondelete="CASCADE"), nullable=True, index=True)
    content = Column(Text, nullable=False)
    likes_count = Column(Integer, default=0)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    post = relationship("Post", back_populates="comments")
    parent = relationship("Comment", remote_side=[id], backref="replies")

    def __repr__(self):
        return f"<Comment {self.id} on {self.post_id}>"


class Follow(Base):
    """Follow relationship model."""

    __tablename__ = "follows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    follower_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    following_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("follower_id != following_id"),
    )

    def __repr__(self):
        return f"<Follow {self.follower_id} -> {self.following_id}>"


class Referral(Base):
    """Referral model."""

    __tablename__ = "referrals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    referrer_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    referred_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    code = Column(String(20), unique=True, nullable=False, index=True)

    # Rewards
    referrer_reward = Column(DECIMAL(10, 2), default=100.00)
    referred_reward = Column(DECIMAL(10, 2), default=50.00)
    is_rewarded = Column(Boolean, default=False)
    rewarded_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Referral {self.code}>"
