"""Project collaboration models."""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, ForeignKey, DECIMAL, Enum, CheckConstraint, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from core.database import Base


class ProjectType(str, enum.Enum):
    """Project type enumeration."""
    COMMUNITY = "community"
    GROUP = "group"
    SOLO = "solo"


class ProjectStatus(str, enum.Enum):
    """Project status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MemberRole(str, enum.Enum):
    """Project member role enumeration."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class RewardDistribution(str, enum.Enum):
    """Reward distribution type."""
    EQUAL = "equal"
    CONTRIBUTION_BASED = "contribution-based"
    MERIT_BASED = "merit-based"


class Project(Base):
    """Project model."""

    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    type = Column(Enum(ProjectType), nullable=False, index=True)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, index=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Metadata
    repository = Column(String(500), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    language = Column(String(50), nullable=True)
    difficulty = Column(Integer, nullable=True)

    # Stats
    members_count = Column(Integer, default=1)
    contributions_count = Column(Integer, default=0)
    stars_count = Column(Integer, default=0)

    # Rewards
    reward_pool = Column(DECIMAL(10, 2), nullable=True)
    reward_distribution = Column(Enum(RewardDistribution), default=RewardDistribution.CONTRIBUTION_BASED)

    # Settings
    is_public = Column(Boolean, default=True)
    allow_join = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deadline = Column(DateTime, nullable=True)

    # Relationships
    owner = relationship("User", back_populates="projects_owned")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    chat_messages = relationship("ProjectChatMessage", back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5"),
    )

    def __repr__(self):
        return f"<Project {self.name}>"


class ProjectMember(Base):
    """Project member model."""

    __tablename__ = "project_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Enum(MemberRole), default=MemberRole.MEMBER)

    # Stats
    contributions = Column(Integer, default=0)
    commits = Column(Integer, default=0)
    lines_added = Column(Integer, default=0)
    lines_removed = Column(Integer, default=0)

    # Rewards
    earned_rewards = Column(DECIMAL(10, 2), default=0.00)

    # Status
    is_active = Column(Boolean, default=True)
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])

    def __repr__(self):
        return f"<ProjectMember {self.user_id} in {self.project_id}>"


class ProjectChatMessage(Base):
    """Project chat message model."""

    __tablename__ = "project_chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)

    # Attachments
    attachments = Column(ARRAY(Text), nullable=True)

    # References
    reply_to = Column(UUID(as_uuid=True), ForeignKey("project_chat_messages.id", ondelete="SET NULL"), nullable=True)

    # Status
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="chat_messages")
    replied_message = relationship("ProjectChatMessage", remote_side=[id], backref="replies")

    def __repr__(self):
        return f"<ProjectChatMessage {self.id} in {self.project_id}>"
