"""Project collaboration routes."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from core.database import get_db
from core.security import get_current_user
from core.exceptions import NotFoundException, ForbiddenException, ConflictException
from models.user import User
from models.project import (
    Project,
    ProjectMember,
    ProjectChatMessage,
    ProjectType,
    ProjectStatus,
    MemberRole,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
async def list_projects(
    type: Optional[str] = None,
    status: Optional[str] = "active",
    limit: int = Query(20, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List projects."""
    query = db.query(Project)

    if type:
        try:
            project_type = ProjectType(type)
            query = query.filter(Project.type == project_type)
        except ValueError:
            pass

    if status:
        try:
            project_status = ProjectStatus(status)
            query = query.filter(Project.status == project_status)
        except ValueError:
            pass

    # Only show public projects or projects user is member of
    member_projects = db.query(ProjectMember.project_id).filter(
        ProjectMember.user_id == current_user.id
    ).subquery()

    query = query.filter(
        or_(
            Project.is_public == True,
            Project.id.in_(member_projects),
            Project.owner_id == current_user.id
        )
    )

    projects = query.order_by(desc(Project.created_at)).limit(limit).offset(offset).all()

    # Format response
    result = []
    for project in projects:
        # Check if user is member
        member = db.query(ProjectMember).filter(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == current_user.id
        ).first()

        result.append({
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
            "type": project.type.value,
            "status": project.status.value,
            "ownerId": str(project.owner_id),
            "ownerUsername": project.owner.username,
            "repository": project.repository,
            "tags": project.tags or [],
            "language": project.language,
            "difficulty": project.difficulty,
            "members": project.members_count,
            "contributions": project.contributions_count,
            "stars": project.stars_count,
            "rewardPool": float(project.reward_pool) if project.reward_pool else None,
            "rewardDistribution": project.reward_distribution.value if project.reward_distribution else None,
            "isPublic": project.is_public,
            "allowJoin": project.allow_join,
            "createdAt": int(project.created_at.timestamp() * 1000),
            "updatedAt": int(project.updated_at.timestamp() * 1000),
            "deadline": int(project.deadline.timestamp() * 1000) if project.deadline else None,
            "isMember": member is not None,
            "userRole": member.role.value if member else None,
        })

    return result


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project."""
    # Validate project type
    try:
        project_type = ProjectType(project_data["type"])
    except (ValueError, KeyError):
        raise HTTPException(status_code=400, detail="Invalid project type")

    new_project = Project(
        name=project_data["name"],
        description=project_data["description"],
        type=project_type,
        owner_id=current_user.id,
        repository=project_data.get("repository"),
        tags=project_data.get("tags", []),
        language=project_data.get("language"),
        difficulty=project_data.get("difficulty"),
        reward_pool=project_data.get("rewardPool"),
        is_public=project_data.get("isPublic", True),
        allow_join=project_data.get("allowJoin", True),
    )

    db.add(new_project)
    db.flush()

    # Add owner as member
    owner_member = ProjectMember(
        project_id=new_project.id,
        user_id=current_user.id,
        role=MemberRole.OWNER,
    )
    db.add(owner_member)

    db.commit()
    db.refresh(new_project)

    return {
        "id": str(new_project.id),
        "name": new_project.name,
        "description": new_project.description,
        "type": new_project.type.value,
        "status": new_project.status.value,
        "ownerId": str(new_project.owner_id),
        "ownerUsername": current_user.username,
        "members": 1,
        "contributions": 0,
        "stars": 0,
        "isPublic": new_project.is_public,
        "allowJoin": new_project.allow_join,
        "createdAt": int(new_project.created_at.timestamp() * 1000),
        "isMember": True,
        "userRole": "owner",
    }


@router.get("/{project_id}")
async def get_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get project details."""
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise NotFoundException("Project not found")

    # Check access
    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == current_user.id
    ).first()

    if not project.is_public and not member and project.owner_id != current_user.id:
        raise ForbiddenException("Access denied")

    return {
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
        "type": project.type.value,
        "status": project.status.value,
        "ownerId": str(project.owner_id),
        "ownerUsername": project.owner.username,
        "repository": project.repository,
        "tags": project.tags or [],
        "language": project.language,
        "difficulty": project.difficulty,
        "members": project.members_count,
        "contributions": project.contributions_count,
        "stars": project.stars_count,
        "rewardPool": float(project.reward_pool) if project.reward_pool else None,
        "rewardDistribution": project.reward_distribution.value if project.reward_distribution else None,
        "isPublic": project.is_public,
        "allowJoin": project.allow_join,
        "createdAt": int(project.created_at.timestamp() * 1000),
        "updatedAt": int(project.updated_at.timestamp() * 1000),
        "deadline": int(project.deadline.timestamp() * 1000) if project.deadline else None,
        "isMember": member is not None,
        "userRole": member.role.value if member else None,
    }


@router.post("/{project_id}/join", status_code=status.HTTP_204_NO_CONTENT)
async def join_project(
    project_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Join a project."""
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise NotFoundException("Project not found")

    if not project.allow_join:
        raise ForbiddenException("Project does not allow joining")

    # Check if already member
    existing_member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == current_user.id
    ).first()

    if existing_member:
        raise ConflictException("Already a member of this project")

    # Add as member
    new_member = ProjectMember(
        project_id=project_id,
        user_id=current_user.id,
        role=MemberRole.MEMBER,
    )
    db.add(new_member)

    # Update member count
    project.members_count += 1

    db.commit()

    return None


@router.get("/{project_id}/members")
async def get_project_members(
    project_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get project members."""
    # Check if user has access to project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise NotFoundException("Project not found")

    members = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.is_active == True
    ).limit(limit).offset(offset).all()

    result = []
    for member in members:
        user = member.user
        result.append({
            "id": str(member.id),
            "userId": str(user.id),
            "username": user.username,
            "avatar": user.avatar_url,
            "role": member.role.value,
            "contributions": member.contributions,
            "commits": member.commits,
            "earnedRewards": float(member.earned_rewards),
            "joinedAt": int(member.joined_at.timestamp() * 1000),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Project Chat
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{project_id}/chat")
async def get_chat_messages(
    project_id: UUID,
    limit: int = Query(50, le=100),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get project chat messages."""
    # Check if user is member
    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == current_user.id,
        ProjectMember.is_active == True
    ).first()

    if not member:
        raise ForbiddenException("Must be a project member to view chat")

    messages = db.query(ProjectChatMessage).filter(
        ProjectChatMessage.project_id == project_id,
        ProjectChatMessage.is_deleted == False
    ).order_by(ProjectChatMessage.created_at).limit(limit).offset(offset).all()

    result = []
    for msg in messages:
        user = db.query(User).filter(User.id == msg.user_id).first()
        result.append({
            "id": str(msg.id),
            "projectId": str(msg.project_id),
            "userId": str(msg.user_id),
            "username": user.username if user else "Unknown",
            "avatar": user.avatar_url if user else None,
            "content": msg.content,
            "attachments": msg.attachments or [],
            "replyTo": str(msg.reply_to) if msg.reply_to else None,
            "isEdited": msg.is_edited,
            "timestamp": int(msg.created_at.timestamp() * 1000),
        })

    return result


@router.post("/{project_id}/chat", status_code=status.HTTP_201_CREATED)
async def send_chat_message(
    project_id: UUID,
    message_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a chat message."""
    # Check if user is member
    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == current_user.id,
        ProjectMember.is_active == True
    ).first()

    if not member:
        raise ForbiddenException("Must be a project member to send messages")

    new_message = ProjectChatMessage(
        project_id=project_id,
        user_id=current_user.id,
        content=message_data["content"],
        attachments=message_data.get("attachments", []),
        reply_to=message_data.get("replyTo"),
    )

    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    return {
        "id": str(new_message.id),
        "projectId": str(new_message.project_id),
        "userId": str(new_message.user_id),
        "username": current_user.username,
        "avatar": current_user.avatar_url,
        "content": new_message.content,
        "attachments": new_message.attachments or [],
        "replyTo": str(new_message.reply_to) if new_message.reply_to else None,
        "isEdited": False,
        "timestamp": int(new_message.created_at.timestamp() * 1000),
    }
