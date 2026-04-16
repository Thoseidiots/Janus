# J-MAXING Backend

FastAPI-based backend for the J-MAXING code marketplace with social features, collaborative projects, and media sharing.

## Features

- 🔐 **JWT Authentication** - Secure user authentication with access and refresh tokens
- 👥 **Social Features** - Follow users, create posts, like and comment
- 🤝 **Project Collaboration** - Community, group, and solo projects with real-time chat
- 📸 **Media Platform** - Share images, videos, GIFs, and audio with AI categorization
- 💰 **Referral System** - Earn Janus Credits (JC) by referring friends
- 🗄️ **PostgreSQL Database** - Robust relational database with full schema
- 📤 **File Upload** - Support for images, videos, and audio files
- 🔍 **Full-Text Search** - Search across projects, media, and users

## Tech Stack

- **FastAPI** 0.104+ - Modern async Python web framework
- **PostgreSQL** 14+ - Primary database
- **SQLAlchemy** 2.0+ - ORM and database toolkit
- **Pydantic** 2.0+ - Data validation using Python type hints
- **Python-JOSE** - JWT token generation and validation
- **Passlib** - Password hashing with bcrypt

## Project Structure

```
jmaxing_backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This file
│
├── api/                   # API route handlers
│   └── routes/
│       ├── auth.py       # Authentication endpoints
│       ├── users.py      # User management
│       ├── social.py     # Social features (posts, comments, follows)
│       ├── projects.py   # Project collaboration
│       └── media.py      # Media platform
│
├── core/                  # Core functionality
│   ├── config.py         # Configuration management
│   ├── database.py       # Database connection
│   ├── security.py       # JWT and password hashing
│   └── exceptions.py     # Custom exceptions
│
├── models/                # SQLAlchemy models
│   ├── user.py
│   ├── post.py
│   ├── project.py
│   └── media.py
│
├── schemas/               # Pydantic schemas (request/response)
│   ├── user.py
│   ├── social.py
│   ├── project.py
│   └── media.py
│
├── services/              # Business logic
│   ├── auth_service.py
│   ├── user_service.py
│   ├── social_service.py
│   ├── project_service.py
│   └── media_service.py
│
├── database/              # Database files
│   └── schema.sql        # Complete PostgreSQL schema
│
└── tests/                 # Test suite
    ├── test_auth.py
    ├── test_users.py
    ├── test_social.py
    ├── test_projects.py
    └── test_media.py
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- pip or conda

### Installation

1. **Clone and navigate to backend directory:**
```bash
cd jmaxing_backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up database:**
```bash
# Create PostgreSQL database
createdb jmaxing

# Run schema
psql -d jmaxing -f database/schema.sql
```

5. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Run the server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/api/docs`

## Environment Variables

Create a `.env` file with these variables:

```env
# Application
APP_NAME=J-MAXING
ENVIRONMENT=development
DEBUG=True

# Database
DATABASE_URL=postgresql://jmaxing:password@localhost:5432/jmaxing

# Security
SECRET_KEY=your-secret-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=["http://localhost:5173"]

# File Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=104857600  # 100MB
STORAGE_PROVIDER=local  # local, s3, cloudinary

# AI (Optional)
AI_CATEGORIZATION_ENABLED=False
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get tokens
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Logout (invalidate token)

### Users
- `GET /api/users/me` - Get current user profile
- `PUT /api/users/me` - Update current user profile
- `GET /api/users/{username}` - Get user by username
- `GET /api/users/{user_id}/posts` - Get user's posts
- `GET /api/users/{user_id}/projects` - Get user's projects

### Social
- `GET /api/social/feed` - Get personalized feed
- `POST /api/social/posts` - Create a post
- `GET /api/social/posts/{post_id}` - Get post details
- `POST /api/social/posts/{post_id}/like` - Like a post
- `POST /api/social/posts/{post_id}/comment` - Comment on post
- `POST /api/social/users/{user_id}/follow` - Follow a user
- `DELETE /api/social/users/{user_id}/follow` - Unfollow a user
- `GET /api/social/referrals` - Get referral stats

### Projects
- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/projects/{project_id}` - Get project details
- `PUT /api/projects/{project_id}` - Update project
- `POST /api/projects/{project_id}/join` - Join project
- `GET /api/projects/{project_id}/chat` - Get chat messages
- `POST /api/projects/{project_id}/chat` - Send chat message

### Media
- `GET /api/media` - List media items
- `POST /api/media/upload` - Upload media
- `GET /api/media/{media_id}` - Get media details
- `POST /api/media/{media_id}/like` - Like media
- `GET /api/media/categories` - List categories
- `GET /api/media/trending` - Get trending media

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8 .
```

### Type Checking
```bash
mypy .
```

## Database Migrations

Using Alembic for database migrations:

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn + Uvicorn

```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Settings

For production, update `.env`:

```env
ENVIRONMENT=production
DEBUG=False
DATABASE_URL=postgresql://user:pass@prod-db:5432/jmaxing
SECRET_KEY=<strong-random-key>
CORS_ORIGINS=["https://yourdomain.com"]
```

## Performance

- Connection pooling configured (20 connections + 10 overflow)
- GZip compression for responses >1KB
- Request timing middleware for monitoring
- Database indexes on frequently queried columns
- Full-text search with trigram indexes

## Security

- Passwords hashed with bcrypt
- JWT tokens with expiration
- CORS configured for frontend origins
- SQL injection protection via SQLAlchemy ORM
- Input validation with Pydantic
- Rate limiting (optional, via middleware)

## Monitoring

Health check endpoint:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/Thoseidiots/Janus/issues
- Documentation: http://localhost:8000/api/docs

## License

Research & Development

---

**Status**: Backend foundation complete, ready for frontend integration
**Maintained by**: Janus Team
