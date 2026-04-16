-- J-MAXING Database Schema
-- PostgreSQL 14+
-- Complete schema for code marketplace, social features, projects, and media

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- For password hashing

-- ============================================================================
-- USERS & AUTHENTICATION
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,

    -- Profile
    avatar_url TEXT,
    bio TEXT,
    location VARCHAR(100),
    website VARCHAR(255),
    github VARCHAR(100),
    twitter VARCHAR(100),

    -- Stats
    balance DECIMAL(10, 2) DEFAULT 0.00,
    reputation INTEGER DEFAULT 0,
    rank INTEGER DEFAULT 1,
    tasks_completed INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,

    -- Settings
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    email_verified BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_reputation ON users(reputation DESC);

CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sessions_token ON user_sessions(token);
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);

-- ============================================================================
-- SOCIAL FEATURES
-- ============================================================================

CREATE TABLE follows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    follower_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    following_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(follower_id, following_id),
    CHECK (follower_id != following_id)
);

CREATE INDEX idx_follows_follower ON follows(follower_id);
CREATE INDEX idx_follows_following ON follows(following_id);

CREATE TYPE post_type AS ENUM ('submission', 'job', 'achievement', 'status', 'showcase');

CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type post_type NOT NULL,
    content TEXT NOT NULL,

    -- Code content (for showcases)
    code TEXT,
    language VARCHAR(50),

    -- References
    job_id UUID,
    submission_id UUID,

    -- Job/submission details
    score DECIMAL(5, 2),
    payout DECIMAL(10, 2),

    -- Media
    images TEXT[],  -- Array of image URLs
    tags TEXT[],

    -- Engagement
    likes_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    views_count INTEGER DEFAULT 0,

    -- Status
    is_pinned BOOLEAN DEFAULT FALSE,
    is_deleted BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_type ON posts(type);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_posts_tags ON posts USING GIN(tags);

CREATE TABLE post_likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(post_id, user_id)
);

CREATE INDEX idx_post_likes_post ON post_likes(post_id);
CREATE INDEX idx_post_likes_user ON post_likes(user_id);

CREATE TABLE comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES comments(id) ON DELETE CASCADE,  -- For nested comments
    content TEXT NOT NULL,
    likes_count INTEGER DEFAULT 0,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_comments_post ON comments(post_id);
CREATE INDEX idx_comments_user ON comments(user_id);
CREATE INDEX idx_comments_parent ON comments(parent_id);

-- ============================================================================
-- REFERRAL SYSTEM
-- ============================================================================

CREATE TABLE referrals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    referrer_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    referred_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code VARCHAR(20) UNIQUE NOT NULL,

    -- Rewards
    referrer_reward DECIMAL(10, 2) DEFAULT 100.00,  -- 100 JC
    referred_reward DECIMAL(10, 2) DEFAULT 50.00,   -- 50 JC
    is_rewarded BOOLEAN DEFAULT FALSE,
    rewarded_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(referrer_id, referred_id)
);

CREATE INDEX idx_referrals_code ON referrals(code);
CREATE INDEX idx_referrals_referrer ON referrals(referrer_id);

-- ============================================================================
-- PROJECTS & COLLABORATION
-- ============================================================================

CREATE TYPE project_type AS ENUM ('community', 'group', 'solo');
CREATE TYPE project_status AS ENUM ('active', 'completed', 'archived');
CREATE TYPE member_role AS ENUM ('owner', 'admin', 'member', 'viewer');
CREATE TYPE reward_distribution AS ENUM ('equal', 'contribution-based', 'merit-based');

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    type project_type NOT NULL,
    status project_status DEFAULT 'active',
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Metadata
    repository VARCHAR(500),  -- GitHub repo URL
    tags TEXT[],
    language VARCHAR(50),
    difficulty INTEGER CHECK (difficulty BETWEEN 1 AND 5),

    -- Stats
    members_count INTEGER DEFAULT 1,
    contributions_count INTEGER DEFAULT 0,
    stars_count INTEGER DEFAULT 0,

    -- Rewards (for community projects)
    reward_pool DECIMAL(10, 2),
    reward_distribution reward_distribution DEFAULT 'contribution-based',

    -- Settings
    is_public BOOLEAN DEFAULT TRUE,
    allow_join BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deadline TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_projects_type ON projects(type);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_owner ON projects(owner_id);
CREATE INDEX idx_projects_tags ON projects USING GIN(tags);
CREATE INDEX idx_projects_created_at ON projects(created_at DESC);

CREATE TABLE project_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role member_role DEFAULT 'member',

    -- Stats
    contributions INTEGER DEFAULT 0,
    commits INTEGER DEFAULT 0,
    lines_added INTEGER DEFAULT 0,
    lines_removed INTEGER DEFAULT 0,

    -- Rewards
    earned_rewards DECIMAL(10, 2) DEFAULT 0.00,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    invited_by UUID REFERENCES users(id),

    -- Timestamps
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(project_id, user_id)
);

CREATE INDEX idx_project_members_project ON project_members(project_id);
CREATE INDEX idx_project_members_user ON project_members(user_id);

CREATE TABLE project_chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,

    -- Attachments
    attachments TEXT[],  -- URLs to files

    -- References
    reply_to UUID REFERENCES project_chat_messages(id) ON DELETE SET NULL,

    -- Status
    is_edited BOOLEAN DEFAULT FALSE,
    is_deleted BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_project ON project_chat_messages(project_id);
CREATE INDEX idx_chat_messages_created_at ON project_chat_messages(created_at DESC);

-- ============================================================================
-- MEDIA PLATFORM
-- ============================================================================

CREATE TYPE media_type AS ENUM ('image', 'video', 'gif', 'audio');

CREATE TABLE media_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    color VARCHAR(7),  -- Hex color

    -- Stats
    item_count INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0,
    trending BOOLEAN DEFAULT FALSE,

    -- Auto-creation
    is_auto_created BOOLEAN DEFAULT FALSE,
    created_from_searches INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_media_categories_slug ON media_categories(slug);
CREATE INDEX idx_media_categories_trending ON media_categories(trending);

-- Insert default categories
INSERT INTO media_categories (name, slug, description, icon, color) VALUES
('Memes', 'memes', 'Funny images and memes', '😂', '#FF6B6B'),
('Dances', 'dances', 'Dance videos and choreography', '💃', '#4ECDC4'),
('Art', 'art', 'Digital and traditional art', '🎨', '#95E1D3'),
('Photography', 'photography', 'Photos and photography', '📸', '#F38181'),
('Animations', 'animations', 'Animated content and GIFs', '🎬', '#AA96DA'),
('Music', 'music', 'Music and audio tracks', '🎵', '#FCBAD3'),
('Videos', 'videos', 'General video content', '🎥', '#FFFFD2'),
('Tutorials', 'tutorials', 'Educational tutorials', '📚', '#A8D8EA'),
('Gaming', 'gaming', 'Gaming content and clips', '🎮', '#FFB6B9'),
('Comedy', 'comedy', 'Comedy skits and jokes', '🤣', '#FEC8D8'),
('Education', 'education', 'Educational content', '🎓', '#FFDCB4'),
('Other', 'other', 'Uncategorized content', '📁', '#E2E2E2');

CREATE TABLE media_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Content
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    type media_type NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,

    -- Categorization
    category_id UUID NOT NULL REFERENCES media_categories(id),
    suggested_category_id UUID REFERENCES media_categories(id),
    ai_confidence DECIMAL(3, 2),  -- 0.00 to 1.00
    tags TEXT[],

    -- Metadata
    width INTEGER,
    height INTEGER,
    duration INTEGER,  -- seconds
    file_size BIGINT,
    mime_type VARCHAR(100),

    -- Engagement
    views_count INTEGER DEFAULT 0,
    likes_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    downloads_count INTEGER DEFAULT 0,

    -- Status
    is_processing BOOLEAN DEFAULT FALSE,
    is_flagged BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_media_user ON media_items(user_id);
CREATE INDEX idx_media_category ON media_items(category_id);
CREATE INDEX idx_media_type ON media_items(type);
CREATE INDEX idx_media_tags ON media_items USING GIN(tags);
CREATE INDEX idx_media_created_at ON media_items(created_at DESC);
CREATE INDEX idx_media_views ON media_items(views_count DESC);

CREATE TABLE media_likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_id UUID NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(media_id, user_id)
);

CREATE INDEX idx_media_likes_media ON media_likes(media_id);
CREATE INDEX idx_media_likes_user ON media_likes(user_id);

CREATE TABLE media_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_id UUID NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES media_comments(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    likes_count INTEGER DEFAULT 0,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_media_comments_media ON media_comments(media_id);
CREATE INDEX idx_media_comments_user ON media_comments(user_id);

-- ============================================================================
-- SEARCH & TRENDS
-- ============================================================================

CREATE TABLE search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    category VARCHAR(50),  -- 'media', 'projects', 'users', etc.
    results_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_search_queries_created_at ON search_queries(created_at DESC);
CREATE INDEX idx_search_queries_category ON search_queries(category);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_posts_updated_at BEFORE UPDATE ON posts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_media_items_updated_at BEFORE UPDATE ON media_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update follower/following counts
CREATE OR REPLACE FUNCTION update_follow_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE users SET following_count = following_count + 1 WHERE id = NEW.follower_id;
        UPDATE users SET followers_count = followers_count + 1 WHERE id = NEW.following_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE users SET following_count = following_count - 1 WHERE id = OLD.follower_id;
        UPDATE users SET followers_count = followers_count - 1 WHERE id = OLD.following_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER follow_counts_trigger AFTER INSERT OR DELETE ON follows
    FOR EACH ROW EXECUTE FUNCTION update_follow_counts();

-- Update post likes count
CREATE OR REPLACE FUNCTION update_post_likes_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET likes_count = likes_count + 1 WHERE id = NEW.post_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET likes_count = likes_count - 1 WHERE id = OLD.post_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER post_likes_count_trigger AFTER INSERT OR DELETE ON post_likes
    FOR EACH ROW EXECUTE FUNCTION update_post_likes_count();

-- Update media likes count
CREATE OR REPLACE FUNCTION update_media_likes_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE media_items SET likes_count = likes_count + 1 WHERE id = NEW.media_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE media_items SET likes_count = likes_count - 1 WHERE id = OLD.media_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER media_likes_count_trigger AFTER INSERT OR DELETE ON media_likes
    FOR EACH ROW EXECUTE FUNCTION update_media_likes_count();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- User profile with stats
CREATE VIEW user_profiles AS
SELECT
    u.*,
    COUNT(DISTINCT p.id) as posts_count,
    COUNT(DISTINCT pm.id) as projects_count
FROM users u
LEFT JOIN posts p ON p.user_id = u.id AND p.is_deleted = FALSE
LEFT JOIN project_members pm ON pm.user_id = u.id AND pm.is_active = TRUE
GROUP BY u.id;

-- Trending media items (last 7 days)
CREATE VIEW trending_media AS
SELECT
    mi.*,
    (mi.views_count * 1.0 + mi.likes_count * 2.0 + mi.shares_count * 3.0) as trend_score
FROM media_items mi
WHERE mi.created_at > NOW() - INTERVAL '7 days'
    AND mi.is_approved = TRUE
ORDER BY trend_score DESC;

-- ============================================================================
-- INDEXES FOR FULL TEXT SEARCH
-- ============================================================================

CREATE INDEX idx_posts_content_trgm ON posts USING GIN (content gin_trgm_ops);
CREATE INDEX idx_projects_name_trgm ON projects USING GIN (name gin_trgm_ops);
CREATE INDEX idx_media_title_trgm ON media_items USING GIN (title gin_trgm_ops);
