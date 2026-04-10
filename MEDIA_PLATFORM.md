# Media Platform Documentation

## Overview

The J-MAXING Media Platform allows users to share and discover memes, dances, art, videos, music, and more with AI-powered automatic categorization and dynamic category creation from trending searches.

## Features

### 📤 Media Upload
- **Drag & Drop Upload**: Simple drag-and-drop interface for uploading media
- **Supported Formats**:
  - Images: JPG, PNG, GIF, WebP
  - Videos: MP4, WebM, MOV
  - Audio: MP3, WAV, OGG
- **File Preview**: See your media before uploading
- **Title & Description**: Add descriptive information
- **Tags**: Improve discoverability with relevant tags

### 🤖 AI-Powered Categorization

The platform uses machine learning to automatically categorize uploaded media:

```typescript
export interface AIAnalysis {
  detectedType: MediaType                // image, video, gif, audio
  suggestedCategory: MediaCategory       // memes, dances, art, etc.
  confidence: number                     // 0-1 confidence score
  detectedObjects: string[]              // Objects found in image/video
  detectedText?: string                  // OCR text detection
  isNSFW: boolean                        // Content safety check
  quality: number                        // Quality assessment (0-1)
  tags: string[]                         // Auto-generated tags
}
```

**How it works**:
1. User uploads media file
2. AI analyzes content (file type, visual elements, text)
3. AI suggests category with confidence score
4. User can accept or override AI suggestion
5. Media is published with final categorization

**Confidence Levels**:
- **High (85%+)**: AI is very confident - usually correct
- **Medium (65-85%)**: AI has moderate confidence - user should verify
- **Low (<65%)**: AI is uncertain - manual selection recommended

### 📁 12 Predefined Categories

Each category has unique branding and examples:

| Category | Icon | Color | Description | Examples |
|----------|------|-------|-------------|----------|
| **Memes** | 😂 | #39FF14 | Funny images, reaction pics, viral content | Drake meme, Distracted boyfriend |
| **Dances** | 💃 | #FF10F0 | Dance videos, choreography, TikTok trends | Renegade, Savage dance |
| **Art** | 🎨 | #00FFFF | Digital art, illustrations, paintings | Digital painting, Character design |
| **Photography** | 📸 | #0EA5E9 | Professional photos, landscapes, portraits | Nature photography, Street photography |
| **Animations** | 🎬 | #F59E0B | Animated content, GIFs, motion graphics | 2D animation, Stop motion |
| **Music** | 🎵 | #8B5CF6 | Songs, beats, covers, compositions | Original songs, Covers |
| **Videos** | 🎥 | #EF4444 | General video content, vlogs, short films | Vlogs, Documentaries |
| **Tutorials** | 📚 | #10B981 | How-to guides, educational content | Coding tutorials, Life hacks |
| **Gaming** | 🎮 | #6366F1 | Game clips, highlights, playthroughs | Gameplay, Highlights |
| **Comedy** | 🤣 | #EC4899 | Stand-up, sketches, funny videos | Stand-up clips, Skits |
| **Education** | 🎓 | #14B8A6 | Educational content, explanations | Science, History |
| **Other** | 📁 | #6B7280 | Content that doesn't fit other categories | Misc, Uncategorized |

### 🔥 Dynamic Category Creation

Categories can be automatically created based on popular search trends:

```typescript
export interface TrendingSearch {
  query: string                    // Search query text
  count: number                    // Number of times searched
  category?: MediaCategory         // Existing category (if applicable)
  shouldCreateCategory: boolean    // True if popular enough for new category
  timestamp: number                // When trend started
}
```

**How it works**:
1. Users search for content (e.g., "cosplay", "vtubers", "3d printing")
2. System tracks search frequency
3. When threshold reached (e.g., 100+ searches), backend auto-creates new category
4. New category appears in browse filters
5. Users can now post directly to new category

**Example**: If 200 users search for "cosplay" and no cosplay category exists, the system automatically creates a "Cosplay" category with:
- Auto-generated icon and color
- Description based on search patterns
- Populated with retroactively categorized content

### 📊 Media Feed & Discovery

**Browse Options**:
- **All Media**: View everything
- **Category Filters**: Browse specific categories
- **Type Filters**: Filter by image, video, GIF, or audio
- **Search**: Find media by title, description, or tags
- **Trending Section**: See what's popular right now

**Trending Algorithm**:
- Views × Recency weight
- Likes × Engagement weight
- Shares × Viral weight
- Time decay factor

### 💬 Engagement Features

**Like System**:
```typescript
likeMedia: async (mediaId: string) => {
  const response = await apiClient.post(`/media/${mediaId}/like`)
  return response.data
}
```

**Download Tracking**:
```typescript
downloadMedia: async (mediaId: string) => {
  const response = await apiClient.get(`/media/${mediaId}/download`, {
    responseType: 'blob',
  })
  return response.data
}
```

**Share Tracking**:
```typescript
shareMedia: async (mediaId: string) => {
  const response = await apiClient.post(`/media/${mediaId}/share`)
  return response.data
}
```

**Comments**:
```typescript
export interface MediaComment {
  id: string
  mediaId: string
  userId: string
  username: string
  avatar?: string
  content: string
  likes: number
  isLiked?: boolean
  timestamp: number
}
```

### 📈 Statistics & Analytics

Each media item tracks:
- **Views**: Number of times viewed
- **Likes**: Total likes received
- **Downloads**: Number of downloads
- **Shares**: Times shared externally
- **Comments**: Total comment count

User stats:
```typescript
export interface MediaStats {
  totalViews: number
  totalLikes: number
  totalShares: number
  totalDownloads: number
  topCategories: {
    category: MediaCategory
    count: number
    percentage: number
  }[]
  recentTrends: {
    date: string
    views: number
    likes: number
  }[]
}
```

## Page Structure

### 1. Media Page (`/media`)

**Features**:
- Category tabs with emoji icons
- Search bar with trending search suggestions
- Type filters (All, Images, Videos, GIFs, Audio)
- Trending section (top 5 items)
- Grid layout of media cards
- Category info panel

**Media Card Display**:
- Media preview (thumbnail or full image)
- AI categorization badge (if confidence > 90%)
- Category badge with color
- Title and description
- Tags (first 3)
- Stats (views, likes, downloads)
- Creator avatar and username

### 2. Upload Page (`/media/upload`)

**Upload Flow**:
1. **File Selection**: Drag & drop or click to browse
2. **AI Analysis**:
   - Shows loading state during analysis
   - Displays suggested category with confidence
   - Shows suggested tags
   - Indicates confidence level (high/medium/low)
3. **Media Details**:
   - Title input (required)
   - Description textarea (optional)
   - Tags input (comma-separated)
   - AI tag suggestions shown below
4. **Category Selection**:
   - Grid of all categories
   - AI suggestion highlighted with blue ring
   - User can override AI choice
5. **Submit**: Upload and redirect to media detail page

**AI Suggestion Display**:
```typescript
<div className="glass-panel p-6 border-l-4 border-neon-green">
  <Sparkles /> AI Analysis Complete

  Suggested: {category.name} - {confidence}%
  Confidence Bar: [████████░░] 87%

  Suggested Tags: #funny #viral #trending

  ✓ High confidence - This looks correct!
</div>
```

### 3. Media Detail Page (`/media/:id`)

**Layout**:
- **Left Column (2/3 width)**:
  - Full media display (image/video/audio player)
  - AI categorization badge overlay
  - Action bar (like, download, share, comment count)
  - Owner actions (edit, delete) if applicable
  - Media info (title, description, tags, category)
  - Creator profile link
  - Comments section with reply functionality

- **Right Sidebar (1/3 width)**:
  - Stats panel (views, likes, downloads, shares)
  - AI Analysis panel (suggested category, confidence)
  - Related media from same category

**Comment System**:
- Login required to comment
- Avatar + username + timestamp
- Like individual comments
- Real-time updates

**Share Options**:
- Native Web Share API (if supported)
- Copy link to clipboard
- Tracks share count

## API Reference

### Media Endpoints

```typescript
// Browse feed
getFeed(filters?: {
  category?: MediaCategory
  type?: MediaType
  tags?: string[]
  search?: string
  trending?: boolean
  limit?: number
  offset?: number
}): Promise<MediaItem[]>

// Get single media
getMedia(mediaId: string): Promise<MediaItem>

// Upload media (triggers AI analysis)
uploadMedia(upload: MediaUpload): Promise<MediaItem>

// Manual AI re-analysis
analyzeMedia(mediaId: string): Promise<AIAnalysis>

// Update category (override AI)
updateCategory(mediaId: string, category: MediaCategory): Promise<void>

// Delete media
deleteMedia(mediaId: string): Promise<void>

// Interactions
likeMedia(mediaId: string): Promise<void>
unlikeMedia(mediaId: string): Promise<void>
shareMedia(mediaId: string): Promise<void>
downloadMedia(mediaId: string): Promise<Blob>

// Comments
getComments(mediaId: string): Promise<MediaComment[]>
addComment(mediaId: string, content: string): Promise<MediaComment>
deleteComment(mediaId: string, commentId: string): Promise<void>

// Categories
getCategories(): Promise<MediaCategory[]>
followCategory(categoryId: string): Promise<void>
unfollowCategory(categoryId: string): Promise<void>

// Trending & Search
getTrending(limit?: number): Promise<MediaItem[]>
getTrendingSearches(): Promise<TrendingSearch[]>
searchMedia(query: string, limit?: number): Promise<MediaItem[]>

// User media
getMyMedia(): Promise<MediaItem[]>

// Stats
getStats(mediaId: string): Promise<MediaStats>

// Moderation
reportMedia(mediaId: string, reason: string): Promise<void>
```

## AI Categorization Details

### Training Data

The AI model is trained on:
- Visual features (colors, shapes, composition)
- Object detection (people, text, logos)
- Text extraction (OCR for memes with text)
- Audio analysis (music genre, speech detection)
- File metadata (file type, dimensions, duration)
- User behavior patterns

### Confidence Calculation

```typescript
const calculateConfidence = (features: AIFeatures): number => {
  const weights = {
    visualMatch: 0.4,      // Visual similarity to category examples
    objectDetection: 0.3,   // Detected objects match category
    textAnalysis: 0.2,      // OCR text matches category keywords
    metadataMatch: 0.1,     // File properties match category
  }

  return (
    features.visualMatch * weights.visualMatch +
    features.objectDetection * weights.objectDetection +
    features.textAnalysis * weights.textAnalysis +
    features.metadataMatch * weights.metadataMatch
  )
}
```

### User Feedback Loop

When users override AI suggestions:
1. System logs: `{ mediaId, aiSuggestion, userChoice, timestamp }`
2. AI model retrains on corrections
3. Future predictions improve accuracy
4. Confidence scores adjust based on historical accuracy

## Trending Search Thresholds

```typescript
const CATEGORY_CREATION_THRESHOLDS = {
  searchCount: 100,        // Minimum unique searches
  uniqueUsers: 50,         // Minimum unique searchers
  timeWindow: 604800000,   // Within 7 days (ms)
  resultGap: 20,           // % of searches with <5 results
}

const shouldCreateCategory = (trend: TrendingSearch): boolean => {
  return (
    trend.count >= CATEGORY_CREATION_THRESHOLDS.searchCount &&
    trend.uniqueUsers >= CATEGORY_CREATION_THRESHOLDS.uniqueUsers &&
    Date.now() - trend.timestamp <= CATEGORY_CREATION_THRESHOLDS.timeWindow &&
    trend.resultGapPercentage >= CATEGORY_CREATION_THRESHOLDS.resultGap
  )
}
```

## Content Moderation

### Automated Checks

```typescript
export interface AIAnalysis {
  isNSFW: boolean          // NSFW content detection
  quality: number          // Quality score (0-1)
  // ... other fields
}
```

**NSFW Detection**:
- Flags inappropriate content
- Requires manual review before publishing
- Can be appealed by user

**Quality Assessment**:
- Low resolution → warning
- Blurry/corrupted → rejected
- Duplicate detection → prevents spam

### User Reporting

```typescript
reportMedia: async (mediaId: string, reason: string) => {
  const response = await apiClient.post(`/media/${mediaId}/report`, {
    reason,
  })
  return response.data
}
```

Report categories:
- Spam
- Inappropriate content
- Copyright violation
- Incorrect category
- Low quality

## Search & Discovery

### Search Algorithm

```typescript
const searchMedia = (query: string): MediaItem[] => {
  const results = []

  // 1. Exact title matches (highest priority)
  results.push(...exactTitleMatches(query))

  // 2. Partial title matches
  results.push(...partialTitleMatches(query))

  // 3. Tag matches
  results.push(...tagMatches(query))

  // 4. Description matches
  results.push(...descriptionMatches(query))

  // 5. AI-detected content matches
  results.push(...aiContentMatches(query))

  // Sort by relevance + engagement
  return sortByRelevance(results, query)
}
```

### Trending Searches Display

Shows popular searches with indicators:
- 🔥 emoji for searches about to become categories
- Search count badge
- Click to auto-fill search

## Performance Optimizations

### Image Optimization

```typescript
export interface MediaItem {
  url: string              // Full resolution
  thumbnailUrl?: string    // Optimized thumbnail
  // ...
}
```

- Thumbnails generated on upload
- Lazy loading for images
- Progressive JPEG/WebP formats
- CDN caching

### Pagination

```typescript
getFeed(filters?: {
  limit?: number       // Default: 50
  offset?: number      // Default: 0
})
```

- Infinite scroll on mobile
- "Load More" button on desktop
- Prefetch next page on scroll

## Mobile Responsiveness

- **Grid Layout**: 2 columns on mobile, 3 on tablet, 4 on desktop
- **Touch Optimized**: Large tap targets, swipe gestures
- **Media Player**: Native controls for video/audio
- **Upload**: Mobile camera access for photo/video capture

## Success Metrics

### Platform KPIs
- Total media uploads
- Daily active uploaders
- Average uploads per user
- Category distribution
- AI accuracy rate
- User override rate

### Engagement Metrics
- Views per media item
- Like rate
- Download rate
- Share rate
- Comment rate
- Time on platform

### AI Performance
- Categorization accuracy (%)
- Average confidence score
- User override frequency
- Model retraining frequency

## Future Enhancements

### Planned Features
1. **Collections**: Users can create curated collections
2. **Playlists**: Audio/video playlists
3. **Remix/Edit**: Built-in editing tools
4. **Collaborations**: Co-author media items
5. **NFT Integration**: Mint media as NFTs
6. **Copyright Detection**: Prevent unauthorized uploads
7. **Multi-language**: Support for non-English content
8. **Advanced Search**: Reverse image search, similar media
9. **Recommendations**: Personalized feed based on interests
10. **Live Streaming**: Live video/audio broadcasts

### AI Improvements
- Better object detection
- Emotion/sentiment analysis for memes
- Music genre classification
- Video scene detection
- Automatic highlight clips
- Style transfer and filters

## Integration with J-MAXING

### Earning JC
- **Upload Bonus**: 10 JC per upload
- **Engagement Rewards**:
  - 1 JC per 100 views
  - 5 JC per 10 likes
  - 10 JC per download
  - 15 JC per share
- **Trending Bonus**: 100 JC if media reaches trending
- **Quality Bonus**: 50 JC for high-quality uploads (AI score > 0.9)

### Spending JC
- **Promote Media**: Spend JC to boost visibility
- **Featured Placement**: Pay to be featured on homepage
- **Custom Categories**: Create personal categories (1000 JC)
- **Remove Ads**: Ad-free experience (500 JC/month)

### Social Integration
- Media appears in social feed
- Can be shared to user profiles
- Comments integrate with social comments system
- Followers get notified of new uploads

## Technical Stack

- **Frontend**: React 18 + TypeScript
- **State Management**: React Query (server state) + Zustand (client state)
- **Styling**: TailwindCSS + Framer Motion
- **File Upload**: FormData API with drag & drop
- **Media Display**: Native HTML5 video/audio players
- **AI Analysis**: Backend ML models (Python/TensorFlow)
- **Storage**: Object storage (S3/CloudFlare R2) for media files
- **CDN**: CloudFlare for global media delivery
- **Database**: Media metadata, categories, comments, stats

## Conclusion

The J-MAXING Media Platform combines powerful AI categorization with community-driven discovery to create a seamless media sharing experience. Users can upload any media, let AI handle categorization, and discover trending content - all while earning JC rewards for quality contributions.
