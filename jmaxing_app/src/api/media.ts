import { apiClient } from './client'

// Media Types
export type MediaType = 'image' | 'video' | 'gif' | 'audio'
export type MediaCategory =
  | 'memes'
  | 'dances'
  | 'art'
  | 'photography'
  | 'animations'
  | 'music'
  | 'videos'
  | 'tutorials'
  | 'gaming'
  | 'comedy'
  | 'education'
  | 'other'

export interface MediaItem {
  id: string
  userId: string
  username: string
  avatar?: string

  // Content
  url: string
  thumbnailUrl?: string
  type: MediaType
  title: string
  description?: string

  // Categorization
  category: MediaCategory
  suggestedCategory?: MediaCategory  // AI suggestion
  aiConfidence?: number              // 0-1 confidence score
  tags: string[]

  // Metadata
  width?: number
  height?: number
  duration?: number  // For videos/audio
  fileSize: number
  mimeType: string

  // Engagement
  views: number
  likes: number
  shares: number
  comments: number
  downloads: number
  isLiked?: boolean

  // Status
  isProcessing: boolean
  isFlagged: boolean
  isApproved: boolean

  // Timestamps
  createdAt: number
  updatedAt: number
}

export interface MediaCategory {
  id: string
  name: string
  slug: string
  description: string
  icon: string
  color: string

  // Stats
  itemCount: number
  followers: number
  trending: boolean

  // Auto-creation
  isAutoCreated: boolean
  createdFromSearches: number  // Number of searches that triggered creation

  createdAt: number
}

export interface MediaUpload {
  file: File
  title: string
  description?: string
  category?: MediaCategory
  tags: string[]
  // AI will analyze and suggest category if not provided
}

export interface AIAnalysis {
  detectedType: MediaType
  suggestedCategory: MediaCategory
  confidence: number
  detectedObjects: string[]
  detectedText?: string
  isNSFW: boolean
  quality: number  // 0-1
  tags: string[]
}

export interface TrendingSearch {
  query: string
  count: number
  category?: MediaCategory
  shouldCreateCategory: boolean
  timestamp: number
}

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

// Media API
export const mediaAPI = {
  // Browse media
  getFeed: async (filters?: {
    category?: MediaCategory
    type?: MediaType
    tags?: string[]
    search?: string
    trending?: boolean
    limit?: number
    offset?: number
  }) => {
    const response = await apiClient.get<MediaItem[]>('/media', {
      params: filters,
    })
    return response.data
  },

  // Get single media item
  getMedia: async (mediaId: string) => {
    const response = await apiClient.get<MediaItem>(`/media/${mediaId}`)
    return response.data
  },

  // Upload media
  uploadMedia: async (upload: MediaUpload) => {
    const formData = new FormData()
    formData.append('file', upload.file)
    formData.append('title', upload.title)
    if (upload.description) formData.append('description', upload.description)
    if (upload.category) formData.append('category', upload.category)
    formData.append('tags', JSON.stringify(upload.tags))

    const response = await apiClient.post('/media/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // AI analysis (automatic on upload)
  analyzeMedia: async (mediaId: string) => {
    const response = await apiClient.post<AIAnalysis>(`/media/${mediaId}/analyze`)
    return response.data
  },

  // Update media category (if user disagrees with AI)
  updateCategory: async (mediaId: string, category: MediaCategory) => {
    const response = await apiClient.put(`/media/${mediaId}/category`, {
      category,
    })
    return response.data
  },

  // Delete media
  deleteMedia: async (mediaId: string) => {
    const response = await apiClient.delete(`/media/${mediaId}`)
    return response.data
  },

  // Interactions
  likeMedia: async (mediaId: string) => {
    const response = await apiClient.post(`/media/${mediaId}/like`)
    return response.data
  },

  unlikeMedia: async (mediaId: string) => {
    const response = await apiClient.delete(`/media/${mediaId}/like`)
    return response.data
  },

  shareMedia: async (mediaId: string) => {
    const response = await apiClient.post(`/media/${mediaId}/share`)
    return response.data
  },

  downloadMedia: async (mediaId: string) => {
    const response = await apiClient.get(`/media/${mediaId}/download`, {
      responseType: 'blob',
    })
    return response.data
  },

  // Comments
  getComments: async (mediaId: string) => {
    const response = await apiClient.get<MediaComment[]>(
      `/media/${mediaId}/comments`
    )
    return response.data
  },

  addComment: async (mediaId: string, content: string) => {
    const response = await apiClient.post(`/media/${mediaId}/comments`, {
      content,
    })
    return response.data
  },

  deleteComment: async (mediaId: string, commentId: string) => {
    const response = await apiClient.delete(
      `/media/${mediaId}/comments/${commentId}`
    )
    return response.data
  },

  // Categories
  getCategories: async () => {
    const response = await apiClient.get<MediaCategory[]>('/media/categories')
    return response.data
  },

  followCategory: async (categoryId: string) => {
    const response = await apiClient.post(`/media/categories/${categoryId}/follow`)
    return response.data
  },

  unfollowCategory: async (categoryId: string) => {
    const response = await apiClient.delete(
      `/media/categories/${categoryId}/follow`
    )
    return response.data
  },

  // Trending & Search
  getTrending: async (limit: number = 20) => {
    const response = await apiClient.get<MediaItem[]>('/media/trending', {
      params: { limit },
    })
    return response.data
  },

  getTrendingSearches: async () => {
    const response = await apiClient.get<TrendingSearch[]>(
      '/media/trending/searches'
    )
    return response.data
  },

  searchMedia: async (query: string, limit: number = 20) => {
    const response = await apiClient.get<MediaItem[]>('/media/search', {
      params: { q: query, limit },
    })
    return response.data
  },

  // My media
  getMyMedia: async () => {
    const response = await apiClient.get<MediaItem[]>('/media/me')
    return response.data
  },

  // Stats
  getStats: async (mediaId: string) => {
    const response = await apiClient.get<MediaStats>(`/media/${mediaId}/stats`)
    return response.data
  },

  // Report
  reportMedia: async (mediaId: string, reason: string) => {
    const response = await apiClient.post(`/media/${mediaId}/report`, {
      reason,
    })
    return response.data
  },
}

// Category metadata
export const CATEGORY_INFO: Record<MediaCategory, {
  name: string
  description: string
  icon: string
  color: string
  examples: string[]
}> = {
  memes: {
    name: 'Memes',
    description: 'Funny images, reaction pics, and viral content',
    icon: '😂',
    color: '#39FF14',
    examples: ['Drake meme', 'Distracted boyfriend', 'Woman yelling at cat'],
  },
  dances: {
    name: 'Dances',
    description: 'Dance videos, choreography, TikTok trends',
    icon: '💃',
    color: '#FF10F0',
    examples: ['Renegade', 'Savage dance', 'Shuffling'],
  },
  art: {
    name: 'Art',
    description: 'Digital art, illustrations, paintings, drawings',
    icon: '🎨',
    color: '#00FFFF',
    examples: ['Digital painting', 'Character design', '3D renders'],
  },
  photography: {
    name: 'Photography',
    description: 'Professional photos, landscapes, portraits',
    icon: '📸',
    color: '#0EA5E9',
    examples: ['Nature photography', 'Street photography', 'Portraits'],
  },
  animations: {
    name: 'Animations',
    description: 'Animated content, GIFs, motion graphics',
    icon: '🎬',
    color: '#F59E0B',
    examples: ['2D animation', '3D animation', 'Stop motion'],
  },
  music: {
    name: 'Music',
    description: 'Songs, beats, covers, original compositions',
    icon: '🎵',
    color: '#8B5CF6',
    examples: ['Original songs', 'Covers', 'Beats'],
  },
  videos: {
    name: 'Videos',
    description: 'General video content, vlogs, short films',
    icon: '🎥',
    color: '#EF4444',
    examples: ['Vlogs', 'Short films', 'Documentaries'],
  },
  tutorials: {
    name: 'Tutorials',
    description: 'How-to guides, coding tutorials, educational content',
    icon: '📚',
    color: '#10B981',
    examples: ['Coding tutorials', 'Art tutorials', 'Life hacks'],
  },
  gaming: {
    name: 'Gaming',
    description: 'Game clips, highlights, playthroughs',
    icon: '🎮',
    color: '#6366F1',
    examples: ['Gameplay', 'Highlights', 'Game reviews'],
  },
  comedy: {
    name: 'Comedy',
    description: 'Stand-up, sketches, funny videos',
    icon: '🤣',
    color: '#EC4899',
    examples: ['Stand-up clips', 'Skits', 'Pranks'],
  },
  education: {
    name: 'Education',
    description: 'Educational content, explanations, learning',
    icon: '🎓',
    color: '#14B8A6',
    examples: ['Science', 'History', 'Math'],
  },
  other: {
    name: 'Other',
    description: 'Content that doesn\'t fit other categories',
    icon: '📁',
    color: '#6B7280',
    examples: ['Misc', 'Uncategorized'],
  },
}
