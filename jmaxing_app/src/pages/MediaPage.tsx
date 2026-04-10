import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  mediaAPI,
  type MediaItem,
  type MediaCategory,
  type MediaType,
  CATEGORY_INFO,
} from '../api/media'
import { useAuthStore } from '../store/authStore'
import {
  Search,
  Upload,
  TrendingUp,
  Heart,
  Download,
  Eye,
  Play,
  Image as ImageIcon,
  Music,
  Film,
} from 'lucide-react'

export default function MediaPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const [selectedCategory, setSelectedCategory] = useState<MediaCategory | 'all'>('all')
  const [selectedType, setSelectedType] = useState<MediaType | 'all'>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  // Fetch media feed
  const { data: mediaItems, isLoading } = useQuery({
    queryKey: ['media', selectedCategory, selectedType, searchQuery],
    queryFn: () =>
      mediaAPI.getFeed({
        category: selectedCategory === 'all' ? undefined : selectedCategory,
        type: selectedType === 'all' ? undefined : selectedType,
        search: searchQuery || undefined,
        limit: 50,
      }),
  })

  // Fetch trending
  const { data: trending } = useQuery({
    queryKey: ['mediaTrending'],
    queryFn: () => mediaAPI.getTrending(10),
  })

  // Fetch trending searches
  const { data: trendingSearches } = useQuery({
    queryKey: ['trendingSearches'],
    queryFn: () => mediaAPI.getTrendingSearches(),
  })

  const categories = Object.entries(CATEGORY_INFO) as [MediaCategory, typeof CATEGORY_INFO[MediaCategory]][]

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-5xl font-bold mb-4">
              <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
                Media Gallery
              </span>
            </h1>
            <p className="text-gray-400 text-lg">
              Share and discover memes, dances, art, and more
            </p>
          </div>

          {user && (
            <button
              onClick={() => navigate('/media/upload')}
              className="btn-neon flex items-center gap-2"
            >
              <Upload className="w-5 h-5" />
              Upload Media
            </button>
          )}
        </div>

        {/* Search */}
        <div className="relative max-w-xl mb-6">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search media..."
            className="w-full pl-12 pr-4 py-3 bg-gray-900 border border-gray-800 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-neon-green transition-colors"
          />
        </div>

        {/* Trending Searches */}
        {trendingSearches && trendingSearches.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-5 h-5 text-neon-green" />
              <h3 className="text-sm font-semibold text-white">Trending Searches</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {trendingSearches.slice(0, 10).map((search) => (
                <button
                  key={search.query}
                  onClick={() => setSearchQuery(search.query)}
                  className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded-full text-sm text-gray-300 transition-colors"
                >
                  {search.query}
                  {search.shouldCreateCategory && (
                    <span className="ml-2 text-neon-green">🔥</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Category Tabs */}
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap flex items-center gap-2 ${
              selectedCategory === 'all'
                ? 'bg-neon-green text-gray-950'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            All Media
          </button>
          {categories.map(([key, info]) => (
            <button
              key={key}
              onClick={() => setSelectedCategory(key)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap flex items-center gap-2 ${
                selectedCategory === key
                  ? 'bg-neon-green text-gray-950'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              <span>{info.icon}</span>
              {info.name}
            </button>
          ))}
        </div>

        {/* Type Filters */}
        <div className="flex items-center gap-2 mt-4">
          <span className="text-sm text-gray-500">Filter by type:</span>
          {[
            { key: 'all', label: 'All', icon: null },
            { key: 'image', label: 'Images', icon: ImageIcon },
            { key: 'video', label: 'Videos', icon: Film },
            { key: 'gif', label: 'GIFs', icon: Play },
            { key: 'audio', label: 'Audio', icon: Music },
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setSelectedType(key as MediaType | 'all')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${
                selectedType === key
                  ? 'bg-neon-blue text-gray-950'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {Icon && <Icon className="w-4 h-4" />}
              {label}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Trending Section */}
      {trending && trending.length > 0 && selectedCategory === 'all' && !searchQuery && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-6 h-6 text-neon-pink" />
            <h2 className="text-2xl font-bold text-white">Trending Now</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {trending.slice(0, 5).map((item, index) => (
              <MediaCard key={item.id} item={item} index={index} compact />
            ))}
          </div>
        </motion.div>
      )}

      {/* Media Grid */}
      {isLoading ? (
        <div className="flex justify-center py-20">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green"></div>
        </div>
      ) : !mediaItems || mediaItems.length === 0 ? (
        <div className="glass-panel p-16 text-center">
          <div className="text-6xl mb-4">📁</div>
          <h3 className="text-2xl font-bold text-white mb-2">No media found</h3>
          <p className="text-gray-400 mb-6">
            {searchQuery
              ? 'Try a different search term or category'
              : 'Be the first to upload media in this category'}
          </p>
          {user && (
            <button
              onClick={() => navigate('/media/upload')}
              className="btn-neon"
            >
              Upload Media
            </button>
          )}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className={
            viewMode === 'grid'
              ? 'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'
              : 'space-y-4'
          }
        >
          {mediaItems.map((item, index) => (
            <MediaCard key={item.id} item={item} index={index} />
          ))}
        </motion.div>
      )}

      {/* Category Info Panel */}
      {selectedCategory !== 'all' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-panel p-8 mt-12"
        >
          <div className="flex items-center gap-4 mb-4">
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-3xl"
              style={{ backgroundColor: `${CATEGORY_INFO[selectedCategory].color}20` }}
            >
              {CATEGORY_INFO[selectedCategory].icon}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">
                {CATEGORY_INFO[selectedCategory].name}
              </h2>
              <p className="text-gray-400">
                {CATEGORY_INFO[selectedCategory].description}
              </p>
            </div>
          </div>
          <div className="mt-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-2">
              Popular Examples:
            </h3>
            <div className="flex flex-wrap gap-2">
              {CATEGORY_INFO[selectedCategory].examples.map((example) => (
                <span
                  key={example}
                  className="px-3 py-1 bg-gray-800 text-gray-300 rounded-lg text-sm"
                >
                  {example}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

// Media Card Component
function MediaCard({
  item,
  index,
  compact = false,
}: {
  item: MediaItem
  index: number
  compact?: boolean
}) {
  const navigate = useNavigate()

  const getTypeIcon = (type: MediaType) => {
    switch (type) {
      case 'video':
        return <Play className="w-6 h-6" />
      case 'audio':
        return <Music className="w-6 h-6" />
      default:
        return null
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: compact ? 0 : index * 0.05 }}
      onClick={() => navigate(`/media/${item.id}`)}
      className="glass-panel overflow-hidden hover:border-neon-green/30 transition-all cursor-pointer group relative"
    >
      {/* Media Preview */}
      <div className={`relative ${compact ? 'h-32' : 'h-64'} bg-gray-900 overflow-hidden`}>
        {item.type === 'audio' ? (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-neon-pink/20 to-neon-blue/20">
            <Music className="w-16 h-16 text-neon-pink" />
          </div>
        ) : (
          <>
            <img
              src={item.thumbnailUrl || item.url}
              alt={item.title}
              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
            />
            {item.type === 'video' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                <div className="w-16 h-16 rounded-full bg-neon-green/20 backdrop-blur-sm flex items-center justify-center">
                  <Play className="w-8 h-8 text-neon-green" />
                </div>
              </div>
            )}
          </>
        )}

        {/* AI Badge */}
        {item.suggestedCategory && item.aiConfidence && item.aiConfidence > 0.9 && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-neon-green/90 rounded text-xs font-bold text-gray-950">
            AI: {Math.round(item.aiConfidence * 100)}%
          </div>
        )}

        {/* Category Badge */}
        <div
          className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold backdrop-blur-sm"
          style={{ backgroundColor: `${CATEGORY_INFO[item.category].color}90` }}
        >
          {CATEGORY_INFO[item.category].icon} {CATEGORY_INFO[item.category].name}
        </div>
      </div>

      {/* Info */}
      {!compact && (
        <div className="p-4">
          <h3 className="font-bold text-white mb-1 line-clamp-2 group-hover:text-neon-green transition-colors">
            {item.title}
          </h3>
          {item.description && (
            <p className="text-sm text-gray-400 mb-3 line-clamp-1">
              {item.description}
            </p>
          )}

          {/* Tags */}
          {item.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {item.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Stats */}
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Eye className="w-4 h-4" />
              <span>{item.views.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <Heart className={`w-4 h-4 ${item.isLiked ? 'text-neon-pink fill-neon-pink' : ''}`} />
              <span>{item.likes.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <Download className="w-4 h-4" />
              <span>{item.downloads.toLocaleString()}</span>
            </div>
          </div>

          {/* User */}
          <div className="mt-3 pt-3 border-t border-gray-800 flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-xs font-bold text-gray-950">
              {item.username.slice(0, 2).toUpperCase()}
            </div>
            <span className="text-sm text-gray-400">@{item.username}</span>
          </div>
        </div>
      )}
    </motion.div>
  )
}
