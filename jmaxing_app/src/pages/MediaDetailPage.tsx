import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { mediaAPI, CATEGORY_INFO } from '../api/media'
import { useAuthStore } from '../store/authStore'
import {
  Heart,
  Download,
  Share2,
  Eye,
  MessageCircle,
  Flag,
  Trash2,
  Edit,
  Sparkles,
  ChevronLeft,
  Send,
} from 'lucide-react'

export default function MediaDetailPage() {
  const { mediaId } = useParams<{ mediaId: string }>()
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [commentText, setCommentText] = useState('')
  const [showShareMenu, setShowShareMenu] = useState(false)

  // Fetch media
  const { data: media, isLoading } = useQuery({
    queryKey: ['media', mediaId],
    queryFn: () => mediaAPI.getMedia(mediaId!),
    enabled: !!mediaId,
  })

  // Fetch comments
  const { data: comments } = useQuery({
    queryKey: ['mediaComments', mediaId],
    queryFn: () => mediaAPI.getComments(mediaId!),
    enabled: !!mediaId,
  })

  // Fetch related media
  const { data: relatedMedia } = useQuery({
    queryKey: ['relatedMedia', media?.category],
    queryFn: () =>
      mediaAPI.getFeed({
        category: media?.category,
        limit: 6,
      }),
    enabled: !!media,
  })

  // Like mutation
  const likeMutation = useMutation({
    mutationFn: () =>
      media?.isLiked
        ? mediaAPI.unlikeMedia(mediaId!)
        : mediaAPI.likeMedia(mediaId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['media', mediaId] })
    },
  })

  // Download mutation
  const downloadMutation = useMutation({
    mutationFn: () => mediaAPI.downloadMedia(mediaId!),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = media?.title || 'download'
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      queryClient.invalidateQueries({ queryKey: ['media', mediaId] })
    },
  })

  // Share mutation
  const shareMutation = useMutation({
    mutationFn: () => mediaAPI.shareMedia(mediaId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['media', mediaId] })
    },
  })

  // Comment mutation
  const commentMutation = useMutation({
    mutationFn: (content: string) => mediaAPI.addComment(mediaId!, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mediaComments', mediaId] })
      setCommentText('')
    },
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: () => mediaAPI.deleteMedia(mediaId!),
    onSuccess: () => {
      navigate('/media')
    },
  })

  const handleShare = async () => {
    if (navigator.share && media) {
      try {
        await navigator.share({
          title: media.title,
          text: media.description || '',
          url: window.location.href,
        })
        shareMutation.mutate()
      } catch (err) {
        // User cancelled share or share not supported
        setShowShareMenu(true)
      }
    } else {
      setShowShareMenu(true)
    }
  }

  const copyLink = () => {
    navigator.clipboard.writeText(window.location.href)
    shareMutation.mutate()
    setShowShareMenu(false)
    alert('Link copied to clipboard!')
  }

  const handleComment = () => {
    if (commentText.trim()) {
      commentMutation.mutate(commentText)
    }
  }

  if (isLoading || !media) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green"></div>
      </div>
    )
  }

  const isOwner = user?.id === media.userId

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Back Button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-2 text-gray-400 hover:text-white mb-6 transition-colors"
      >
        <ChevronLeft className="w-5 h-5" />
        Back
      </button>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Media Display */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel overflow-hidden"
          >
            <div className="relative bg-gray-900">
              {media.type === 'video' ? (
                <video src={media.url} controls className="w-full max-h-[600px]" />
              ) : media.type === 'audio' ? (
                <div className="h-96 flex flex-col items-center justify-center bg-gradient-to-br from-neon-pink/20 to-neon-blue/20">
                  <div className="w-32 h-32 rounded-full bg-neon-pink/30 flex items-center justify-center mb-6">
                    <span className="text-6xl">🎵</span>
                  </div>
                  <audio src={media.url} controls className="w-full max-w-md px-8" />
                </div>
              ) : (
                <img
                  src={media.url}
                  alt={media.title}
                  className="w-full max-h-[600px] object-contain"
                />
              )}

              {/* AI Badge */}
              {media.suggestedCategory && media.aiConfidence && media.aiConfidence > 0.8 && (
                <div className="absolute top-4 left-4 px-3 py-2 bg-neon-green/90 backdrop-blur-sm rounded-lg flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-gray-950" />
                  <span className="text-sm font-bold text-gray-950">
                    AI Categorized: {Math.round(media.aiConfidence * 100)}%
                  </span>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="p-6 border-t border-gray-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {/* Like */}
                  <button
                    onClick={() => likeMutation.mutate()}
                    disabled={!user || likeMutation.isPending}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors disabled:opacity-50"
                  >
                    <Heart
                      className={`w-5 h-5 ${
                        media.isLiked
                          ? 'text-neon-pink fill-neon-pink'
                          : 'text-gray-400'
                      }`}
                    />
                    <span className="text-white font-semibold">
                      {media.likes.toLocaleString()}
                    </span>
                  </button>

                  {/* Download */}
                  <button
                    onClick={() => downloadMutation.mutate()}
                    disabled={downloadMutation.isPending}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                  >
                    <Download className="w-5 h-5 text-gray-400" />
                    <span className="text-white font-semibold">
                      {media.downloads.toLocaleString()}
                    </span>
                  </button>

                  {/* Share */}
                  <div className="relative">
                    <button
                      onClick={handleShare}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                    >
                      <Share2 className="w-5 h-5 text-gray-400" />
                      <span className="text-white font-semibold">
                        {media.shares.toLocaleString()}
                      </span>
                    </button>

                    {showShareMenu && (
                      <div className="absolute top-full mt-2 left-0 glass-panel p-4 min-w-[200px] z-10">
                        <button
                          onClick={copyLink}
                          className="w-full px-4 py-2 text-left text-white hover:bg-gray-800 rounded-lg transition-colors"
                        >
                          Copy Link
                        </button>
                        <button
                          onClick={() => setShowShareMenu(false)}
                          className="w-full px-4 py-2 text-left text-gray-400 hover:bg-gray-800 rounded-lg transition-colors mt-1"
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Comments */}
                  <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-800">
                    <MessageCircle className="w-5 h-5 text-gray-400" />
                    <span className="text-white font-semibold">
                      {media.comments.toLocaleString()}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isOwner && (
                    <>
                      <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors">
                        <Edit className="w-5 h-5 text-gray-400" />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm('Are you sure you want to delete this media?')) {
                            deleteMutation.mutate()
                          }
                        }}
                        className="p-2 rounded-lg bg-gray-800 hover:bg-red-900 transition-colors"
                      >
                        <Trash2 className="w-5 h-5 text-red-400" />
                      </button>
                    </>
                  )}
                  {!isOwner && user && (
                    <button className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors">
                      <Flag className="w-5 h-5 text-gray-400" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6"
          >
            {/* Title & Category */}
            <div className="flex items-start justify-between mb-4">
              <h1 className="text-3xl font-bold text-white">{media.title}</h1>
              <div
                className="px-3 py-1 rounded-lg flex items-center gap-2"
                style={{
                  backgroundColor: `${CATEGORY_INFO[media.category].color}20`,
                }}
              >
                <span>{CATEGORY_INFO[media.category].icon}</span>
                <span
                  className="font-semibold"
                  style={{ color: CATEGORY_INFO[media.category].color }}
                >
                  {CATEGORY_INFO[media.category].name}
                </span>
              </div>
            </div>

            {/* Description */}
            {media.description && (
              <p className="text-gray-400 mb-4">{media.description}</p>
            )}

            {/* Tags */}
            {media.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-4">
                {media.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 bg-gray-800 text-neon-green rounded-lg text-sm"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            {/* Creator */}
            <div className="flex items-center gap-3 pt-4 border-t border-gray-800">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-lg font-bold text-gray-950">
                {media.username.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <button
                  onClick={() => navigate(`/users/${media.userId}`)}
                  className="font-semibold text-white hover:text-neon-green transition-colors"
                >
                  @{media.username}
                </button>
                <div className="text-sm text-gray-500">
                  {new Date(media.createdAt).toLocaleDateString()}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Comments */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-6"
          >
            <h2 className="text-2xl font-bold text-white mb-6">
              Comments ({comments?.length || 0})
            </h2>

            {/* Add Comment */}
            {user && (
              <div className="mb-6 flex gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-neon-pink to-neon-blue flex items-center justify-center text-sm font-bold text-gray-950 flex-shrink-0">
                  {user.username.slice(0, 2).toUpperCase()}
                </div>
                <div className="flex-1">
                  <input
                    type="text"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleComment()}
                    placeholder="Add a comment..."
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-neon-green transition-colors mb-2"
                  />
                  <button
                    onClick={handleComment}
                    disabled={!commentText.trim() || commentMutation.isPending}
                    className="btn-secondary px-4 py-2 text-sm disabled:opacity-50"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Comment List */}
            <div className="space-y-4">
              {comments?.map((comment) => (
                <div key={comment.id} className="flex gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-sm font-bold text-gray-950 flex-shrink-0">
                    {comment.username.slice(0, 2).toUpperCase()}
                  </div>
                  <div className="flex-1 bg-gray-900 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <button
                        onClick={() => navigate(`/users/${comment.userId}`)}
                        className="font-semibold text-white hover:text-neon-green transition-colors"
                      >
                        @{comment.username}
                      </button>
                      <span className="text-xs text-gray-500">
                        {new Date(comment.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-gray-300 text-sm">{comment.content}</p>
                    <div className="flex items-center gap-4 mt-3">
                      <button className="flex items-center gap-1 text-xs text-gray-500 hover:text-neon-pink transition-colors">
                        <Heart className="w-3 h-3" />
                        {comment.likes}
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              {comments?.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No comments yet. Be the first to comment!
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">Stats</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-400">
                  <Eye className="w-4 h-4" />
                  <span className="text-sm">Views</span>
                </div>
                <span className="font-bold text-white">
                  {media.views.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-400">
                  <Heart className="w-4 h-4" />
                  <span className="text-sm">Likes</span>
                </div>
                <span className="font-bold text-neon-pink">
                  {media.likes.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-400">
                  <Download className="w-4 h-4" />
                  <span className="text-sm">Downloads</span>
                </div>
                <span className="font-bold text-neon-blue">
                  {media.downloads.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-400">
                  <Share2 className="w-4 h-4" />
                  <span className="text-sm">Shares</span>
                </div>
                <span className="font-bold text-neon-green">
                  {media.shares.toLocaleString()}
                </span>
              </div>
            </div>
          </motion.div>

          {/* AI Analysis */}
          {media.suggestedCategory && media.aiConfidence && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-panel p-6 border-l-4 border-neon-green"
            >
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-5 h-5 text-neon-green" />
                <h3 className="text-lg font-bold text-white">AI Analysis</h3>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Suggested Category</span>
                  <span className="text-white font-semibold">
                    {CATEGORY_INFO[media.suggestedCategory].name}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Confidence</span>
                  <span className="text-neon-green font-bold">
                    {Math.round(media.aiConfidence * 100)}%
                  </span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden mt-3">
                  <div
                    className="h-full bg-gradient-to-r from-neon-green to-neon-blue"
                    style={{ width: `${media.aiConfidence * 100}%` }}
                  />
                </div>
              </div>
            </motion.div>
          )}

          {/* Related Media */}
          {relatedMedia && relatedMedia.length > 1 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-panel p-6"
            >
              <h3 className="text-lg font-bold text-white mb-4">
                More from {CATEGORY_INFO[media.category].name}
              </h3>
              <div className="space-y-3">
                {relatedMedia
                  .filter((item) => item.id !== media.id)
                  .slice(0, 5)
                  .map((item) => (
                    <button
                      key={item.id}
                      onClick={() => navigate(`/media/${item.id}`)}
                      className="w-full flex gap-3 hover:bg-gray-800/50 rounded-lg p-2 transition-colors"
                    >
                      <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0 bg-gray-900">
                        {item.thumbnailUrl ? (
                          <img
                            src={item.thumbnailUrl}
                            alt={item.title}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-2xl">
                            {CATEGORY_INFO[item.category].icon}
                          </div>
                        )}
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-semibold text-white text-sm line-clamp-2">
                          {item.title}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          {item.views.toLocaleString()} views
                        </div>
                      </div>
                    </button>
                  ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
