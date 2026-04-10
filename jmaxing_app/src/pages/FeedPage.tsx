import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { socialAPI, type Post } from '../api/social'
import { useAuthStore } from '../store/authStore'
import Editor from '@monaco-editor/react'

export default function FeedPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'feed' | 'explore'>('feed')
  const [newPost, setNewPost] = useState('')
  const [showCodeEditor, setShowCodeEditor] = useState(false)
  const [postCode, setPostCode] = useState('')

  // Fetch feed
  const { data: posts, isLoading } = useQuery({
    queryKey: ['feed', activeTab],
    queryFn: () =>
      activeTab === 'feed' ? socialAPI.getFeed() : socialAPI.getExplorePosts(),
    enabled: !!user,
  })

  // Create post mutation
  const createPostMutation = useMutation({
    mutationFn: (post: { content: string; code?: string }) =>
      socialAPI.createPost({
        type: post.code ? 'showcase' : 'status',
        content: post.content,
        code: post.code,
        language: 'python',
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feed'] })
      setNewPost('')
      setPostCode('')
      setShowCodeEditor(false)
    },
  })

  // Like mutation
  const likeMutation = useMutation({
    mutationFn: ({ postId, isLiked }: { postId: string; isLiked: boolean }) =>
      isLiked ? socialAPI.unlikePost(postId) : socialAPI.likePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feed'] })
    },
  })

  const handleCreatePost = () => {
    if (!newPost.trim() && !postCode.trim()) return
    createPostMutation.mutate({
      content: newPost,
      code: postCode || undefined,
    })
  }

  const handleLike = (postId: string, isLiked: boolean) => {
    likeMutation.mutate({ postId, isLiked })
  }

  if (!user) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Login Required</h2>
        <p className="text-gray-400 mb-6">
          Join J-MAXING to see what the community is building
        </p>
        <button onClick={() => navigate('/login')} className="btn-neon">
          Login
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Feed */}
        <div className="lg:col-span-2 space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-4xl font-bold mb-6">
              <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
                Feed
              </span>
            </h1>

            {/* Tabs */}
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setActiveTab('feed')}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  activeTab === 'feed'
                    ? 'bg-neon-green text-gray-950'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Following
              </button>
              <button
                onClick={() => setActiveTab('explore')}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  activeTab === 'explore'
                    ? 'bg-neon-green text-gray-950'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Explore
              </button>
            </div>
          </motion.div>

          {/* Create Post */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6"
          >
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-xl font-bold text-gray-950">
                {user.username.slice(0, 2).toUpperCase()}
              </div>

              <div className="flex-1">
                <textarea
                  value={newPost}
                  onChange={(e) => setNewPost(e.target.value)}
                  placeholder="Share your thoughts, code, or achievements..."
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-4 text-white placeholder-gray-500 focus:outline-none focus:border-neon-green transition-colors resize-none"
                  rows={3}
                />

                {showCodeEditor && (
                  <div className="mt-4 bg-gray-950 rounded-lg overflow-hidden border border-gray-800">
                    <Editor
                      height="200px"
                      language="python"
                      value={postCode}
                      onChange={(value) => setPostCode(value || '')}
                      theme="vs-dark"
                      options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        scrollBeyondLastLine: false,
                      }}
                    />
                  </div>
                )}

                <div className="flex items-center justify-between mt-4">
                  <button
                    onClick={() => setShowCodeEditor(!showCodeEditor)}
                    className="text-neon-blue hover:text-neon-blue/80 text-sm font-semibold"
                  >
                    {showCodeEditor ? '✕ Remove Code' : '{ } Add Code'}
                  </button>

                  <button
                    onClick={handleCreatePost}
                    disabled={createPostMutation.isPending}
                    className="btn-neon px-6 py-2"
                  >
                    {createPostMutation.isPending ? 'Posting...' : 'Post'}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Posts */}
          {isLoading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-neon-green"></div>
            </div>
          ) : !posts || posts.length === 0 ? (
            <div className="glass-panel p-12 text-center">
              <div className="text-6xl mb-4">📱</div>
              <h3 className="text-xl font-bold text-white mb-2">
                {activeTab === 'feed' ? 'Your feed is empty' : 'No posts yet'}
              </h3>
              <p className="text-gray-400 mb-6">
                {activeTab === 'feed'
                  ? 'Follow other developers to see their activity'
                  : 'Be the first to share something!'}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {posts.map((post, index) => (
                <PostCard
                  key={post.id}
                  post={post}
                  index={index}
                  onLike={handleLike}
                  onUserClick={(userId) => navigate(`/users/${userId}`)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* User Stats */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">Your Stats</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Balance</span>
                <span className="text-neon-green font-bold">
                  {user.balance} JC
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Reputation</span>
                <span className="text-white font-semibold">{user.reputation}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Rank</span>
                <span className="text-white font-semibold">#{user.rank}</span>
              </div>
            </div>
          </motion.div>

          {/* Trending Tags */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">
              Trending Topics
            </h3>
            <div className="space-y-2">
              {['#python', '#optimization', '#algorithms', '#refactoring', '#performance'].map(
                (tag) => (
                  <button
                    key={tag}
                    className="block w-full text-left px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors"
                  >
                    <span className="text-neon-blue font-semibold">{tag}</span>
                  </button>
                )
              )}
            </div>
          </motion.div>

          {/* Suggestions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">
              Who to Follow
            </h3>
            <div className="space-y-3">
              {['CodeMaster', 'PyNinja', 'AlgoQueen'].map((username) => (
                <div
                  key={username}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-neon-pink to-neon-blue flex items-center justify-center text-sm font-bold text-gray-950">
                      {username.slice(0, 2).toUpperCase()}
                    </div>
                    <span className="text-white font-semibold">{username}</span>
                  </div>
                  <button className="text-neon-green hover:text-neon-green/80 text-sm font-semibold">
                    Follow
                  </button>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

// Post Card Component
function PostCard({
  post,
  index,
  onLike,
  onUserClick,
}: {
  post: Post
  index: number
  onLike: (postId: string, isLiked: boolean) => void
  onUserClick: (userId: string) => void
}) {
  const navigate = useNavigate()
  const [showComments, setShowComments] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="glass-panel p-6"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => onUserClick(post.userId)}
        >
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-xl font-bold text-gray-950">
            {post.username.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div className="font-semibold text-white">{post.username}</div>
            <div className="text-xs text-gray-500">
              {new Date(post.timestamp).toLocaleDateString()}
            </div>
          </div>
        </div>

        {post.type === 'submission' && post.score && (
          <div className="badge-success">
            Score: {(post.score * 100).toFixed(0)}%
          </div>
        )}
      </div>

      {/* Content */}
      <div className="text-white mb-4 whitespace-pre-wrap">{post.content}</div>

      {/* Code */}
      {post.code && (
        <div className="bg-gray-950 rounded-lg overflow-hidden border border-gray-800 mb-4">
          <Editor
            height="200px"
            language={post.language || 'python'}
            value={post.code}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 14,
              scrollBeyondLastLine: false,
            }}
          />
        </div>
      )}

      {/* Payout */}
      {post.payout && post.payout > 0 && (
        <div className="bg-neon-green/10 border border-neon-green/30 rounded-lg p-3 mb-4">
          <span className="text-neon-green font-bold">
            💎 Earned {post.payout} JC
          </span>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-6 pt-4 border-t border-gray-800">
        <button
          onClick={() => onLike(post.id, post.isLiked || false)}
          className={`flex items-center gap-2 transition-colors ${
            post.isLiked
              ? 'text-neon-pink'
              : 'text-gray-400 hover:text-neon-pink'
          }`}
        >
          <span className="text-xl">{post.isLiked ? '❤️' : '🤍'}</span>
          <span className="font-semibold">{post.likes}</span>
        </button>

        <button
          onClick={() => setShowComments(!showComments)}
          className="flex items-center gap-2 text-gray-400 hover:text-neon-blue transition-colors"
        >
          <span className="text-xl">💬</span>
          <span className="font-semibold">{post.comments}</span>
        </button>

        <button className="flex items-center gap-2 text-gray-400 hover:text-neon-green transition-colors">
          <span className="text-xl">🔄</span>
          <span className="font-semibold">{post.shares}</span>
        </button>

        {post.jobId && (
          <button
            onClick={() => navigate(`/jobs/${post.jobId}`)}
            className="ml-auto text-neon-green hover:text-neon-green/80 text-sm font-semibold"
          >
            View Job →
          </button>
        )}
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="mt-4 pt-4 border-t border-gray-800">
          <div className="text-gray-400 text-sm">
            Comments coming soon...
          </div>
        </div>
      )}
    </motion.div>
  )
}
