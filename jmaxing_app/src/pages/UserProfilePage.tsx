import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { socialAPI } from '../api/social'
import { useAuthStore } from '../store/authStore'
import Editor from '@monaco-editor/react'

export default function UserProfilePage() {
  const { userId } = useParams<{ userId: string }>()
  const { user: currentUser } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'posts' | 'followers' | 'following'>(
    'posts'
  )

  // Fetch user profile
  const { data: profile, isLoading } = useQuery({
    queryKey: ['profile', userId],
    queryFn: () => socialAPI.getProfile(userId!),
    enabled: !!userId,
  })

  // Fetch user posts
  const { data: posts } = useQuery({
    queryKey: ['userPosts', userId],
    queryFn: () => socialAPI.getUserPosts(userId!),
    enabled: !!userId && activeTab === 'posts',
  })

  // Fetch followers/following
  const { data: followers } = useQuery({
    queryKey: ['followers', userId],
    queryFn: () => socialAPI.getFollowers(userId!),
    enabled: !!userId && activeTab === 'followers',
  })

  const { data: following } = useQuery({
    queryKey: ['following', userId],
    queryFn: () => socialAPI.getFollowing(userId!),
    enabled: !!userId && activeTab === 'following',
  })

  // Follow/unfollow mutation
  const followMutation = useMutation({
    mutationFn: (isFollowing: boolean) =>
      isFollowing ? socialAPI.unfollow(userId!) : socialAPI.follow(userId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profile', userId] })
    },
  })

  const handleFollow = () => {
    if (profile) {
      followMutation.mutate(profile.isFollowing || false)
    }
  }

  if (isLoading || !profile) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green"></div>
      </div>
    )
  }

  const isOwnProfile = currentUser?.id === profile.id

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Profile Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel p-8 mb-8"
      >
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-6">
            {/* Avatar */}
            <div className="w-32 h-32 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-5xl font-bold text-gray-950">
              {profile.username.slice(0, 2).toUpperCase()}
            </div>

            {/* User Info */}
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                {profile.username}
              </h1>
              {profile.bio && (
                <p className="text-gray-400 mb-3">{profile.bio}</p>
              )}

              {/* Social Links */}
              <div className="flex items-center gap-4 mb-4">
                {profile.location && (
                  <span className="text-gray-500 text-sm flex items-center gap-1">
                    📍 {profile.location}
                  </span>
                )}
                {profile.website && (
                  <a
                    href={profile.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-neon-blue hover:text-neon-blue/80 text-sm"
                  >
                    🔗 Website
                  </a>
                )}
                {profile.github && (
                  <a
                    href={`https://github.com/${profile.github}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-white text-sm"
                  >
                    GitHub
                  </a>
                )}
              </div>

              {/* Stats */}
              <div className="flex items-center gap-6">
                <button
                  onClick={() => setActiveTab('followers')}
                  className="hover:text-neon-green transition-colors"
                >
                  <span className="font-bold text-white">
                    {profile.followers}
                  </span>{' '}
                  <span className="text-gray-400">Followers</span>
                </button>
                <button
                  onClick={() => setActiveTab('following')}
                  className="hover:text-neon-green transition-colors"
                >
                  <span className="font-bold text-white">
                    {profile.following}
                  </span>{' '}
                  <span className="text-gray-400">Following</span>
                </button>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            {isOwnProfile ? (
              <>
                <button
                  onClick={() => navigate('/profile')}
                  className="btn-secondary"
                >
                  Edit Profile
                </button>
                <button
                  onClick={() => navigate('/referrals')}
                  className="btn-neon"
                >
                  Refer & Earn
                </button>
              </>
            ) : (
              <button
                onClick={handleFollow}
                disabled={followMutation.isPending}
                className={
                  profile.isFollowing ? 'btn-secondary' : 'btn-neon'
                }
              >
                {followMutation.isPending
                  ? 'Loading...'
                  : profile.isFollowing
                  ? 'Unfollow'
                  : 'Follow'}
              </button>
            )}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-green mb-1">
              #{profile.rank}
            </div>
            <div className="text-xs text-gray-400">Rank</div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white mb-1">
              {profile.reputation}
            </div>
            <div className="text-xs text-gray-400">Reputation</div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-blue mb-1">
              {profile.tasksCompleted}
            </div>
            <div className="text-xs text-gray-400">Tasks Completed</div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-pink mb-1">
              {profile.balance} JC
            </div>
            <div className="text-xs text-gray-400">Balance</div>
          </div>
        </div>
      </motion.div>

      {/* Tabs */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="flex gap-2 mb-6"
      >
        <button
          onClick={() => setActiveTab('posts')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            activeTab === 'posts'
              ? 'bg-neon-green text-gray-950'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Posts
        </button>
        <button
          onClick={() => setActiveTab('followers')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            activeTab === 'followers'
              ? 'bg-neon-green text-gray-950'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Followers ({profile.followers})
        </button>
        <button
          onClick={() => setActiveTab('following')}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            activeTab === 'following'
              ? 'bg-neon-green text-gray-950'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Following ({profile.following})
        </button>
      </motion.div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {activeTab === 'posts' && (
          <div className="space-y-6">
            {!posts || posts.length === 0 ? (
              <div className="glass-panel p-12 text-center">
                <div className="text-6xl mb-4">📝</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  No posts yet
                </h3>
                <p className="text-gray-400">
                  {isOwnProfile
                    ? 'Share your first code or achievement!'
                    : `${profile.username} hasn't posted anything yet`}
                </p>
              </div>
            ) : (
              posts.map((post, index) => (
                <motion.div
                  key={post.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="glass-panel p-6"
                >
                  {/* Post header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-xl font-bold text-gray-950">
                        {profile.username.slice(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div className="font-semibold text-white">
                          {profile.username}
                        </div>
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
                  <div className="text-white mb-4 whitespace-pre-wrap">
                    {post.content}
                  </div>

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
                        }}
                      />
                    </div>
                  )}

                  {/* Payout */}
                  {post.payout && post.payout > 0 && (
                    <div className="bg-neon-green/10 border border-neon-green/30 rounded-lg p-3">
                      <span className="text-neon-green font-bold">
                        💎 Earned {post.payout} JC
                      </span>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex items-center gap-6 pt-4 mt-4 border-t border-gray-800">
                    <div className="flex items-center gap-2 text-gray-400">
                      <span className="text-xl">❤️</span>
                      <span className="font-semibold">{post.likes}</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-400">
                      <span className="text-xl">💬</span>
                      <span className="font-semibold">{post.comments}</span>
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        )}

        {activeTab === 'followers' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {!followers || followers.length === 0 ? (
              <div className="glass-panel p-12 text-center col-span-2">
                <div className="text-6xl mb-4">👥</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  No followers yet
                </h3>
              </div>
            ) : (
              followers.map((follower) => (
                <UserCard key={follower.id} user={follower} />
              ))
            )}
          </div>
        )}

        {activeTab === 'following' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {!following || following.length === 0 ? (
              <div className="glass-panel p-12 text-center col-span-2">
                <div className="text-6xl mb-4">🔍</div>
                <h3 className="text-xl font-bold text-white mb-2">
                  Not following anyone
                </h3>
              </div>
            ) : (
              following.map((user) => (
                <UserCard key={user.id} user={user} />
              ))
            )}
          </div>
        )}
      </motion.div>
    </div>
  )
}

// User Card Component
function UserCard({ user }: { user: any }) {
  const navigate = useNavigate()

  return (
    <div
      onClick={() => navigate(`/users/${user.id}`)}
      className="glass-panel p-4 hover:border-neon-green/30 transition-all cursor-pointer"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-pink to-neon-blue flex items-center justify-center text-xl font-bold text-gray-950">
            {user.username.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div className="font-semibold text-white">{user.username}</div>
            <div className="text-xs text-gray-500">
              Rank #{user.rank} • {user.reputation} rep
            </div>
          </div>
        </div>
        <button className="text-neon-green hover:text-neon-green/80 text-sm font-semibold">
          View Profile
        </button>
      </div>
    </div>
  )
}
