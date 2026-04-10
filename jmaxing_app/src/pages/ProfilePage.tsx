import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { submissionsAPI } from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useNavigate } from 'react-router-dom'

export default function ProfilePage() {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const { data: submissions } = useQuery({
    queryKey: ['mySubmissions'],
    queryFn: submissionsAPI.getMySubmissions,
    enabled: !!user,
  })

  if (!user) {
    return null
  }

  const acceptedSubmissions = submissions?.filter((s) => s.status === 'scored') || []
  const rejectedSubmissions = submissions?.filter((s) => s.status === 'rejected') || []
  const pendingSubmissions = submissions?.filter((s) => s.status === 'pending') || []

  const totalEarned = acceptedSubmissions.reduce((sum, s) => sum + s.payout, 0)
  const avgScore =
    acceptedSubmissions.length > 0
      ? acceptedSubmissions.reduce((sum, s) => sum + s.score, 0) / acceptedSubmissions.length
      : 0

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Profile Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel p-8 mb-8"
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-6">
            {/* Avatar */}
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-4xl font-bold text-gray-950">
              {user.username.slice(0, 2).toUpperCase()}
            </div>

            {/* User Info */}
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                {user.username}
              </h1>
              <p className="text-gray-400 mb-3">{user.email}</p>
              <div className="flex items-center gap-4">
                <span className="badge-success">Rank #{user.rank}</span>
                <span className="text-gray-400">
                  Joined {new Date(user.joinedAt).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>

          {/* Logout Button */}
          <button
            onClick={handleLogout}
            className="btn-secondary text-sm"
          >
            Logout
          </button>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
      >
        <div className="stat-card">
          <div className="text-3xl font-bold text-neon-green mb-2">
            {user.balance.toLocaleString()} JC
          </div>
          <div className="text-sm text-gray-400">Current Balance</div>
        </div>

        <div className="stat-card">
          <div className="text-3xl font-bold text-neon-blue mb-2">
            {totalEarned.toLocaleString()} JC
          </div>
          <div className="text-sm text-gray-400">Total Earned</div>
        </div>

        <div className="stat-card">
          <div className="text-3xl font-bold text-neon-pink mb-2">
            {user.tasksCompleted}
          </div>
          <div className="text-sm text-gray-400">Tasks Completed</div>
        </div>

        <div className="stat-card">
          <div className="text-3xl font-bold text-white mb-2">
            {(avgScore * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-400">Avg Score</div>
        </div>
      </motion.div>

      {/* Reputation & Achievements */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-panel p-6 mb-8"
      >
        <h2 className="text-xl font-bold text-white mb-4">Reputation</h2>
        <div className="flex items-center gap-4 mb-4">
          <div className="flex-1">
            <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-neon-green to-neon-blue"
                style={{ width: `${(user.reputation / 1000) * 100}%` }}
              ></div>
            </div>
          </div>
          <span className="text-white font-bold">{user.reputation} / 1000</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-2xl mb-2">🏆</div>
            <div className="text-sm font-semibold text-white mb-1">
              Code Master
            </div>
            <div className="text-xs text-gray-500">
              {acceptedSubmissions.length >= 10 ? 'Unlocked' : `${10 - acceptedSubmissions.length} more to unlock`}
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-2xl mb-2">⚡</div>
            <div className="text-sm font-semibold text-white mb-1">
              Speed Demon
            </div>
            <div className="text-xs text-gray-500">
              {acceptedSubmissions.filter(s => s.time_taken < 300).length >= 5 ? 'Unlocked' : 'Locked'}
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-2xl mb-2">💎</div>
            <div className="text-sm font-semibold text-white mb-1">
              Perfect Score
            </div>
            <div className="text-xs text-gray-500">
              {acceptedSubmissions.filter(s => s.score >= 0.95).length >= 1 ? 'Unlocked' : 'Locked'}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Submission History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-panel p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white">Submission History</h2>
          <div className="flex items-center gap-3 text-sm">
            <span className="badge-success">{acceptedSubmissions.length} Accepted</span>
            <span className="badge-warning">{pendingSubmissions.length} Pending</span>
            <span className="badge-error">{rejectedSubmissions.length} Rejected</span>
          </div>
        </div>

        {!submissions || submissions.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📝</div>
            <h3 className="text-xl font-bold text-white mb-2">
              No submissions yet
            </h3>
            <p className="text-gray-400 mb-6">
              Start solving jobs to build your reputation!
            </p>
            <button
              onClick={() => navigate('/jobs')}
              className="btn-neon"
            >
              Browse Jobs
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {submissions.slice(0, 10).map((submission) => (
              <div
                key={submission.submission_id}
                className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer"
                onClick={() => navigate(`/jobs/${submission.job_id}`)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        submission.status === 'scored'
                          ? 'bg-neon-green'
                          : submission.status === 'rejected'
                          ? 'bg-red-500'
                          : 'bg-yellow-500'
                      }`}
                    ></div>

                    <div>
                      <div className="font-semibold text-white mb-1">
                        Job #{submission.job_id.slice(0, 8)}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        <span>
                          {new Date(submission.timestamp).toLocaleDateString()}
                        </span>
                        <span>•</span>
                        <span>{submission.time_taken}s</span>
                        {submission.status === 'scored' && (
                          <>
                            <span>•</span>
                            <span className="text-neon-green">
                              Score: {(submission.score * 100).toFixed(0)}%
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {submission.status === 'scored' && (
                    <div className="text-right">
                      <div className="text-2xl font-bold text-neon-green">
                        +{submission.payout} JC
                      </div>
                    </div>
                  )}

                  {submission.status === 'rejected' && (
                    <span className="badge-error">Rejected</span>
                  )}

                  {submission.status === 'pending' && (
                    <span className="badge-warning">Evaluating...</span>
                  )}
                </div>
              </div>
            ))}

            {submissions.length > 10 && (
              <div className="text-center pt-4">
                <button className="text-neon-green hover:text-neon-green/80 text-sm">
                  View all {submissions.length} submissions →
                </button>
              </div>
            )}
          </div>
        )}
      </motion.div>

      {/* Activity Chart Placeholder */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-panel p-6 mt-8"
      >
        <h2 className="text-xl font-bold text-white mb-4">Activity Overview</h2>
        <div className="bg-gray-900 rounded-lg p-8 border border-gray-800 text-center">
          <p className="text-gray-500">Activity chart coming soon...</p>
          <p className="text-sm text-gray-600 mt-2">
            Track your daily submissions, earnings, and score trends
          </p>
        </div>
      </motion.div>
    </div>
  )
}
