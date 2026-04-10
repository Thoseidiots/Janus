import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { leaderboardAPI } from '../api/client'
import { useAuthStore } from '../store/authStore'

export default function LeaderboardPage() {
  const { user } = useAuthStore()
  const [limit, setLimit] = useState(100)

  const { data: leaderboard, isLoading } = useQuery({
    queryKey: ['leaderboard', limit],
    queryFn: () => leaderboardAPI.get(limit),
  })

  const userRank = leaderboard?.find((entry) => entry.node_id === user?.id)

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
            Leaderboard
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Top earners competing on the Janus mesh
        </p>
      </motion.div>

      {/* User's Rank Card */}
      {user && userRank && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass-panel p-6 mb-8 border-2 border-neon-green/30"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-4xl font-bold text-neon-green">
                  #{userRank.rank}
                </div>
                <div className="text-xs text-gray-500 mt-1">Your Rank</div>
              </div>
              <div className="h-16 w-px bg-gray-700"></div>
              <div>
                <div className="text-xl font-bold text-white mb-1">
                  {userRank.username}
                </div>
                <div className="text-sm text-gray-400">
                  {userRank.tasks_completed} tasks completed
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-neon-green mb-1">
                {userRank.total_earned.toLocaleString()} JC
              </div>
              <div className="text-sm text-gray-400">
                Avg Score: {(userRank.avg_score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex items-center justify-between mb-6"
      >
        <div className="flex items-center gap-4">
          <span className="text-gray-400">Show:</span>
          <button
            onClick={() => setLimit(10)}
            className={`px-4 py-2 rounded-lg transition-all ${
              limit === 10
                ? 'bg-neon-green text-gray-950 font-semibold'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Top 10
          </button>
          <button
            onClick={() => setLimit(50)}
            className={`px-4 py-2 rounded-lg transition-all ${
              limit === 50
                ? 'bg-neon-green text-gray-950 font-semibold'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Top 50
          </button>
          <button
            onClick={() => setLimit(100)}
            className={`px-4 py-2 rounded-lg transition-all ${
              limit === 100
                ? 'bg-neon-green text-gray-950 font-semibold'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Top 100
          </button>
        </div>

        <div className="text-sm text-gray-500">
          {leaderboard?.length || 0} entries
        </div>
      </motion.div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green mx-auto mb-4"></div>
            <p className="text-gray-400">Loading leaderboard...</p>
          </div>
        </div>
      )}

      {/* Leaderboard Table */}
      {!isLoading && leaderboard && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-panel overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">
                    Rank
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">
                    User
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">
                    Total Earned
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">
                    Avg Score
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">
                    Tasks
                  </th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">
                    Reputation
                  </th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((entry, index) => (
                  <motion.tr
                    key={entry.node_id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.02 }}
                    className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
                      entry.node_id === user?.id ? 'bg-neon-green/5' : ''
                    }`}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        {entry.rank <= 3 ? (
                          <div
                            className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                              entry.rank === 1
                                ? 'bg-yellow-500 text-gray-950'
                                : entry.rank === 2
                                ? 'bg-gray-300 text-gray-950'
                                : 'bg-orange-600 text-white'
                            }`}
                          >
                            {entry.rank}
                          </div>
                        ) : (
                          <div className="w-8 h-8 flex items-center justify-center text-gray-400 font-semibold">
                            {entry.rank}
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-semibold text-white flex items-center gap-2">
                          {entry.username}
                          {entry.node_id === user?.id && (
                            <span className="badge-info text-xs">You</span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500">
                          {entry.node_id.slice(0, 12)}...
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="font-bold text-neon-green text-lg">
                        {entry.total_earned.toLocaleString()} JC
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-24 h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-neon-green to-neon-blue"
                            style={{ width: `${entry.avg_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-white font-semibold w-12 text-right">
                          {(entry.avg_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span className="text-white font-semibold">
                        {entry.tasks_completed}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <span className="text-white font-semibold">
                          {entry.reputation}
                        </span>
                        <div
                          className={`w-2 h-2 rounded-full ${
                            entry.reputation >= 1000
                              ? 'bg-neon-green'
                              : entry.reputation >= 500
                              ? 'bg-neon-blue'
                              : 'bg-gray-600'
                          }`}
                        ></div>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {/* Empty State */}
      {!isLoading && (!leaderboard || leaderboard.length === 0) && (
        <div className="glass-panel p-12 text-center">
          <div className="text-6xl mb-4">🏆</div>
          <h3 className="text-xl font-bold text-white mb-2">
            No entries yet
          </h3>
          <p className="text-gray-400">
            Be the first to earn JC and claim the top spot!
          </p>
        </div>
      )}

      {/* Stats Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="grid grid-cols-3 gap-6 mt-8"
      >
        <div className="glass-panel p-6 text-center">
          <div className="text-3xl font-bold text-neon-green mb-2">
            {leaderboard?.[0]?.total_earned.toLocaleString() || '0'} JC
          </div>
          <div className="text-sm text-gray-400">Top Earner</div>
        </div>
        <div className="glass-panel p-6 text-center">
          <div className="text-3xl font-bold text-neon-blue mb-2">
            {leaderboard
              ? (
                  leaderboard.reduce((sum, e) => sum + e.avg_score, 0) /
                  leaderboard.length *
                  100
                ).toFixed(1)
              : '0'}
            %
          </div>
          <div className="text-sm text-gray-400">Avg Quality Score</div>
        </div>
        <div className="glass-panel p-6 text-center">
          <div className="text-3xl font-bold text-neon-pink mb-2">
            {leaderboard?.reduce((sum, e) => sum + e.tasks_completed, 0) || 0}
          </div>
          <div className="text-sm text-gray-400">Total Tasks Completed</div>
        </div>
      </motion.div>
    </div>
  )
}
