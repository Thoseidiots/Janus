import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Code2, Trophy, Clock, TrendingUp, Filter, Search } from 'lucide-react'

// Mock data - replace with API calls
const mockJobs = [
  {
    job_id: '1',
    input_code: 'def process(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result',
    tests: [{ input: [1, 2, 3], expected: [2, 4, 6] }],
    reward_jc: 150,
    difficulty: 2.5,
    created_at: Date.now() - 3600000,
    status: 'open' as const,
    submissions: 5,
    best_score: 0.82,
  },
  {
    job_id: '2',
    input_code: 'function calculate(x, y) {\n  return x + y;\n}',
    tests: [{ input: [2, 3], expected: 5 }],
    reward_jc: 200,
    difficulty: 3.0,
    created_at: Date.now() - 7200000,
    status: 'open' as const,
    submissions: 12,
    best_score: 0.91,
  },
]

export default function JobBoardPage() {
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty < 2) return 'text-green-400'
    if (difficulty < 3.5) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getDifficultyLabel = (difficulty: number) => {
    if (difficulty < 2) return 'Easy'
    if (difficulty < 3.5) return 'Medium'
    return 'Hard'
  }

  return (
    <div className="min-h-screen bg-gray-950 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl font-bold mb-4">
            Available <span className="neon-text">Jobs</span>
          </h1>
          <p className="text-xl text-gray-400">
            Browse code improvement tasks. Pick one, submit better code, earn JC.
          </p>
        </motion.div>

        {/* Filters */}
        <div className="glass-panel p-6 mb-8">
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search jobs..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:outline-none focus:border-janus-500"
              />
            </div>

            {/* Difficulty filter */}
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:outline-none focus:border-janus-500"
            >
              <option value="all">All Difficulties</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>
        </div>

        {/* Job Cards */}
        <div className="space-y-4">
          {mockJobs.map((job, index) => (
            <motion.div
              key={job.job_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link to={`/jobs/${job.job_id}`} className="block job-card">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <Code2 className="w-5 h-5 text-janus-400" />
                      <h3 className="text-xl font-bold">Job #{job.job_id}</h3>
                      <span className={`badge ${getDifficultyColor(job.difficulty)}`}>
                        {getDifficultyLabel(job.difficulty)}
                      </span>
                    </div>

                    <div className="flex items-center space-x-6 text-sm text-gray-400">
                      <div className="flex items-center space-x-1">
                        <Trophy className="w-4 h-4" />
                        <span>{job.reward_jc} JC</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="w-4 h-4" />
                        <span>{new Date(job.created_at).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <TrendingUp className="w-4 h-4" />
                        <span>{job.submissions} submissions</span>
                      </div>
                      {job.best_score && (
                        <div className="flex items-center space-x-1">
                          <span className="text-neon-green">
                            Best: {(job.best_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-3xl font-bold text-neon-green mb-1">
                      {job.reward_jc}
                    </div>
                    <div className="text-xs text-gray-400">JC Reward</div>
                  </div>
                </div>

                {/* Code Preview */}
                <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-x-auto">
                  <pre className="text-gray-300">
                    {job.input_code.slice(0, 200)}
                    {job.input_code.length > 200 && '...'}
                  </pre>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Empty State */}
        {mockJobs.length === 0 && (
          <div className="text-center py-16">
            <Code2 className="w-16 h-16 mx-auto mb-4 text-gray-600" />
            <h3 className="text-2xl font-bold mb-2 text-gray-400">No jobs found</h3>
            <p className="text-gray-500">Try adjusting your filters</p>
          </div>
        )}
      </div>
    </div>
  )
}
