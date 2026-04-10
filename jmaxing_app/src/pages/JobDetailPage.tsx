import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import Editor from '@monaco-editor/react'
import { jobsAPI, submissionsAPI } from '../api/client'
import { useAuthStore } from '../store/authStore'

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [code, setCode] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const { data: job, isLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => jobsAPI.get(jobId!),
    enabled: !!jobId,
  })

  const { data: submissions } = useQuery({
    queryKey: ['submissions', jobId],
    queryFn: () => submissionsAPI.getForJob(jobId!),
    enabled: !!jobId,
  })

  const submitMutation = useMutation({
    mutationFn: () => jobsAPI.submit(jobId!, code),
    onSuccess: (data) => {
      setIsSubmitting(false)
      alert(
        `Solution submitted! Score: ${data.score.toFixed(2)} | Payout: ${data.payout} JC`
      )
      navigate('/jobs')
    },
    onError: () => {
      setIsSubmitting(false)
      alert('Submission failed. Please try again.')
    },
  })

  const handleSubmit = () => {
    if (!code.trim()) {
      alert('Please write some code before submitting')
      return
    }
    if (!user) {
      alert('Please login to submit solutions')
      navigate('/login')
      return
    }
    setIsSubmitting(true)
    submitMutation.mutate()
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green mx-auto mb-4"></div>
          <p className="text-gray-400">Loading job...</p>
        </div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Job not found</h2>
          <button
            onClick={() => navigate('/jobs')}
            className="btn-primary"
          >
            Back to Jobs
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <button
          onClick={() => navigate('/jobs')}
          className="text-neon-green hover:text-neon-green/80 mb-4 flex items-center gap-2"
        >
          <span>←</span> Back to Jobs
        </button>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              Job #{job.job_id.slice(0, 8)}
            </h1>
            <div className="flex items-center gap-4">
              <span className={`badge-${job.status === 'open' ? 'success' : 'info'}`}>
                {job.status}
              </span>
              <span className="text-gray-400">
                Difficulty: {job.difficulty.toFixed(1)}/5
              </span>
              <span className="text-gray-400">
                {submissions?.length || 0} submissions
              </span>
            </div>
          </div>

          <div className="glass-panel p-4 text-center">
            <div className="text-3xl font-bold text-neon-green mb-1">
              {job.reward_jc} JC
            </div>
            <div className="text-sm text-gray-400">Reward</div>
            {job.best_score && (
              <div className="text-xs text-gray-500 mt-2">
                Best: {(job.best_score * 100).toFixed(0)}%
              </div>
            )}
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Original Code + Tests */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="glass-panel p-6 mb-6">
            <h2 className="text-xl font-bold text-white mb-4">
              Original Code
            </h2>
            <div className="bg-gray-950 rounded-lg overflow-hidden border border-gray-800">
              <Editor
                height="300px"
                language="python"
                value={job.input_code}
                theme="vs-dark"
                options={{
                  readOnly: true,
                  minimap: { enabled: false },
                  fontSize: 14,
                  scrollBeyondLastLine: false,
                }}
              />
            </div>
          </div>

          <div className="glass-panel p-6">
            <h2 className="text-xl font-bold text-white mb-4">
              Test Cases ({job.tests.length})
            </h2>
            <div className="space-y-3">
              {job.tests.map((test, idx) => (
                <div
                  key={idx}
                  className="bg-gray-900 rounded-lg p-4 border border-gray-800"
                >
                  <div className="text-sm text-gray-400 mb-2">
                    Test {idx + 1}
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Input</div>
                      <code className="text-neon-blue text-sm">
                        {JSON.stringify(test.input)}
                      </code>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Expected</div>
                      <code className="text-neon-green text-sm">
                        {JSON.stringify(test.expected)}
                      </code>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Right Column: Solution Editor */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="glass-panel p-6">
            <h2 className="text-xl font-bold text-white mb-4">
              Your Solution
            </h2>

            {!user && (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-4">
                <p className="text-yellow-400 text-sm">
                  ⚠️ You must be logged in to submit solutions
                </p>
              </div>
            )}

            <div className="bg-gray-950 rounded-lg overflow-hidden border border-gray-800 mb-4">
              <Editor
                height="400px"
                language="python"
                value={code}
                onChange={(value) => setCode(value || '')}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  scrollBeyondLastLine: false,
                  tabSize: 2,
                }}
              />
            </div>

            {/* Scoring Info */}
            <div className="bg-gray-900 rounded-lg p-4 mb-4 border border-gray-800">
              <h3 className="text-sm font-semibold text-white mb-3">
                Scoring Algorithm
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Correctness (40%)</span>
                  <span className="text-gray-300">All tests must pass</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Quality (30%)</span>
                  <span className="text-gray-300">Static analysis</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Improvement (30%)</span>
                  <span className="text-gray-300">Better than original</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-800">
                <h4 className="text-xs font-semibold text-white mb-2">
                  Payout Tiers
                </h4>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500">&lt; 0.3</span>
                    <span className="text-red-400">0 JC (rejected)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">0.3 - 0.7</span>
                    <span className="text-yellow-400">50% reward</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">0.7 - 0.9</span>
                    <span className="text-green-400">100% reward</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">&gt; 0.9</span>
                    <span className="text-neon-green">150% bonus!</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !user || job.status !== 'open'}
              className="btn-neon w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Evaluating...
                </span>
              ) : (
                `Submit Solution`
              )}
            </button>

            {job.status !== 'open' && (
              <p className="text-center text-gray-500 text-sm mt-2">
                This job is {job.status}
              </p>
            )}
          </div>

          {/* Recent Submissions */}
          {submissions && submissions.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-panel p-6 mt-6"
            >
              <h3 className="text-lg font-bold text-white mb-4">
                Recent Submissions
              </h3>
              <div className="space-y-2">
                {submissions.slice(0, 5).map((sub) => (
                  <div
                    key={sub.submission_id}
                    className="flex items-center justify-between bg-gray-900 rounded-lg p-3 border border-gray-800"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-gray-400 text-sm">
                        {sub.node_id.slice(0, 8)}
                      </span>
                      <span
                        className={`badge-${
                          sub.status === 'scored'
                            ? 'success'
                            : sub.status === 'rejected'
                            ? 'error'
                            : 'warning'
                        }`}
                      >
                        {sub.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      {sub.status === 'scored' && (
                        <>
                          <span className="text-sm text-gray-400">
                            Score: {(sub.score * 100).toFixed(0)}%
                          </span>
                          <span className="text-neon-green font-semibold">
                            +{sub.payout} JC
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
