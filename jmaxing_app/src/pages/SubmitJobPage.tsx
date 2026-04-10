import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import Editor from '@monaco-editor/react'
import { jobsAPI } from '../api/client'
import { useAuthStore } from '../store/authStore'

interface TestCase {
  input: string
  expected: string
}

export default function SubmitJobPage() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [code, setCode] = useState('')
  const [reward, setReward] = useState('100')
  const [difficulty, setDifficulty] = useState('2.5')
  const [tests, setTests] = useState<TestCase[]>([
    { input: '', expected: '' },
  ])

  const createJobMutation = useMutation({
    mutationFn: () => {
      const parsedTests = tests
        .filter((t) => t.input.trim() && t.expected.trim())
        .map((t) => ({
          input: JSON.parse(t.input),
          expected: JSON.parse(t.expected),
        }))

      return jobsAPI.create({
        input_code: code,
        tests: parsedTests,
        reward_jc: parseInt(reward),
        difficulty: parseFloat(difficulty),
      })
    },
    onSuccess: (data) => {
      alert(`Job created successfully! Job ID: ${data.job_id}`)
      navigate('/jobs')
    },
    onError: (error: any) => {
      alert(`Failed to create job: ${error.message}`)
    },
  })

  const addTestCase = () => {
    setTests([...tests, { input: '', expected: '' }])
  }

  const removeTestCase = (index: number) => {
    setTests(tests.filter((_, i) => i !== index))
  }

  const updateTestCase = (index: number, field: 'input' | 'expected', value: string) => {
    const newTests = [...tests]
    newTests[index][field] = value
    setTests(newTests)
  }

  const handleSubmit = () => {
    if (!code.trim()) {
      alert('Please provide the code that needs improvement')
      return
    }

    if (tests.filter((t) => t.input.trim() && t.expected.trim()).length === 0) {
      alert('Please provide at least one test case')
      return
    }

    if (!reward || parseInt(reward) <= 0) {
      alert('Please provide a valid reward amount')
      return
    }

    createJobMutation.mutate()
  }

  if (!user) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">
          Login Required
        </h2>
        <p className="text-gray-400 mb-6">
          You must be logged in to submit jobs
        </p>
        <button
          onClick={() => navigate('/login')}
          className="btn-neon"
        >
          Go to Login
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
            Submit a Job
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Post code that needs improvement and offer JC rewards
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Code Editor */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">
              Code to Improve
            </h2>
            <p className="text-sm text-gray-400 mb-4">
              Paste the code that needs optimization, refactoring, or quality improvements
            </p>
            <div className="bg-gray-950 rounded-lg overflow-hidden border border-gray-800">
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
          </motion.div>

          {/* Test Cases */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white">Test Cases</h2>
              <button
                onClick={addTestCase}
                className="text-neon-green hover:text-neon-green/80 text-sm font-semibold"
              >
                + Add Test
              </button>
            </div>

            <p className="text-sm text-gray-400 mb-4">
              Define test cases to verify correctness. Use JSON format.
            </p>

            <div className="space-y-4">
              {tests.map((test, index) => (
                <div
                  key={index}
                  className="bg-gray-900 rounded-lg p-4 border border-gray-800"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-white">
                      Test {index + 1}
                    </span>
                    {tests.length > 1 && (
                      <button
                        onClick={() => removeTestCase(index)}
                        className="text-red-400 hover:text-red-300 text-sm"
                      >
                        Remove
                      </button>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs text-gray-500 mb-2 block">
                        Input (JSON)
                      </label>
                      <textarea
                        value={test.input}
                        onChange={(e) =>
                          updateTestCase(index, 'input', e.target.value)
                        }
                        placeholder='[1, 2, 3]'
                        className="w-full bg-gray-950 border border-gray-800 rounded-lg p-3 text-neon-blue font-mono text-sm focus:outline-none focus:border-neon-green transition-colors"
                        rows={3}
                      />
                    </div>

                    <div>
                      <label className="text-xs text-gray-500 mb-2 block">
                        Expected Output (JSON)
                      </label>
                      <textarea
                        value={test.expected}
                        onChange={(e) =>
                          updateTestCase(index, 'expected', e.target.value)
                        }
                        placeholder='6'
                        className="w-full bg-gray-950 border border-gray-800 rounded-lg p-3 text-neon-green font-mono text-sm focus:outline-none focus:border-neon-green transition-colors"
                        rows={3}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Job Settings */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">
              Job Settings
            </h2>

            <div className="space-y-4">
              {/* Reward */}
              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Reward (JC)
                </label>
                <input
                  type="number"
                  value={reward}
                  onChange={(e) => setReward(e.target.value)}
                  min="10"
                  step="10"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
                <p className="text-xs text-gray-500 mt-2">
                  Minimum: 10 JC • Your balance: {user.balance} JC
                </p>
              </div>

              {/* Difficulty */}
              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Difficulty (1-5)
                </label>
                <input
                  type="number"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  min="1"
                  max="5"
                  step="0.1"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
                <div className="flex items-center gap-2 mt-2">
                  <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                      style={{ width: `${(parseFloat(difficulty) / 5) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 w-12">
                    {difficulty}/5
                  </span>
                </div>
              </div>

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={createJobMutation.isPending}
                className="btn-neon w-full text-lg py-4 mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {createJobMutation.isPending ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Creating...
                  </span>
                ) : (
                  `Submit Job (${reward} JC)`
                )}
              </button>
            </div>
          </motion.div>

          {/* Info Box */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-6 border-l-4 border-neon-blue"
          >
            <h3 className="text-sm font-bold text-white mb-3">
              How It Works
            </h3>
            <div className="space-y-3 text-xs text-gray-400">
              <div className="flex gap-3">
                <span className="text-neon-green font-bold">1.</span>
                <span>
                  Your job is posted to the marketplace and visible to all developers
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green font-bold">2.</span>
                <span>
                  Developers submit improved versions of your code
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green font-bold">3.</span>
                <span>
                  Solutions are automatically evaluated based on correctness, quality, and improvement
                </span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green font-bold">4.</span>
                <span>
                  The best solution is delivered to you, and the developer is paid
                </span>
              </div>
            </div>
          </motion.div>

          {/* Tips */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-panel p-6"
          >
            <h3 className="text-sm font-bold text-white mb-3">
              💡 Tips for Better Results
            </h3>
            <ul className="space-y-2 text-xs text-gray-400 list-disc list-inside">
              <li>Provide clear, working code as a starting point</li>
              <li>Add comprehensive test cases covering edge cases</li>
              <li>Set appropriate rewards based on complexity</li>
              <li>Higher rewards attract more submissions</li>
              <li>Be specific about what improvements you want</li>
            </ul>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
