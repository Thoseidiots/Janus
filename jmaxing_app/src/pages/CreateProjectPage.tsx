import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tantml:function_calls>
import { motion } from 'framer-motion'
import { projectsAPI, type ProjectType } from '../api/projects'
import { useAuthStore } from '../store/authStore'
import { Users, Lock, Star, GitBranch } from 'lucide-react'

export default function CreateProjectPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()

  const [projectType, setProjectType] = useState<ProjectType>('community')
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    repository: '',
    language: 'Python',
    difficulty: 3,
    tags: '',
    rewardPool: '',
    rewardDistribution: 'contribution-based' as const,
    isPublic: true,
    allowJoin: true,
    deadline: '',
  })

  const createMutation = useMutation({
    mutationFn: () =>
      projectsAPI.createProject({
        ...formData,
        type: projectType,
        tags: formData.tags.split(',').map((t) => t.trim()).filter(Boolean),
        rewardPool: formData.rewardPool ? parseInt(formData.rewardPool) : undefined,
        deadline: formData.deadline ? new Date(formData.deadline).getTime() : undefined,
      }),
    onSuccess: (data) => {
      navigate(`/projects/${data.id}`)
    },
  })

  const handleSubmit = () => {
    if (!formData.name.trim() || !formData.description.trim()) {
      alert('Please fill in required fields')
      return
    }
    createMutation.mutate()
  }

  if (!user) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Login Required</h2>
        <p className="text-gray-400 mb-6">
          You must be logged in to create a project
        </p>
        <button onClick={() => navigate('/login')} className="btn-neon">
          Login
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
            Create New Project
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Start a collaborative project and earn JC together
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Project Type Selection */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">Project Type</h2>

            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  type: 'community' as ProjectType,
                  icon: Users,
                  label: 'Community',
                  description: 'Open to everyone',
                  color: 'neon-green',
                },
                {
                  type: 'group' as ProjectType,
                  icon: Lock,
                  label: 'Group',
                  description: 'Invite-only team',
                  color: 'neon-blue',
                },
                {
                  type: 'solo' as ProjectType,
                  icon: Star,
                  label: 'Solo',
                  description: 'Work alone',
                  color: 'neon-pink',
                },
              ].map(({ type, icon: Icon, label, description, color }) => (
                <button
                  key={type}
                  onClick={() => setProjectType(type)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    projectType === type
                      ? `border-${color} bg-${color}/5`
                      : 'border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <Icon
                    className={`w-8 h-8 mx-auto mb-2 ${
                      projectType === type ? `text-${color}` : 'text-gray-500'
                    }`}
                  />
                  <div className="text-sm font-semibold text-white">
                    {label}
                  </div>
                  <div className="text-xs text-gray-500">{description}</div>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Basic Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">Basic Information</h2>

            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Project Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  placeholder="My Awesome Project"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Description *
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  placeholder="Describe what your project is about..."
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors resize-none"
                  rows={4}
                />
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Repository URL (optional)
                </label>
                <input
                  type="url"
                  value={formData.repository}
                  onChange={(e) =>
                    setFormData({ ...formData, repository: e.target.value })
                  }
                  placeholder="https://github.com/username/repo"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Primary Language
                  </label>
                  <select
                    value={formData.language}
                    onChange={(e) =>
                      setFormData({ ...formData, language: e.target.value })
                    }
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                  >
                    <option value="Python">Python</option>
                    <option value="JavaScript">JavaScript</option>
                    <option value="TypeScript">TypeScript</option>
                    <option value="Rust">Rust</option>
                    <option value="Go">Go</option>
                    <option value="Java">Java</option>
                    <option value="C++">C++</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Difficulty (1-5)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={formData.difficulty}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        difficulty: parseInt(e.target.value),
                      })
                    }
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Tags (comma-separated)
                </label>
                <input
                  type="text"
                  value={formData.tags}
                  onChange={(e) =>
                    setFormData({ ...formData, tags: e.target.value })
                  }
                  placeholder="web3, defi, nft"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-2 block">
                  Deadline (optional)
                </label>
                <input
                  type="date"
                  value={formData.deadline}
                  onChange={(e) =>
                    setFormData({ ...formData, deadline: e.target.value })
                  }
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                />
              </div>
            </div>
          </motion.div>

          {/* Rewards (Community Projects) */}
          {projectType === 'community' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-panel p-6"
            >
              <h2 className="text-xl font-bold text-white mb-4">
                Reward Pool (Optional)
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Total JC Reward Pool
                  </label>
                  <input
                    type="number"
                    min="0"
                    value={formData.rewardPool}
                    onChange={(e) =>
                      setFormData({ ...formData, rewardPool: e.target.value })
                    }
                    placeholder="1000"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Your balance: {user.balance} JC
                  </p>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Distribution Method
                  </label>
                  <select
                    value={formData.rewardDistribution}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        rewardDistribution: e.target.value as any,
                      })
                    }
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                  >
                    <option value="equal">Equal - Split evenly among all contributors</option>
                    <option value="contribution-based">
                      Contribution-Based - Based on lines of code changed
                    </option>
                    <option value="merit-based">
                      Merit-Based - Manual approval by project owner
                    </option>
                  </select>
                </div>
              </div>
            </motion.div>
          )}

          {/* Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">Settings</h2>

            <div className="space-y-4">
              <label className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-white">Public Project</div>
                  <div className="text-sm text-gray-400">
                    Anyone can view this project
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={formData.isPublic}
                  onChange={(e) =>
                    setFormData({ ...formData, isPublic: e.target.checked })
                  }
                  className="w-5 h-5"
                />
              </label>

              {projectType === 'community' && (
                <label className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold text-white">Allow Anyone to Join</div>
                    <div className="text-sm text-gray-400">
                      Users can join without approval
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={formData.allowJoin}
                    onChange={(e) =>
                      setFormData({ ...formData, allowJoin: e.target.checked })
                    }
                    className="w-5 h-5"
                  />
                </label>
              )}
            </div>
          </motion.div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={createMutation.isPending}
            className="btn-neon w-full text-lg py-4"
          >
            {createMutation.isPending ? 'Creating...' : 'Create Project'}
          </button>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">Quick Tips</h3>
            <div className="space-y-3 text-sm text-gray-400">
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Choose a clear, descriptive project name</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Add relevant tags for discoverability</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Link your GitHub repo if you have one</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Set realistic deadlines and difficulty</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6 border-l-4 border-neon-blue"
          >
            <h3 className="text-lg font-bold text-white mb-3">
              Project Type Guide
            </h3>
            <div className="space-y-4 text-sm text-gray-400">
              <div>
                <div className="font-semibold text-neon-green mb-1">Community</div>
                <p>Best for open-source projects where anyone can contribute</p>
              </div>
              <div>
                <div className="font-semibold text-neon-blue mb-1">Group</div>
                <p>Perfect for team projects with invited collaborators</p>
              </div>
              <div>
                <div className="font-semibold text-neon-pink mb-1">Solo</div>
                <p>Ideal for personal projects you want to track</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
