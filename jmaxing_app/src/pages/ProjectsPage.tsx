import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { projectsAPI, type Project, type ProjectType } from '../api/projects'
import { useAuthStore } from '../store/authStore'
import {
  Users,
  Star,
  GitBranch,
  Lock,
  Globe,
  PlusCircle,
  Search,
  Filter,
} from 'lucide-react'

export default function ProjectsPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<'all' | 'my' | 'community' | 'group' | 'solo'>('all')
  const [searchQuery, setSearchQuery] = useState('')

  // Fetch projects
  const { data: allProjects, isLoading } = useQuery({
    queryKey: ['projects', activeTab, searchQuery],
    queryFn: () => {
      if (activeTab === 'my') {
        return projectsAPI.getMyProjects()
      }
      const type = activeTab === 'all' ? undefined : (activeTab as ProjectType)
      return projectsAPI.listProjects({
        type,
        search: searchQuery || undefined,
      })
    },
  })

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
                Projects
              </span>
            </h1>
            <p className="text-gray-400 text-lg">
              Collaborate on code projects and earn JC together
            </p>
          </div>

          <button
            onClick={() => navigate('/projects/create')}
            className="btn-neon flex items-center gap-2"
          >
            <PlusCircle className="w-5 h-5" />
            Create Project
          </button>
        </div>

        {/* Search */}
        <div className="relative max-w-xl mb-6">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search projects..."
            className="w-full pl-12 pr-4 py-3 bg-gray-900 border border-gray-800 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-neon-green transition-colors"
          />
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-2 overflow-x-auto">
          {[
            { key: 'all', label: 'All Projects', icon: Globe },
            { key: 'my', label: 'My Projects', icon: Users },
            { key: 'community', label: 'Community', icon: Users },
            { key: 'group', label: 'Group', icon: Lock },
            { key: 'solo', label: 'Solo', icon: Star },
          ].map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap flex items-center gap-2 ${
                  activeTab === tab.key
                    ? 'bg-neon-green text-gray-950'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </div>
      </motion.div>

      {/* Project Grid */}
      {isLoading ? (
        <div className="flex justify-center py-20">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green"></div>
        </div>
      ) : !allProjects || allProjects.length === 0 ? (
        <div className="glass-panel p-16 text-center">
          <div className="text-6xl mb-4">📁</div>
          <h3 className="text-2xl font-bold text-white mb-2">
            {activeTab === 'my' ? 'No projects yet' : 'No projects found'}
          </h3>
          <p className="text-gray-400 mb-6">
            {activeTab === 'my'
              ? 'Create your first project or join an existing one'
              : 'Try adjusting your filters or search query'}
          </p>
          {activeTab === 'my' && (
            <button
              onClick={() => navigate('/projects/create')}
              className="btn-neon"
            >
              Create Project
            </button>
          )}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {allProjects.map((project, index) => (
            <ProjectCard key={project.id} project={project} index={index} />
          ))}
        </motion.div>
      )}

      {/* Info Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-panel p-8 mt-12"
      >
        <h2 className="text-2xl font-bold text-white mb-6">
          Project Types Explained
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 rounded-full bg-neon-green/10 flex items-center justify-center">
                <Users className="w-6 h-6 text-neon-green" />
              </div>
              <h3 className="text-lg font-bold text-white">Community</h3>
            </div>
            <p className="text-gray-400 text-sm">
              Open to everyone. Anyone can join and contribute. Rewards are
              distributed based on contribution quality and quantity.
            </p>
          </div>

          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 rounded-full bg-neon-blue/10 flex items-center justify-center">
                <Lock className="w-6 h-6 text-neon-blue" />
              </div>
              <h3 className="text-lg font-bold text-white">Group</h3>
            </div>
            <p className="text-gray-400 text-sm">
              Invite-only teams. Perfect for working with friends or trusted
              collaborators. Includes private group chat.
            </p>
          </div>

          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 rounded-full bg-neon-pink/10 flex items-center justify-center">
                <Star className="w-6 h-6 text-neon-pink" />
              </div>
              <h3 className="text-lg font-bold text-white">Solo</h3>
            </div>
            <p className="text-gray-400 text-sm">
              Work alone at your own pace. Track your progress and optionally
              showcase your work when finished.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

// Project Card Component
function ProjectCard({ project, index }: { project: Project; index: number }) {
  const navigate = useNavigate()

  const typeColors = {
    community: 'neon-green',
    group: 'neon-blue',
    solo: 'neon-pink',
  }

  const typeIcons = {
    community: Users,
    group: Lock,
    solo: Star,
  }

  const Icon = typeIcons[project.type]
  const color = typeColors[project.type]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      onClick={() => navigate(`/projects/${project.id}`)}
      className="glass-panel p-6 hover:border-neon-green/30 transition-all cursor-pointer group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg bg-${color}/10 flex items-center justify-center`}>
            <Icon className={`w-5 h-5 text-${color}`} />
          </div>
          <div>
            <h3 className="font-bold text-white group-hover:text-neon-green transition-colors line-clamp-1">
              {project.name}
            </h3>
            <p className="text-xs text-gray-500">by {project.ownerUsername}</p>
          </div>
        </div>

        {project.isPublic ? (
          <Globe className="w-4 h-4 text-gray-500" />
        ) : (
          <Lock className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {/* Description */}
      <p className="text-gray-400 text-sm mb-4 line-clamp-2">
        {project.description}
      </p>

      {/* Tags */}
      {project.tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {project.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded"
            >
              #{tag}
            </span>
          ))}
          {project.tags.length > 3 && (
            <span className="text-xs px-2 py-1 bg-gray-800 text-gray-400 rounded">
              +{project.tags.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-gray-500">
        <div className="flex items-center gap-1">
          <Users className="w-4 h-4" />
          <span>{project.members}</span>
        </div>
        <div className="flex items-center gap-1">
          <GitBranch className="w-4 h-4" />
          <span>{project.contributions}</span>
        </div>
        <div className="flex items-center gap-1">
          <Star className="w-4 h-4" />
          <span>{project.stars}</span>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 pt-4 border-t border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`badge-${project.status === 'active' ? 'success' : 'info'}`}>
            {project.status}
          </span>
          {project.language && (
            <span className="text-xs text-gray-500">{project.language}</span>
          )}
        </div>

        {project.rewardPool && (
          <div className="text-neon-green font-bold text-sm">
            {project.rewardPool} JC
          </div>
        )}
      </div>
    </motion.div>
  )
}
