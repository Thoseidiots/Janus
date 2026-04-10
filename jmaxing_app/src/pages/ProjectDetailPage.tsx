import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { projectsAPI } from '../api/projects'
import { useAuthStore } from '../store/authStore'
import {
  Users,
  Star,
  GitBranch,
  MessageCircle,
  Settings,
  UserPlus,
  LogOut,
  Award,
  Send,
  Code2,
} from 'lucide-react'

export default function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'overview' | 'members' | 'contributions' | 'chat'>('overview')
  const [chatMessage, setChatMessage] = useState('')

  // Fetch project
  const { data: project, isLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsAPI.getProject(projectId!),
    enabled: !!projectId,
  })

  // Fetch members
  const { data: members } = useQuery({
    queryKey: ['projectMembers', projectId],
    queryFn: () => projectsAPI.getMembers(projectId!),
    enabled: !!projectId && activeTab === 'members',
  })

  // Fetch contributions
  const { data: contributions } = useQuery({
    queryKey: ['projectContributions', projectId],
    queryFn: () => projectsAPI.getContributions(projectId!),
    enabled: !!projectId && activeTab === 'contributions',
  })

  // Fetch chat
  const { data: messages } = useQuery({
    queryKey: ['projectChat', projectId],
    queryFn: () => projectsAPI.getMessages(projectId!),
    enabled: !!projectId && activeTab === 'chat' && (project?.type === 'group' || project?.isMember),
    refetchInterval: 3000, // Poll every 3s for new messages
  })

  // Join mutation
  const joinMutation = useMutation({
    mutationFn: () => projectsAPI.joinProject(projectId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })
      queryClient.invalidateQueries({ queryKey: ['projectMembers', projectId] })
    },
  })

  // Leave mutation
  const leaveMutation = useMutation({
    mutationFn: () => projectsAPI.leaveProject(projectId!),
    onSuccess: () => {
      navigate('/projects')
    },
  })

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: (content: string) => projectsAPI.sendMessage(projectId!, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projectChat', projectId] })
      setChatMessage('')
    },
  })

  // Star mutation
  const starMutation = useMutation({
    mutationFn: () => projectsAPI.starProject(projectId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })
    },
  })

  const handleJoin = () => {
    if (project?.type === 'community' && project?.allowJoin) {
      joinMutation.mutate()
    }
  }

  const handleSendMessage = () => {
    if (chatMessage.trim()) {
      sendMessageMutation.mutate(chatMessage)
    }
  }

  if (isLoading || !project) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green"></div>
      </div>
    )
  }

  const isOwner = user?.id === project.ownerId
  const isMember = project.isMember
  const canJoin = project.type === 'community' && project.allowJoin && !isMember

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel p-8 mb-8"
      >
        <div className="flex items-start justify-between mb-6">
          <div className="flex-1">
            <div className="flex items-center gap-4 mb-4">
              <h1 className="text-4xl font-bold text-white">{project.name}</h1>
              <span className={`badge-${project.type === 'community' ? 'success' : project.type === 'group' ? 'info' : 'warning'}`}>
                {project.type}
              </span>
              <span className={`badge-${project.status === 'active' ? 'success' : 'info'}`}>
                {project.status}
              </span>
            </div>

            <p className="text-gray-400 mb-4">{project.description}</p>

            {/* Tags */}
            {project.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-4">
                {project.tags.map((tag) => (
                  <span
                    key={tag}
                    className="text-sm px-3 py-1 bg-gray-800 text-neon-green rounded-lg"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            {/* Owner */}
            <div className="text-sm text-gray-500">
              Created by{' '}
              <button
                onClick={() => navigate(`/users/${project.ownerId}`)}
                className="text-neon-blue hover:text-neon-blue/80"
              >
                @{project.ownerUsername}
              </button>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => starMutation.mutate()}
              className="btn-secondary flex items-center gap-2"
            >
              <Star className="w-4 h-4" />
              {project.stars}
            </button>

            {canJoin && (
              <button
                onClick={handleJoin}
                disabled={joinMutation.isPending}
                className="btn-neon"
              >
                {joinMutation.isPending ? 'Joining...' : 'Join Project'}
              </button>
            )}

            {isMember && !isOwner && (
              <button
                onClick={() => leaveMutation.mutate()}
                className="btn-secondary text-red-400 hover:text-red-300"
              >
                <LogOut className="w-4 h-4" />
              </button>
            )}

            {isOwner && (
              <button
                onClick={() => navigate(`/projects/${projectId}/settings`)}
                className="btn-secondary"
              >
                <Settings className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-green mb-1">
              {project.members}
            </div>
            <div className="text-xs text-gray-400">Members</div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-neon-blue mb-1">
              {project.contributions}
            </div>
            <div className="text-xs text-gray-400">Contributions</div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white mb-1">
              {project.language}
            </div>
            <div className="text-xs text-gray-400">Language</div>
          </div>

          {project.rewardPool && (
            <div className="bg-gray-900 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-neon-pink mb-1">
                {project.rewardPool} JC
              </div>
              <div className="text-xs text-gray-400">Reward Pool</div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {[
          { key: 'overview', label: 'Overview', icon: Code2 },
          { key: 'members', label: 'Members', icon: Users },
          { key: 'contributions', label: 'Contributions', icon: GitBranch },
          ...(isMember || project.type === 'community'
            ? [{ key: 'chat', label: 'Chat', icon: MessageCircle }]
            : []),
        ].map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
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

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6">Project Details</h2>

          <div className="space-y-6">
            {project.repository && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Repository</h3>
                <a
                  href={project.repository}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-neon-blue hover:text-neon-blue/80"
                >
                  {project.repository}
                </a>
              </div>
            )}

            <div>
              <h3 className="text-lg font-semibold text-white mb-2">
                Difficulty Level
              </h3>
              <div className="flex items-center gap-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Star
                    key={i}
                    className={`w-5 h-5 ${
                      i < project.difficulty ? 'text-neon-green' : 'text-gray-700'
                    }`}
                    fill={i < project.difficulty ? 'currentColor' : 'none'}
                  />
                ))}
              </div>
            </div>

            {project.rewardDistribution && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Reward Distribution
                </h3>
                <p className="text-gray-400 capitalize">
                  {project.rewardDistribution.replace(/-/g, ' ')}
                </p>
              </div>
            )}

            {project.deadline && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Deadline</h3>
                <p className="text-gray-400">
                  {new Date(project.deadline).toLocaleDateString()}
                </p>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {activeTab === 'members' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">
              Members ({members?.length || 0})
            </h2>
            {isOwner && project.type !== 'solo' && (
              <button className="btn-neon flex items-center gap-2">
                <UserPlus className="w-4 h-4" />
                Invite Members
              </button>
            )}
          </div>

          <div className="space-y-3">
            {members?.map((member) => (
              <div
                key={member.id}
                className="bg-gray-900 rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neon-green to-neon-blue flex items-center justify-center text-xl font-bold text-gray-950">
                    {member.username.slice(0, 2).toUpperCase()}
                  </div>
                  <div>
                    <div className="font-semibold text-white">
                      {member.username}
                    </div>
                    <div className="text-xs text-gray-500 capitalize">
                      {member.role}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6 text-sm">
                  <div className="text-center">
                    <div className="font-bold text-neon-green">
                      {member.contributions}
                    </div>
                    <div className="text-xs text-gray-500">Contributions</div>
                  </div>
                  <div className="text-center">
                    <div className="font-bold text-neon-blue">
                      {member.earnedJC} JC
                    </div>
                    <div className="text-xs text-gray-500">Earned</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {activeTab === 'contributions' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6">
            Recent Contributions
          </h2>

          <div className="space-y-4">
            {contributions?.map((contribution) => (
              <div
                key={contribution.id}
                className="bg-gray-900 rounded-lg p-4 border border-gray-800"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-white mb-1">
                      {contribution.title}
                    </h3>
                    <p className="text-sm text-gray-400">
                      by {contribution.username} •{' '}
                      {new Date(contribution.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                  <span className={`badge-${contribution.status === 'approved' ? 'success' : contribution.status === 'rejected' ? 'error' : 'warning'}`}>
                    {contribution.status}
                  </span>
                </div>

                <p className="text-gray-400 text-sm mb-3">
                  {contribution.description}
                </p>

                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <span className="text-green-400">
                    +{contribution.linesAdded}
                  </span>
                  <span className="text-red-400">
                    -{contribution.linesRemoved}
                  </span>
                  {contribution.reward && (
                    <span className="text-neon-green font-bold">
                      +{contribution.reward} JC
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {activeTab === 'chat' && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-6 h-[600px] flex flex-col"
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto mb-4 space-y-3">
            {messages?.map((message) => (
              <div key={message.id} className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-neon-pink to-neon-blue flex items-center justify-center text-sm font-bold text-gray-950 flex-shrink-0">
                  {message.username.slice(0, 2).toUpperCase()}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-white text-sm">
                      {message.username}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm">{message.content}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Input */}
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type a message..."
              className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-neon-green transition-colors"
            />
            <button
              onClick={handleSendMessage}
              disabled={!chatMessage.trim() || sendMessageMutation.isPending}
              className="btn-neon px-6 py-3"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </motion.div>
      )}
    </div>
  )
}
