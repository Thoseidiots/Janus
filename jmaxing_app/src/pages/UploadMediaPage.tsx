import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  mediaAPI,
  type MediaCategory,
  type MediaType,
  CATEGORY_INFO,
} from '../api/media'
import { useAuthStore } from '../store/authStore'
import {
  Upload,
  X,
  Sparkles,
  AlertCircle,
  Check,
  Image as ImageIcon,
  Film,
  Music,
  Play,
} from 'lucide-react'

export default function UploadMediaPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<MediaCategory | undefined>()
  const [aiSuggestion, setAiSuggestion] = useState<{
    category: MediaCategory
    confidence: number
    tags: string[]
  } | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: () => {
      if (!file || !title.trim()) {
        throw new Error('Please provide a file and title')
      }
      return mediaAPI.uploadMedia({
        file,
        title: title.trim(),
        description: description.trim() || undefined,
        category: selectedCategory,
        tags: tags
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean),
      })
    },
    onSuccess: (data) => {
      navigate(`/media/${data.id}`)
    },
  })

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile)

    // Create preview
    if (selectedFile.type.startsWith('image/') || selectedFile.type.startsWith('video/')) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(selectedFile)
    } else if (selectedFile.type.startsWith('audio/')) {
      setPreview(null) // No preview for audio
    }

    // Simulate AI analysis (in production, this would be an API call)
    setTimeout(() => {
      const mockAiSuggestion = getMockAiSuggestion(selectedFile)
      setAiSuggestion(mockAiSuggestion)
      if (!selectedCategory) {
        setSelectedCategory(mockAiSuggestion.category)
      }
    }, 1500)
  }

  const getMockAiSuggestion = (file: File): {
    category: MediaCategory
    confidence: number
    tags: string[]
  } => {
    // Mock AI categorization based on file type and name
    if (file.type.startsWith('video/')) {
      return {
        category: 'dances',
        confidence: 0.87,
        tags: ['dance', 'performance', 'trending'],
      }
    } else if (file.type.startsWith('audio/')) {
      return {
        category: 'music',
        confidence: 0.92,
        tags: ['music', 'audio', 'sound'],
      }
    } else if (file.name.toLowerCase().includes('meme')) {
      return {
        category: 'memes',
        confidence: 0.95,
        tags: ['meme', 'funny', 'viral'],
      }
    } else if (file.type.startsWith('image/')) {
      return {
        category: 'art',
        confidence: 0.78,
        tags: ['image', 'visual', 'creative'],
      }
    }
    return {
      category: 'other',
      confidence: 0.65,
      tags: ['general'],
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFileSelect(droppedFile)
    }
  }

  const handleSubmit = () => {
    uploadMutation.mutate()
  }

  if (!user) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Login Required</h2>
        <p className="text-gray-400 mb-6">
          You must be logged in to upload media
        </p>
        <button onClick={() => navigate('/login')} className="btn-neon">
          Login
        </button>
      </div>
    )
  }

  const getMediaType = (file: File): MediaType => {
    if (file.type.startsWith('image/')) return 'image'
    if (file.type.startsWith('video/')) return 'video'
    if (file.type.startsWith('audio/')) return 'audio'
    if (file.type === 'image/gif') return 'gif'
    return 'image'
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
            Upload Media
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Share your memes, dances, art, and more with the community
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Upload Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* File Upload */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6"
          >
            <h2 className="text-xl font-bold text-white mb-4">Upload File</h2>

            {!file ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all ${
                  isDragging
                    ? 'border-neon-green bg-neon-green/5'
                    : 'border-gray-800 hover:border-gray-700'
                }`}
              >
                <Upload className="w-16 h-16 mx-auto mb-4 text-gray-500" />
                <p className="text-white font-semibold mb-2">
                  Drop your file here or click to browse
                </p>
                <p className="text-sm text-gray-400">
                  Images, videos, GIFs, or audio files
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,video/*,audio/*"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0]
                    if (selectedFile) handleFileSelect(selectedFile)
                  }}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-4">
                {/* File Preview */}
                <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                  {preview ? (
                    getMediaType(file) === 'video' ? (
                      <video src={preview} controls className="w-full max-h-96" />
                    ) : (
                      <img src={preview} alt="Preview" className="w-full max-h-96 object-contain" />
                    )
                  ) : (
                    <div className="h-64 flex items-center justify-center bg-gradient-to-br from-neon-pink/20 to-neon-blue/20">
                      <Music className="w-24 h-24 text-neon-pink" />
                    </div>
                  )}

                  <button
                    onClick={() => {
                      setFile(null)
                      setPreview(null)
                      setAiSuggestion(null)
                    }}
                    className="absolute top-4 right-4 w-8 h-8 rounded-full bg-red-500 flex items-center justify-center hover:bg-red-600 transition-colors"
                  >
                    <X className="w-5 h-5 text-white" />
                  </button>
                </div>

                {/* File Info */}
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">{file.name}</span>
                  <span className="text-gray-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>
            )}
          </motion.div>

          {/* AI Suggestion */}
          {aiSuggestion && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-panel p-6 border-l-4 border-neon-green"
            >
              <div className="flex items-start gap-3 mb-4">
                <Sparkles className="w-6 h-6 text-neon-green flex-shrink-0 mt-1" />
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-white mb-2">
                    AI Analysis Complete
                  </h3>
                  <p className="text-gray-400 text-sm mb-4">
                    Our AI has analyzed your media and suggests the following
                    categorization
                  </p>

                  <div className="bg-gray-900 rounded-lg p-4 mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">
                          {CATEGORY_INFO[aiSuggestion.category].icon}
                        </span>
                        <div>
                          <div className="font-semibold text-white">
                            {CATEGORY_INFO[aiSuggestion.category].name}
                          </div>
                          <div className="text-xs text-gray-500">
                            {CATEGORY_INFO[aiSuggestion.category].description}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-neon-green font-bold">
                          {Math.round(aiSuggestion.confidence * 100)}%
                        </div>
                        <div className="text-xs text-gray-500">Confidence</div>
                      </div>
                    </div>

                    {/* Confidence Indicator */}
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-neon-green to-neon-blue transition-all"
                        style={{ width: `${aiSuggestion.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Suggested Tags */}
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-gray-400 mb-2">
                      Suggested Tags:
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {aiSuggestion.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-3 py-1 bg-neon-green/10 text-neon-green rounded-lg text-sm"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    {aiSuggestion.confidence > 0.85 ? (
                      <Check className="w-4 h-4 text-neon-green" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-yellow-500" />
                    )}
                    <span>
                      {aiSuggestion.confidence > 0.85
                        ? 'High confidence - This looks correct!'
                        : 'Medium confidence - You may want to adjust the category'}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Media Details */}
          {file && (
            <>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-panel p-6"
              >
                <h2 className="text-xl font-bold text-white mb-4">
                  Media Details
                </h2>

                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">
                      Title *
                    </label>
                    <input
                      type="text"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder="Give your media a catchy title..."
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                    />
                  </div>

                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">
                      Description (optional)
                    </label>
                    <textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Add a description..."
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors resize-none"
                      rows={3}
                    />
                  </div>

                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">
                      Tags (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                      placeholder="funny, viral, trending"
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:border-neon-green transition-colors"
                    />
                    {aiSuggestion && (
                      <p className="text-xs text-gray-500 mt-2">
                        💡 Suggested:{' '}
                        {aiSuggestion.tags.map((tag) => `#${tag}`).join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              </motion.div>

              {/* Category Selection */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="glass-panel p-6"
              >
                <h2 className="text-xl font-bold text-white mb-4">
                  Category{' '}
                  {aiSuggestion && (
                    <span className="text-sm font-normal text-gray-400">
                      (AI suggested: {CATEGORY_INFO[aiSuggestion.category].name})
                    </span>
                  )}
                </h2>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {(Object.entries(CATEGORY_INFO) as [MediaCategory, typeof CATEGORY_INFO[MediaCategory]][]).map(
                    ([key, info]) => (
                      <button
                        key={key}
                        onClick={() => setSelectedCategory(key)}
                        className={`p-4 rounded-lg border-2 transition-all ${
                          selectedCategory === key
                            ? 'border-neon-green bg-neon-green/5'
                            : 'border-gray-800 hover:border-gray-700'
                        } ${
                          aiSuggestion?.category === key
                            ? 'ring-2 ring-neon-blue ring-offset-2 ring-offset-gray-950'
                            : ''
                        }`}
                      >
                        <div className="text-2xl mb-2">{info.icon}</div>
                        <div className="text-sm font-semibold text-white">
                          {info.name}
                        </div>
                      </button>
                    )
                  )}
                </div>
              </motion.div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={!title.trim() || uploadMutation.isPending}
                className="btn-neon w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploadMutation.isPending ? 'Uploading...' : 'Upload Media'}
              </button>

              {uploadMutation.isError && (
                <div className="glass-panel p-4 border-l-4 border-red-500">
                  <div className="flex items-center gap-2 text-red-400">
                    <AlertCircle className="w-5 h-5" />
                    <span>
                      {uploadMutation.error instanceof Error
                        ? uploadMutation.error.message
                        : 'Upload failed. Please try again.'}
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-4">Upload Tips</h3>
            <div className="space-y-3 text-sm text-gray-400">
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Use clear, descriptive titles</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Add relevant tags for discoverability</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>Let AI suggest the best category</span>
              </div>
              <div className="flex gap-3">
                <span className="text-neon-green">✓</span>
                <span>High-quality content gets more engagement</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-6 border-l-4 border-neon-blue"
          >
            <h3 className="text-lg font-bold text-white mb-3">
              AI Categorization
            </h3>
            <p className="text-sm text-gray-400 mb-3">
              Our AI analyzes your media to suggest the best category, ensuring
              your content reaches the right audience.
            </p>
            <div className="flex items-center gap-2 text-sm text-neon-blue">
              <Sparkles className="w-4 h-4" />
              <span>Powered by Machine Learning</span>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-6"
          >
            <h3 className="text-lg font-bold text-white mb-3">
              Supported Formats
            </h3>
            <div className="space-y-2 text-sm text-gray-400">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-neon-green" />
                <span>Images: JPG, PNG, GIF, WebP</span>
              </div>
              <div className="flex items-center gap-2">
                <Film className="w-4 h-4 text-neon-blue" />
                <span>Videos: MP4, WebM, MOV</span>
              </div>
              <div className="flex items-center gap-2">
                <Music className="w-4 h-4 text-neon-pink" />
                <span>Audio: MP3, WAV, OGG</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
