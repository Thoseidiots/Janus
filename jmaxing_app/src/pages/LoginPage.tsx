import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Zap, LogIn, UserPlus } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [isSignup, setIsSignup] = useState(false)
  const navigate = useNavigate()
  const { login } = useAuthStore()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (!username.trim()) return

    login(username.trim())
    navigate('/jobs')
  }

  return (
    <div className="min-h-screen flex items-center justify-center gradient-bg relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />
      <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-janus-500/30 rounded-full blur-3xl animate-pulse-slow" />
      <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-neon-green/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md px-4"
      >
        <div className="glass-panel p-8 border-2">
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-janus-600/20 rounded-xl mb-4">
              <Zap className="w-8 h-8 text-neon-green animate-pulse" />
            </div>
            <h1 className="text-3xl font-bold mb-2">
              <span className="neon-text">J-MAXING</span>
            </h1>
            <p className="text-gray-400">
              {isSignup ? 'Create your account' : 'Sign in to continue'}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg focus:outline-none focus:border-janus-500 transition-colors"
                required
              />
            </div>

            <button
              type="submit"
              className="w-full btn-neon flex items-center justify-center space-x-2"
            >
              {isSignup ? (
                <>
                  <UserPlus className="w-5 h-5" />
                  <span>Create Account</span>
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  <span>Sign In</span>
                </>
              )}
            </button>
          </form>

          {/* Toggle */}
          <div className="mt-6 text-center">
            <button
              onClick={() => setIsSignup(!isSignup)}
              className="text-sm text-gray-400 hover:text-neon-green transition-colors"
            >
              {isSignup ? (
                <>Already have an account? <span className="font-semibold">Sign in</span></>
              ) : (
                <>New to J-MAXING? <span className="font-semibold">Create account</span></>
              )}
            </button>
          </div>

          {/* Info */}
          <div className="mt-8 pt-6 border-t border-gray-800">
            <div className="text-sm text-gray-400 space-y-2">
              <p className="flex items-center">
                <span className="w-2 h-2 bg-neon-green rounded-full mr-2" />
                Start with 1,000 JC free credits
              </p>
              <p className="flex items-center">
                <span className="w-2 h-2 bg-neon-green rounded-full mr-2" />
                No email verification required
              </p>
              <p className="flex items-center">
                <span className="w-2 h-2 bg-neon-green rounded-full mr-2" />
                100% mesh-native, no gatekeepers
              </p>
            </div>
          </div>
        </div>

        {/* Bottom text */}
        <p className="text-center mt-6 text-sm text-gray-500">
          By signing in, you agree to the J-MAXING mesh network terms
        </p>
      </motion.div>
    </div>
  )
}
