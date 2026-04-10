import { Outlet, Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Code2,
  Trophy,
  Wallet,
  User,
  LogOut,
  Zap,
  TrendingUp,
  PlusCircle,
  Home,
  Users,
  Gift,
  FolderGit2,
  Image
} from 'lucide-react'
import { useAuthStore } from '../store/authStore'

export default function Layout() {
  const location = useLocation()
  const { user, logout } = useAuthStore()

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/feed', label: 'Feed', icon: TrendingUp },
    { path: '/media', label: 'Media', icon: Image },
    { path: '/projects', label: 'Projects', icon: FolderGit2 },
    { path: '/jobs', label: 'Jobs', icon: Code2 },
    { path: '/leaderboard', label: 'Leaderboard', icon: Trophy },
    { path: '/referrals', label: 'Refer & Earn', icon: Gift },
  ]

  const isActive = (path: string) => {
    return location.pathname === path
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-panel border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2 group">
              <div className="relative">
                <Zap className="w-8 h-8 text-neon-green group-hover:animate-pulse" />
                <div className="absolute inset-0 bg-neon-green blur-md opacity-50 group-hover:opacity-75 transition-opacity" />
              </div>
              <span className="text-2xl font-bold">
                <span className="neon-text">J-MAXING</span>
              </span>
            </Link>

            {/* Navigation */}
            <nav className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const active = isActive(item.path)
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`
                      px-4 py-2 rounded-lg font-semibold transition-all duration-200
                      flex items-center space-x-2
                      ${active
                        ? 'bg-janus-600 text-white'
                        : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                      }
                    `}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </nav>

            {/* User Menu */}
            {user ? (
              <div className="flex items-center space-x-4">
                {/* Balance */}
                <Link
                  to="/wallet"
                  className="flex items-center space-x-2 px-4 py-2 glass-panel hover:border-janus-500/50 rounded-lg transition-all group"
                >
                  <Wallet className="w-4 h-4 text-neon-green group-hover:animate-pulse" />
                  <span className="font-bold text-neon-green">{user.balance.toLocaleString()}</span>
                  <span className="text-xs text-gray-400">JC</span>
                </Link>

                {/* Submit Job */}
                <Link
                  to="/submit"
                  className="btn-neon flex items-center space-x-2"
                >
                  <PlusCircle className="w-4 h-4" />
                  <span>Submit Job</span>
                </Link>

                {/* Profile */}
                <div className="relative group">
                  <Link
                    to="/profile"
                    className="flex items-center space-x-2 px-3 py-2 glass-panel hover:border-janus-500/50 rounded-lg transition-all"
                  >
                    <User className="w-4 h-4" />
                    <span className="font-semibold">{user.username}</span>
                    <div className="text-xs px-2 py-0.5 bg-janus-600 rounded-full">
                      #{user.rank}
                    </div>
                  </Link>

                  {/* Dropdown */}
                  <div className="absolute right-0 mt-2 w-48 glass-panel border border-gray-700 rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                    <Link
                      to="/profile"
                      className="block px-4 py-2 hover:bg-gray-800 rounded-t-lg"
                    >
                      View Profile
                    </Link>
                    <button
                      onClick={logout}
                      className="w-full text-left px-4 py-2 hover:bg-gray-800 rounded-b-lg flex items-center space-x-2 text-red-400"
                    >
                      <LogOut className="w-4 h-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <Link to="/login" className="btn-primary">
                Sign In
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-bold mb-4 neon-text">J-MAXING</h3>
              <p className="text-gray-400 text-sm">
                Gamified code improvement marketplace on the Janus mesh network.
                No gatekeepers. Just code and earn.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-bold mb-4">Quick Links</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link to="/jobs" className="text-gray-400 hover:text-neon-green transition-colors">
                    Browse Jobs
                  </Link>
                </li>
                <li>
                  <Link to="/leaderboard" className="text-gray-400 hover:text-neon-green transition-colors">
                    Leaderboard
                  </Link>
                </li>
                <li>
                  <a href="https://github.com/Thoseidiots/Janus" className="text-gray-400 hover:text-neon-green transition-colors">
                    GitHub
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-lg font-bold mb-4">Stats</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Earned</span>
                  <span className="text-neon-green font-bold">500K+ JC</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Users</span>
                  <span className="text-neon-green font-bold">1,234</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Jobs Completed</span>
                  <span className="text-neon-green font-bold">5,678</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400 text-sm">
            <p>
              Powered by <span className="text-janus-400 font-semibold">Janus</span> mesh network
              • No API keys • No gatekeepers • 100% autonomous
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
