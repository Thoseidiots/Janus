import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Code2,
  Trophy,
  Zap,
  TrendingUp,
  Shield,
  Users,
  Rocket,
  Target,
  ArrowRight,
  Gift,
  MessageCircle,
  Heart,
} from 'lucide-react'
import { useAuthStore } from '../store/authStore'

export default function HomePage() {
  const { user } = useAuthStore()

  const features = [
    {
      icon: Code2,
      title: 'Code & Earn',
      description: 'Improve code, get paid in Janus Credits. No resumes, no interviews.',
    },
    {
      icon: Users,
      title: 'Social Coding',
      description: 'Follow devs, share code, get feedback. Build your reputation in the community.',
    },
    {
      icon: Gift,
      title: 'Refer & Earn',
      description: 'Invite friends and earn 100 JC per referral. They get 50 JC bonus to start.',
    },
    {
      icon: Trophy,
      title: 'Compete Globally',
      description: 'Climb the leaderboard. Top performers get bonus rewards and reputation.',
    },
    {
      icon: Shield,
      title: 'Quality Verified',
      description: 'Automated testing and static analysis ensures your work is legit.',
    },
    {
      icon: Zap,
      title: 'Instant Payout',
      description: 'Credits hit your wallet immediately. No waiting, no middlemen.',
    },
  ]

  const stats = [
    { label: 'Total Earned', value: '500K+ JC', icon: TrendingUp },
    { label: 'Active Devs', value: '1,234', icon: Users },
    { label: 'Jobs Posted', value: '5,678', icon: Code2 },
    { label: 'Avg Quality', value: '0.87', icon: Target },
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden gradient-bg">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <h1 className="text-5xl sm:text-7xl font-bold mb-6">
              <span className="block">Code Better.</span>
              <span className="block neon-text">Earn Faster.</span>
            </h1>

            <p className="text-xl sm:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Get paid to improve code. Compete with devs worldwide.
              No applications. No bullshit. Just code and earn Janus Credits.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              {user ? (
                <Link to="/jobs" className="btn-neon text-lg px-8 py-4 flex items-center space-x-2">
                  <Code2 className="w-5 h-5" />
                  <span>Browse Jobs</span>
                  <ArrowRight className="w-5 h-5" />
                </Link>
              ) : (
                <Link to="/login" className="btn-neon text-lg px-8 py-4 flex items-center space-x-2">
                  <Rocket className="w-5 h-5" />
                  <span>Start Earning</span>
                  <ArrowRight className="w-5 h-5" />
                </Link>
              )}

              <Link to="/leaderboard" className="btn-secondary text-lg px-8 py-4 flex items-center space-x-2">
                <Trophy className="w-5 h-5" />
                <span>View Leaderboard</span>
              </Link>
            </div>

            {/* Live Stats Bar */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4"
            >
              {stats.map((stat, index) => {
                const Icon = stat.icon
                return (
                  <div key={index} className="glass-panel p-6 text-center">
                    <Icon className="w-6 h-6 mx-auto mb-2 text-neon-green" />
                    <div className="text-2xl font-bold text-neon-green mb-1">
                      {stat.value}
                    </div>
                    <div className="text-sm text-gray-400">{stat.label}</div>
                  </div>
                )
              })}
            </motion.div>
          </motion.div>
        </div>

        {/* Animated gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-janus-500/30 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-neon-green/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />
      </section>

      {/* How It Works */}
      <section className="py-24 bg-gray-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4">How J-MAXING Works</h2>
            <p className="text-xl text-gray-400">Four simple steps to start earning</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              {
                step: '01',
                title: 'Browse Jobs',
                description: 'Find code improvement tasks that match your skills',
              },
              {
                step: '02',
                title: 'Submit Solution',
                description: 'Write better code, run tests, ensure quality',
              },
              {
                step: '03',
                title: 'Get Scored',
                description: 'Automated evaluation scores correctness, quality, improvement',
              },
              {
                step: '04',
                title: 'Earn Credits',
                description: 'JC hits your wallet instantly. Use it or cash out.',
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                <div className="glass-panel p-6 h-full">
                  <div className="text-5xl font-bold text-neon-green/20 mb-4">
                    {item.step}
                  </div>
                  <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                  <p className="text-gray-400">{item.description}</p>
                </div>

                {index < 3 && (
                  <div className="hidden md:block absolute top-1/2 right-0 transform translate-x-1/2 -translate-y-1/2">
                    <ArrowRight className="w-6 h-6 text-janus-500" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4">Why J-MAXING?</h2>
            <p className="text-xl text-gray-400">Built different. For devs who code, not talk.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-panel p-8 hover:border-janus-500/50 transition-all group"
                >
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className="w-12 h-12 bg-janus-600/20 rounded-lg flex items-center justify-center group-hover:bg-janus-600/30 transition-all">
                        <Icon className="w-6 h-6 text-janus-400" />
                      </div>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                      <p className="text-gray-400">{feature.description}</p>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 gradient-bg relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl sm:text-5xl font-bold mb-6">
              Ready to <span className="neon-text">MAX</span> Out?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Join thousands of devs earning JC by writing better code.
              No resume. No interview. Just pure skill.
            </p>

            {user ? (
              <Link to="/jobs" className="btn-neon text-lg px-8 py-4 inline-flex items-center space-x-2">
                <Code2 className="w-5 h-5" />
                <span>Start Earning Now</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
            ) : (
              <Link to="/login" className="btn-neon text-lg px-8 py-4 inline-flex items-center space-x-2">
                <Rocket className="w-5 h-5" />
                <span>Sign Up Free</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
            )}

            <p className="mt-4 text-sm text-gray-400">
              No credit card required • Instant payouts • 100% autonomous
            </p>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
