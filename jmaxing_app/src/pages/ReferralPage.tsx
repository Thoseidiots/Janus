import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { socialAPI } from '../api/social'
import { useAuthStore } from '../store/authStore'
import { useNavigate } from 'react-router-dom'

export default function ReferralPage() {
  const { user } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [copied, setCopied] = useState(false)

  // Fetch referral stats
  const { data: stats } = useQuery({
    queryKey: ['referralStats'],
    queryFn: socialAPI.getReferralStats,
    enabled: !!user,
  })

  // Fetch referrals list
  const { data: referrals } = useQuery({
    queryKey: ['referrals'],
    queryFn: socialAPI.getReferrals,
    enabled: !!user,
  })

  // Generate referral code mutation
  const generateCodeMutation = useMutation({
    mutationFn: socialAPI.generateReferralCode,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['referralStats'] })
    },
  })

  const handleCopy = () => {
    if (stats?.referralLink) {
      navigator.clipboard.writeText(stats.referralLink)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleShare = (platform: 'twitter' | 'facebook' | 'linkedin') => {
    if (!stats?.referralLink) return

    const text = `Join me on J-MAXING and earn Janus Credits by improving code! Use my referral code: ${stats.referralCode}`
    const url = stats.referralLink

    const shareUrls = {
      twitter: `https://twitter.com/intent/tweet?text=${encodeURIComponent(
        text
      )}&url=${encodeURIComponent(url)}`,
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
      linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`,
    }

    window.open(shareUrls[platform], '_blank', 'width=600,height=400')
  }

  if (!user) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Login Required</h2>
        <p className="text-gray-400 mb-6">
          Login to access your referral dashboard
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
        className="text-center mb-12"
      >
        <h1 className="text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
            Refer & Earn
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Invite friends and earn Janus Credits together
        </p>
      </motion.div>

      {/* Referral Stats */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
      >
        <div className="stat-card text-center">
          <div className="text-4xl font-bold text-neon-green mb-2">
            {stats?.totalEarned || 0} JC
          </div>
          <div className="text-sm text-gray-400">Total Earned</div>
        </div>

        <div className="stat-card text-center">
          <div className="text-4xl font-bold text-neon-blue mb-2">
            {stats?.completedReferrals || 0}
          </div>
          <div className="text-sm text-gray-400">Completed Referrals</div>
        </div>

        <div className="stat-card text-center">
          <div className="text-4xl font-bold text-neon-pink mb-2">
            {stats?.pendingReferrals || 0}
          </div>
          <div className="text-sm text-gray-400">Pending Referrals</div>
        </div>

        <div className="stat-card text-center">
          <div className="text-4xl font-bold text-white mb-2">
            {stats?.totalReferrals || 0}
          </div>
          <div className="text-sm text-gray-400">Total Referrals</div>
        </div>
      </motion.div>

      {/* Referral Link */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-panel p-8 mb-8 text-center"
      >
        <h2 className="text-2xl font-bold text-white mb-4">
          Your Referral Link
        </h2>
        <p className="text-gray-400 mb-6">
          Share this link to invite friends. Both you and your friend earn{' '}
          <span className="text-neon-green font-bold">100 JC</span> when they
          complete their first job!
        </p>

        {/* Referral Code Display */}
        <div className="max-w-2xl mx-auto">
          <div className="bg-gray-950 border-2 border-neon-green/30 rounded-lg p-4 mb-4">
            <div className="text-sm text-gray-500 mb-2">Your Code</div>
            <div className="text-3xl font-bold text-neon-green font-mono">
              {stats?.referralCode || 'LOADING...'}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6 flex items-center gap-3">
            <input
              type="text"
              value={stats?.referralLink || 'Generating...'}
              readOnly
              className="flex-1 bg-transparent text-white font-mono text-sm focus:outline-none"
            />
            <button
              onClick={handleCopy}
              className="btn-neon px-6 py-2 whitespace-nowrap"
            >
              {copied ? '✓ Copied!' : 'Copy Link'}
            </button>
          </div>

          {/* Social Share Buttons */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => handleShare('twitter')}
              className="bg-[#1DA1F2] hover:bg-[#1a8cd8] text-white px-6 py-3 rounded-lg font-semibold transition-colors"
            >
              Share on Twitter
            </button>
            <button
              onClick={() => handleShare('facebook')}
              className="bg-[#4267B2] hover:bg-[#365899] text-white px-6 py-3 rounded-lg font-semibold transition-colors"
            >
              Share on Facebook
            </button>
            <button
              onClick={() => handleShare('linkedin')}
              className="bg-[#0077B5] hover:bg-[#006399] text-white px-6 py-3 rounded-lg font-semibold transition-colors"
            >
              Share on LinkedIn
            </button>
          </div>
        </div>
      </motion.div>

      {/* How It Works */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-panel p-8 mb-8"
      >
        <h2 className="text-2xl font-bold text-white mb-6 text-center">
          How Referrals Work
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 bg-neon-green/10 rounded-full flex items-center justify-center text-3xl mx-auto mb-4">
              📤
            </div>
            <h3 className="text-lg font-bold text-white mb-2">1. Share</h3>
            <p className="text-gray-400 text-sm">
              Share your unique referral link with friends via social media,
              email, or messaging apps
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-neon-blue/10 rounded-full flex items-center justify-center text-3xl mx-auto mb-4">
              🎯
            </div>
            <h3 className="text-lg font-bold text-white mb-2">2. They Join</h3>
            <p className="text-gray-400 text-sm">
              Your friend signs up using your link and gets{' '}
              <span className="text-neon-green font-semibold">50 JC bonus</span>{' '}
              to start
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-neon-pink/10 rounded-full flex items-center justify-center text-3xl mx-auto mb-4">
              💰
            </div>
            <h3 className="text-lg font-bold text-white mb-2">3. You Earn</h3>
            <p className="text-gray-400 text-sm">
              When they complete their first job, you both earn{' '}
              <span className="text-neon-green font-semibold">100 JC</span>{' '}
              instantly
            </p>
          </div>
        </div>

        <div className="mt-8 p-6 bg-neon-green/5 border border-neon-green/20 rounded-lg">
          <h4 className="text-lg font-bold text-neon-green mb-3">
            🎁 Bonus Rewards
          </h4>
          <div className="space-y-2 text-sm text-gray-300">
            <div className="flex items-center gap-3">
              <span className="text-neon-green">✓</span>
              <span>
                <strong>5 referrals:</strong> Unlock 500 JC bonus + "Influencer"
                badge
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-neon-green">✓</span>
              <span>
                <strong>10 referrals:</strong> Unlock 1,200 JC bonus + "Community
                Builder" badge
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-neon-green">✓</span>
              <span>
                <strong>25 referrals:</strong> Unlock 3,500 JC bonus + "Ambassador"
                status + 10% earnings boost
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Referral History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-panel p-6"
      >
        <h2 className="text-2xl font-bold text-white mb-6">
          Referral History
        </h2>

        {!referrals || referrals.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🎯</div>
            <h3 className="text-xl font-bold text-white mb-2">
              No referrals yet
            </h3>
            <p className="text-gray-400">
              Start sharing your referral link to earn JC!
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {referrals.map((referral, index) => (
              <motion.div
                key={referral.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-gray-900 rounded-lg p-4 border border-gray-800"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        referral.status === 'completed'
                          ? 'bg-neon-green'
                          : 'bg-yellow-500'
                      }`}
                    ></div>

                    <div>
                      <div className="font-semibold text-white">
                        {referral.referredUsername || 'Anonymous User'}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(referral.createdAt).toLocaleDateString()}
                        {referral.completedAt && (
                          <span>
                            {' '}
                            • Completed{' '}
                            {new Date(referral.completedAt).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    {referral.status === 'completed' ? (
                      <div className="text-neon-green font-bold text-lg">
                        +{referral.reward} JC
                      </div>
                    ) : (
                      <span className="badge-warning text-xs">Pending</span>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>

      {/* FAQ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-panel p-8 mt-8"
      >
        <h2 className="text-2xl font-bold text-white mb-6 text-center">
          Frequently Asked Questions
        </h2>

        <div className="space-y-4">
          <div>
            <h4 className="text-white font-semibold mb-2">
              How many people can I refer?
            </h4>
            <p className="text-gray-400 text-sm">
              There's no limit! Refer as many friends as you want and earn JC for
              each successful referral.
            </p>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-2">
              When do I receive my referral bonus?
            </h4>
            <p className="text-gray-400 text-sm">
              You receive 100 JC instantly when your referred friend completes
              their first job on J-MAXING.
            </p>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-2">
              Can I refer myself with multiple accounts?
            </h4>
            <p className="text-gray-400 text-sm">
              No. Our system detects duplicate accounts and fraudulent activity.
              Violators will have their accounts suspended and earnings forfeited.
            </p>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-2">
              Do referral bonuses expire?
            </h4>
            <p className="text-gray-400 text-sm">
              No. Once earned, your referral bonuses are yours to keep and use
              forever.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
