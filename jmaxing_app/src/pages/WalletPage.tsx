import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { walletAPI } from '../api/client'
import { useAuthStore } from '../store/authStore'

interface Transaction {
  id: string
  type: 'earned' | 'spent' | 'bonus'
  amount: number
  description: string
  timestamp: number
  job_id?: string
}

export default function WalletPage() {
  const { user } = useAuthStore()

  const { data: balance } = useQuery({
    queryKey: ['balance'],
    queryFn: walletAPI.getBalance,
    enabled: !!user,
  })

  const { data: transactions } = useQuery<Transaction[]>({
    queryKey: ['transactions'],
    queryFn: walletAPI.getTransactions,
    enabled: !!user,
  })

  const totalEarned = transactions
    ?.filter((t) => t.type === 'earned' || t.type === 'bonus')
    .reduce((sum, t) => sum + t.amount, 0) || 0

  const totalSpent = transactions
    ?.filter((t) => t.type === 'spent')
    .reduce((sum, t) => sum + t.amount, 0) || 0

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-5xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green via-neon-blue to-neon-pink bg-clip-text text-transparent">
            Wallet
          </span>
        </h1>
        <p className="text-gray-400 text-lg">
          Manage your Janus Credits
        </p>
      </motion.div>

      {/* Balance Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-8 mb-8 text-center bg-gradient-to-br from-gray-900/50 to-gray-800/50 border-2 border-neon-green/30"
      >
        <div className="text-sm text-gray-400 mb-2">Current Balance</div>
        <div className="text-6xl font-bold mb-4">
          <span className="bg-gradient-to-r from-neon-green to-neon-blue bg-clip-text text-transparent">
            {balance?.balance.toLocaleString() || user?.balance.toLocaleString() || '0'} JC
          </span>
        </div>
        <div className="text-sm text-gray-500">
          ≈ ${((balance?.balance || user?.balance || 0) * 0.01).toFixed(2)} USD
        </div>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
      >
        <div className="stat-card">
          <div className="text-3xl font-bold text-neon-green mb-2">
            +{totalEarned.toLocaleString()} JC
          </div>
          <div className="text-sm text-gray-400">Total Earned</div>
        </div>

        <div className="stat-card">
          <div className="text-3xl font-bold text-red-400 mb-2">
            -{totalSpent.toLocaleString()} JC
          </div>
          <div className="text-sm text-gray-400">Total Spent</div>
        </div>

        <div className="stat-card">
          <div className="text-3xl font-bold text-neon-blue mb-2">
            {transactions?.length || 0}
          </div>
          <div className="text-sm text-gray-400">Transactions</div>
        </div>
      </motion.div>

      {/* Transaction History */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-panel p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Transaction History</h2>
          <div className="flex items-center gap-2">
            <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
              Filter
            </button>
            <button className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
              Export
            </button>
          </div>
        </div>

        {!transactions || transactions.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">💰</div>
            <h3 className="text-xl font-bold text-white mb-2">
              No transactions yet
            </h3>
            <p className="text-gray-400">
              Start earning JC by completing jobs!
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {transactions.map((tx, index) => (
              <motion.div
                key={tx.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.03 }}
                className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {/* Icon */}
                    <div
                      className={`w-12 h-12 rounded-full flex items-center justify-center text-xl ${
                        tx.type === 'earned'
                          ? 'bg-neon-green/10 text-neon-green'
                          : tx.type === 'bonus'
                          ? 'bg-neon-blue/10 text-neon-blue'
                          : 'bg-red-500/10 text-red-400'
                      }`}
                    >
                      {tx.type === 'earned' ? '💎' : tx.type === 'bonus' ? '🎁' : '🛒'}
                    </div>

                    {/* Details */}
                    <div>
                      <div className="font-semibold text-white mb-1">
                        {tx.description}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-500">
                        <span>
                          {new Date(tx.timestamp).toLocaleDateString()} at{' '}
                          {new Date(tx.timestamp).toLocaleTimeString()}
                        </span>
                        {tx.job_id && (
                          <>
                            <span>•</span>
                            <span className="text-neon-green">
                              Job #{tx.job_id.slice(0, 8)}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Amount */}
                  <div className="text-right">
                    <div
                      className={`text-2xl font-bold ${
                        tx.type === 'spent' ? 'text-red-400' : 'text-neon-green'
                      }`}
                    >
                      {tx.type === 'spent' ? '-' : '+'}
                      {tx.amount.toLocaleString()} JC
                    </div>
                    {tx.type === 'bonus' && (
                      <span className="text-xs text-neon-blue">Bonus!</span>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8"
      >
        <div className="glass-panel p-6">
          <h3 className="text-lg font-bold text-white mb-3">
            Earn More Credits
          </h3>
          <p className="text-gray-400 text-sm mb-4">
            Complete high-quality code improvements to earn more JC
          </p>
          <button className="btn-neon w-full">
            Browse Jobs
          </button>
        </div>

        <div className="glass-panel p-6">
          <h3 className="text-lg font-bold text-white mb-3">
            Withdraw Credits
          </h3>
          <p className="text-gray-400 text-sm mb-4">
            Convert your JC to cash or other cryptocurrencies
          </p>
          <button className="btn-secondary w-full" disabled>
            Coming Soon
          </button>
        </div>
      </motion.div>

      {/* Info Box */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass-panel p-6 mt-8 border-l-4 border-neon-blue"
      >
        <h3 className="text-lg font-bold text-white mb-3">
          About Janus Credits (JC)
        </h3>
        <div className="space-y-2 text-sm text-gray-400">
          <p>
            • Janus Credits are the native currency of the J-MAXING ecosystem
          </p>
          <p>
            • Earn JC by improving code quality and passing automated tests
          </p>
          <p>
            • Higher quality scores earn bigger payouts and bonus multipliers
          </p>
          <p>
            • JC operates on the Janus mesh network - completely decentralized
          </p>
          <p className="text-neon-green pt-2">
            • 1 JC ≈ $0.01 USD (market rate)
          </p>
        </div>
      </motion.div>
    </div>
  )
}
