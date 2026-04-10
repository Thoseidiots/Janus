import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const auth = localStorage.getItem('jmaxing-auth')
  if (auth) {
    try {
      const { state } = JSON.parse(auth)
      if (state.token) {
        config.headers.Authorization = `Bearer ${state.token}`
      }
    } catch (e) {
      console.error('Failed to parse auth token')
    }
  }
  return config
})

// API types
export interface Job {
  job_id: string
  input_code: string
  tests: Array<{ input: any; expected: any }>
  reward_jc: number
  difficulty: number
  created_at: number
  status: 'open' | 'in_progress' | 'completed'
  submissions: number
  best_score?: number
}

export interface Submission {
  submission_id: string
  job_id: string
  node_id: string
  output_code: string
  score: number
  payout: number
  time_taken: number
  status: 'pending' | 'scored' | 'rejected'
  timestamp: number
}

export interface LeaderboardEntry {
  node_id: string
  username: string
  total_earned: number
  avg_score: number
  tasks_completed: number
  rank: number
  reputation: number
}

// API functions
export const jobsAPI = {
  list: async () => {
    const response = await apiClient.get<Job[]>('/jobs')
    return response.data
  },

  get: async (jobId: string) => {
    const response = await apiClient.get<Job>(`/jobs/${jobId}`)
    return response.data
  },

  submit: async (jobId: string, code: string) => {
    const response = await apiClient.post(`/jobs/${jobId}/submit`, {
      output_code: code,
    })
    return response.data
  },

  create: async (job: Partial<Job>) => {
    const response = await apiClient.post('/jobs', job)
    return response.data
  },
}

export const submissionsAPI = {
  getForJob: async (jobId: string) => {
    const response = await apiClient.get<Submission[]>(`/jobs/${jobId}/submissions`)
    return response.data
  },

  getMySubmissions: async () => {
    const response = await apiClient.get<Submission[]>('/submissions/me')
    return response.data
  },
}

export const leaderboardAPI = {
  get: async (limit: number = 100) => {
    const response = await apiClient.get<LeaderboardEntry[]>('/leaderboard', {
      params: { limit },
    })
    return response.data
  },
}

export const walletAPI = {
  getBalance: async () => {
    const response = await apiClient.get('/credits/balance')
    return response.data
  },

  getTransactions: async () => {
    const response = await apiClient.get('/credits/transactions')
    return response.data
  },
}
