import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface User {
  id: string
  username: string
  email: string
  balance: number
  reputation: number
  tasksCompleted: number
  rank: number
  joinedAt: number
}

interface AuthState {
  user: User | null
  token: string | null
  login: (username: string) => void
  logout: () => void
  updateBalance: (balance: number) => void
  updateUser: (updates: Partial<User>) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,

      login: (username: string) => {
        // In production: call API, get real user data
        const mockUser: User = {
          id: `user_${username}`,
          username,
          email: `${username}@mesh.local`,
          balance: 1000,
          reputation: 100,
          tasksCompleted: 0,
          rank: 999,
          joinedAt: Date.now(),
        }

        set({
          user: mockUser,
          token: mockUser.id,
        })
      },

      logout: () => {
        set({ user: null, token: null })
      },

      updateBalance: (balance: number) => {
        set((state) => ({
          user: state.user ? { ...state.user, balance } : null,
        }))
      },

      updateUser: (updates: Partial<User>) => {
        set((state) => ({
          user: state.user ? { ...state.user, ...updates } : null,
        }))
      },
    }),
    {
      name: 'jmaxing-auth',
    }
  )
)
