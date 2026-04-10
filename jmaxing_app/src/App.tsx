import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import JobBoardPage from './pages/JobBoardPage'
import JobDetailPage from './pages/JobDetailPage'
import LeaderboardPage from './pages/LeaderboardPage'
import ProfilePage from './pages/ProfilePage'
import WalletPage from './pages/WalletPage'
import LoginPage from './pages/LoginPage'
import SubmitJobPage from './pages/SubmitJobPage'
import FeedPage from './pages/FeedPage'
import ReferralPage from './pages/ReferralPage'
import UserProfilePage from './pages/UserProfilePage'
import ProjectsPage from './pages/ProjectsPage'
import ProjectDetailPage from './pages/ProjectDetailPage'
import CreateProjectPage from './pages/CreateProjectPage'
import MediaPage from './pages/MediaPage'
import MediaDetailPage from './pages/MediaDetailPage'
import UploadMediaPage from './pages/UploadMediaPage'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user } = useAuthStore()

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />

        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="jobs" element={<JobBoardPage />} />
          <Route path="jobs/:jobId" element={<JobDetailPage />} />
          <Route path="leaderboard" element={<LeaderboardPage />} />

          {/* Social Routes */}
          <Route path="feed" element={
            <ProtectedRoute>
              <FeedPage />
            </ProtectedRoute>
          } />

          <Route path="users/:userId" element={<UserProfilePage />} />

          <Route path="referrals" element={
            <ProtectedRoute>
              <ReferralPage />
            </ProtectedRoute>
          } />

          {/* Project Routes */}
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="projects/create" element={
            <ProtectedRoute>
              <CreateProjectPage />
            </ProtectedRoute>
          } />
          <Route path="projects/:projectId" element={<ProjectDetailPage />} />

          {/* Media Routes */}
          <Route path="media" element={<MediaPage />} />
          <Route path="media/upload" element={
            <ProtectedRoute>
              <UploadMediaPage />
            </ProtectedRoute>
          } />
          <Route path="media/:mediaId" element={<MediaDetailPage />} />

          {/* User Routes */}
          <Route path="profile" element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          } />

          <Route path="wallet" element={
            <ProtectedRoute>
              <WalletPage />
            </ProtectedRoute>
          } />

          <Route path="submit" element={
            <ProtectedRoute>
              <SubmitJobPage />
            </ProtectedRoute>
          } />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
