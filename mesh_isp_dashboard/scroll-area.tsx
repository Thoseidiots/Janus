import { useState } from "react";
import { useLocation } from "wouter";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Wifi, AlertCircle } from "lucide-react";

export default function Login() {
  const [, setLocation] = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const loginMutation = trpc.auth.login.useMutation();
  const registerMutation = trpc.auth.register.useMutation();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await loginMutation.mutateAsync({ email, password });
      setLocation("/isp");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await registerMutation.mutateAsync({ email, name, password });
      setLocation("/isp");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Wifi className="w-10 h-10 text-blue-400" />
            <h1 className="text-3xl font-bold text-white">MeshISP</h1>
          </div>
          <p className="text-slate-300">Personal Internet Service Provider</p>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert className="mb-6 bg-red-900/20 border-red-700">
            <AlertCircle className="h-4 w-4 text-red-400" />
            <AlertDescription className="text-red-300">{error}</AlertDescription>
          </Alert>
        )}

        {/* Login/Register Card */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">
              {isRegistering ? "Create Account" : "Sign In"}
            </CardTitle>
            <CardDescription>
              {isRegistering
                ? "Set up your MeshISP account"
                : "Access your ISP dashboard"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={isRegistering ? handleRegister : handleLogin} className="space-y-4">
              {isRegistering && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Full Name
                  </label>
                  <Input
                    type="text"
                    placeholder="Your name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="bg-slate-700 border-slate-600 text-white placeholder-slate-400"
                    required
                  />
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Email
                </label>
                <Input
                  type="email"
                  placeholder="your@email.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white placeholder-slate-400"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Password
                </label>
                <Input
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white placeholder-slate-400"
                  required
                />
              </div>

              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={loading}
              >
                {loading ? "Loading..." : isRegistering ? "Create Account" : "Sign In"}
              </Button>
            </form>

            {/* Toggle between login and register */}
            <div className="mt-4 text-center">
              <p className="text-slate-400 text-sm">
                {isRegistering ? "Already have an account?" : "Don't have an account?"}{" "}
                <button
                  type="button"
                  onClick={() => {
                    setIsRegistering(!isRegistering);
                    setError("");
                  }}
                  className="text-blue-400 hover:text-blue-300 font-medium"
                >
                  {isRegistering ? "Sign In" : "Register"}
                </button>
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Demo Info */}
        <div className="mt-6 p-4 bg-slate-800 border border-slate-700 rounded-lg">
          <p className="text-xs text-slate-400">
            <strong>Demo:</strong> Create a new account to get started. This is a local authentication system with no external dependencies.
          </p>
        </div>
      </div>
    </div>
  );
}
