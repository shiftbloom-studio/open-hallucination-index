"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { User } from "@supabase/supabase-js";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

// Types
interface ApiUser {
  id: string;
  email: string;
  name: string | null;
  tokens: number;
  role: "user" | "admin";
  created_at: string;
  updated_at: string;
}

interface ApiKey {
  id: string;
  user_id: string | null;
  user_email: string | null;
  prefix: string;
  name: string;
  type: "standard" | "master" | "guest";
  token_limit: number | null;
  tokens_used: number;
  tokens_remaining: number | null;
  expires_at: string | null;
  is_active: boolean;
  last_used_at: string | null;
  created_at: string;
}

interface NewKeyResponse {
  id: string;
  key: string;
  prefix: string;
  name: string;
  type: "standard" | "master" | "guest";
  token_limit: number | null;
  expires_at: string | null;
  user_id: string | null;
  user_email: string | null;
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: "debug" | "info" | "warning" | "error";
  type: "request" | "response" | "error" | "health" | "auth" | "system";
  method: string;
  path: string;
  status_code: number | null;
  duration_ms: number | null;
  user_id: string | null;
  key_prefix: string | null;
  message: string;
  details: Record<string, unknown> | null;
}

interface LogStats {
  total_requests: number;
  total_errors: number;
  buffer_size: number;
  active_subscribers: number;
  uptime_seconds: number;
  requests_per_minute: number;
}

interface TestResult {
  claim: string;
  verdict: string;
  confidence: number;
  evidence: string;
}

interface TestVerifyResponse {
  success: boolean;
  test_type: string;
  test_text: string;
  claims_found: number;
  verification_score: number;
  duration_ms: number;
  results: TestResult[];
}

interface AdminDashboardClientProps {
  user: User;
}

// Icon components
const IconActivity = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const IconUsers = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const IconKey = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
  </svg>
);

const IconTools = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const IconCheck = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const IconX = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const IconGift = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7" />
  </svg>
);

const IconBeaker = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
  </svg>
);

export default function AdminDashboardClient({ user }: AdminDashboardClientProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "users" | "keys" | "tools">("overview");
  const [users, setUsers] = useState<ApiUser[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logStats, setLogStats] = useState<LogStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [newKey, setNewKey] = useState<NewKeyResponse | null>(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [sseError, setSseError] = useState<string | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Test verification state
  const [testRunning, setTestRunning] = useState(false);
  const [testResult, setTestResult] = useState<TestVerifyResponse | null>(null);
  const [selectedTestType, setSelectedTestType] = useState<"simple" | "complex" | "hallucination">("simple");

  // Token grant state
  const [grantEmail, setGrantEmail] = useState("");
  const [grantTokens, setGrantTokens] = useState("");
  const [grantReason, setGrantReason] = useState("");
  const [grantLoading, setGrantLoading] = useState(false);

  // Form state for creating keys
  const [keyForm, setKeyForm] = useState({
    name: "",
    type: "standard" as "standard" | "master" | "guest",
    user_id: "",
    token_limit: "",
    expires_in_days: "",
  });

  // Fetch users from Supabase via Next.js API
  const fetchUsers = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/users");
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users || []);
      } else {
        console.error("Failed to fetch users:", res.status);
      }
    } catch (error) {
      console.error("Failed to fetch users:", error);
    }
  }, []);

  // Fetch API keys
  const fetchApiKeys = useCallback(async () => {
    try {
      const res = await fetch("/api/ohi/api/v1/admin/keys", {
        headers: { "X-User-Id": user.id },
      });
      if (res.ok) {
        const data = await res.json();
        setApiKeys(data.keys || []);
      }
    } catch (error) {
      console.error("Failed to fetch API keys:", error);
    }
  }, [user.id]);

  // Fetch log stats
  const fetchLogStats = useCallback(async () => {
    try {
      const res = await fetch("/api/ohi/api/v1/admin/logs/stats", {
        headers: { "X-User-Id": user.id },
      });
      if (res.ok) {
        const data = await res.json();
        setLogStats(data);
      }
    } catch (error) {
      console.error("Failed to fetch log stats:", error);
    }
  }, [user.id]);

  // Initial data fetch
  useEffect(() => {
    fetchUsers();
    fetchApiKeys();
    fetchLogStats();
    const statsInterval = setInterval(fetchLogStats, 30000);
    return () => clearInterval(statsInterval);
  }, [fetchUsers, fetchApiKeys, fetchLogStats]);

  // Connect to live logs SSE stream
  useEffect(() => {
    if (activeTab !== "overview") {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
        setSseConnected(false);
      }
      return;
    }

    const fetchRecentLogs = async () => {
      try {
        const res = await fetch("/api/ohi/api/v1/admin/logs/recent?limit=50", {
          headers: { "X-User-Id": user.id },
        });
        if (res.ok) {
          const data = await res.json();
          setLogs(data.reverse());
        }
      } catch (error) {
        console.error("Failed to fetch recent logs:", error);
      }
    };

    fetchRecentLogs();

    const sseUrl = `/api/ohi/api/v1/admin/logs/stream`;
    const connectSSE = () => {
      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setSseConnected(true);
        setSseError(null);
      };

      eventSource.onmessage = (event) => {
        try {
          const logEntry: LogEntry = JSON.parse(event.data);
          if (logEntry.type === "system" && logEntry.message === "heartbeat") return;

          setLogs((prev) => [...prev, logEntry].slice(-200));

          if (logContainerRef.current) {
            setTimeout(() => {
              logContainerRef.current?.scrollTo({
                top: logContainerRef.current.scrollHeight,
                behavior: "smooth",
              });
            }, 50);
          }
        } catch (error) {
          console.error("Failed to parse log entry:", error);
        }
      };

      eventSource.onerror = () => {
        setSseConnected(false);
        setSseError("Connection lost. Reconnecting...");
        eventSource.close();
        eventSourceRef.current = null;
        setTimeout(() => {
          if (activeTab === "overview") connectSSE();
        }, 5000);
      };
    };

    connectSSE();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
        setSseConnected(false);
      }
    };
  }, [activeTab, user.id]);

  // Format log entry for display
  const formatLogEntry = (log: LogEntry): string => {
    const parts: string[] = [];
    if (log.method && log.path) parts.push(`${log.method} ${log.path}`);
    if (log.status_code) parts.push(`- ${log.status_code}`);
    if (log.duration_ms !== null) parts.push(`(${log.duration_ms.toFixed(0)}ms)`);
    if (log.key_prefix) parts.push(`[${log.key_prefix}...]`);
    if (log.message && !log.method) parts.push(log.message);
    return parts.join(" ") || log.message;
  };

  // Get log color based on level/status
  const getLogColor = (log: LogEntry): string => {
    if (log.level === "error") return "text-red-400";
    if (log.level === "warning") return "text-amber-400";
    if (log.type === "auth") return "text-cyan-400";
    if (log.type === "health") return "text-gray-500";
    if (log.type === "system") return "text-purple-400";
    if (log.status_code && log.status_code >= 500) return "text-red-400";
    if (log.status_code && log.status_code >= 400) return "text-amber-400";
    return "text-emerald-400";
  };

  // Create API key
  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const body: Record<string, unknown> = {
        name: keyForm.name,
        type: keyForm.type,
      };
      if (keyForm.type === "standard" && keyForm.user_id) body.user_id = keyForm.user_id;
      if (keyForm.token_limit) body.token_limit = parseInt(keyForm.token_limit, 10);
      if (keyForm.expires_in_days) body.expires_in_days = parseInt(keyForm.expires_in_days, 10);

      const res = await fetch("/api/ohi/api/v1/admin/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-Id": user.id },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const data = await res.json();
        setNewKey(data);
        toast.success("API key created successfully!");
        fetchApiKeys();
        setKeyForm({ name: "", type: "standard", user_id: "", token_limit: "", expires_in_days: "" });
      } else {
        const error = await res.json();
        toast.error(error.detail || "Failed to create API key");
      }
    } catch (error) {
      toast.error("Failed to create API key");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // Revoke API key
  const handleRevokeKey = async (keyId: string) => {
    if (!confirm("Are you sure you want to revoke this API key?")) return;

    try {
      const res = await fetch(`/api/ohi/api/v1/admin/keys/${keyId}`, {
        method: "DELETE",
        headers: { "X-User-Id": user.id },
      });
      if (res.ok) {
        toast.success("API key revoked");
        fetchApiKeys();
      } else {
        toast.error("Failed to revoke API key");
      }
    } catch (error) {
      toast.error("Failed to revoke API key");
      console.error(error);
    }
  };

  // Toggle user role
  const handleToggleRole = async (userId: string, currentRole: string) => {
    const newRole = currentRole === "admin" ? "user" : "admin";
    try {
      const res = await fetch(`/api/admin/users/${userId}/role`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: newRole }),
      });
      if (res.ok) {
        toast.success(`User role updated to ${newRole}`);
        fetchUsers();
      } else {
        const error = await res.json();
        toast.error(error.error || "Failed to update user role");
      }
    } catch (error) {
      toast.error("Failed to update user role");
      console.error(error);
    }
  };

  // Run test verification
  const handleRunTest = async () => {
    setTestRunning(true);
    setTestResult(null);

    try {
      const res = await fetch("/api/ohi/api/v1/admin/tools/test-verify", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-Id": user.id },
        body: JSON.stringify({ test_type: selectedTestType }),
      });

      if (res.ok) {
        const data = await res.json();
        setTestResult(data);
        toast.success("Test verification completed!");
      } else {
        toast.error("Test verification failed");
      }
    } catch (error) {
      toast.error("Test verification failed");
      console.error(error);
    } finally {
      setTestRunning(false);
    }
  };

  // Grant tokens
  const handleGrantTokens = async (e: React.FormEvent) => {
    e.preventDefault();
    setGrantLoading(true);

    try {
      const res = await fetch("/api/admin/grant-tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: grantEmail,
          tokens: parseInt(grantTokens, 10),
          reason: grantReason || "Admin grant",
        }),
      });

      if (res.ok) {
        const data = await res.json();
        toast.success(data.message);
        setGrantEmail("");
        setGrantTokens("");
        setGrantReason("");
        fetchUsers();
      } else {
        const error = await res.json();
        toast.error(error.error || "Failed to grant tokens");
      }
    } catch (error) {
      toast.error("Failed to grant tokens");
      console.error(error);
    } finally {
      setGrantLoading(false);
    }
  };

  const tabs = [
    { id: "overview" as const, label: "Live Monitor", icon: <IconActivity /> },
    { id: "users" as const, label: "Users", icon: <IconUsers /> },
    { id: "keys" as const, label: "API Keys", icon: <IconKey /> },
    { id: "tools" as const, label: "Tools", icon: <IconTools /> },
  ];

  return (
    <div className="min-h-screen">
      {/* Header with gradient */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 p-8 mb-8 shadow-2xl">
        <div className="absolute inset-0 bg-grid-white/10 [mask-image:linear-gradient(0deg,transparent,white)]" />
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <IconTools />
            </div>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              Admin Command Center
            </h1>
          </div>
          <p className="text-white/80 text-lg">
            Monitor, manage, and control your OHI infrastructure
          </p>
        </div>
      </div>

      {/* Quick Stats Row */}
      <div className="grid gap-4 md:grid-cols-4 mb-8">
        <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 p-6 shadow-lg transition-all hover:shadow-xl hover:scale-[1.02]">
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <p className="text-emerald-100 text-sm font-medium">Total Users</p>
            <p className="text-4xl font-bold text-white mt-1">{users.length}</p>
            <p className="text-emerald-200 text-xs mt-2">
              {users.filter((u) => u.role === "admin").length} admins
            </p>
          </div>
        </div>

        <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 p-6 shadow-lg transition-all hover:shadow-xl hover:scale-[1.02]">
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <p className="text-blue-100 text-sm font-medium">Active Keys</p>
            <p className="text-4xl font-bold text-white mt-1">
              {apiKeys.filter((k) => k.is_active).length}
            </p>
            <p className="text-blue-200 text-xs mt-2">
              {apiKeys.filter((k) => k.type === "master").length} master
            </p>
          </div>
        </div>

        <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 p-6 shadow-lg transition-all hover:shadow-xl hover:scale-[1.02]">
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <p className="text-amber-100 text-sm font-medium">Requests/Min</p>
            <p className="text-4xl font-bold text-white mt-1">
              {logStats?.requests_per_minute.toFixed(1) || "0"}
            </p>
            <p className="text-amber-200 text-xs mt-2">
              {logStats?.total_requests.toLocaleString() || 0} total
            </p>
          </div>
        </div>

        <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-rose-500 to-pink-600 p-6 shadow-lg transition-all hover:shadow-xl hover:scale-[1.02]">
          <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <p className="text-rose-100 text-sm font-medium">Errors</p>
            <p className="text-4xl font-bold text-white mt-1">
              {logStats?.total_errors.toLocaleString() || "0"}
            </p>
            <p className="text-rose-200 text-xs mt-2">
              {logStats?.active_subscribers || 0} viewers
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 p-1 bg-muted/50 rounded-xl w-fit backdrop-blur-sm">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-5 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? "bg-white dark:bg-gray-800 text-primary shadow-lg"
                : "text-muted-foreground hover:text-foreground hover:bg-white/50 dark:hover:bg-gray-800/50"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <Card className="border-0 shadow-xl bg-gradient-to-br from-gray-900 to-gray-950 overflow-hidden">
          <CardHeader className="border-b border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${sseConnected ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
                <div>
                  <CardTitle className="text-white">Live API Monitor</CardTitle>
                  <CardDescription className="text-gray-400">
                    {sseConnected ? "Real-time activity stream" : sseError || "Connecting..."}
                  </CardDescription>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setLogs([])} className="border-white/20 text-white hover:bg-white/10">
                  Clear
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => logContainerRef.current?.scrollTo({ top: logContainerRef.current.scrollHeight, behavior: "smooth" })}
                  className="border-white/20 text-white hover:bg-white/10"
                >
                  Latest
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div
              ref={logContainerRef}
              className="h-[600px] overflow-auto p-6 font-mono text-sm leading-relaxed"
            >
              {logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                  <div className="w-16 h-16 border-4 border-gray-700 border-t-emerald-500 rounded-full animate-spin mb-4" />
                  <p>{sseConnected ? "Waiting for API requests..." : "Connecting to stream..."}</p>
                </div>
              ) : (
                logs.map((log) => (
                  <div key={log.id} className={`py-1 ${getLogColor(log)} hover:bg-white/5 px-2 -mx-2 rounded transition-colors`}>
                    <span className="text-gray-600">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{" "}
                    <span className="text-gray-500 font-semibold">[{log.type.toUpperCase().padEnd(8)}]</span>{" "}
                    {formatLogEntry(log)}
                  </div>
                ))
              )}
            </div>
            <div className="border-t border-white/10 px-6 py-3 flex items-center justify-between text-xs text-gray-500">
              <span>{logs.length} entries</span>
              <span className={sseConnected ? "text-emerald-500" : "text-amber-500"}>
                {sseConnected ? "Connected" : "Reconnecting..."}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === "users" && (
        <Card className="shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <IconUsers />
              User Management
            </CardTitle>
            <CardDescription>View and manage user accounts and permissions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-4 text-left font-semibold text-muted-foreground">Email</th>
                    <th className="pb-4 text-left font-semibold text-muted-foreground">Name</th>
                    <th className="pb-4 text-left font-semibold text-muted-foreground">Tokens</th>
                    <th className="pb-4 text-left font-semibold text-muted-foreground">Role</th>
                    <th className="pb-4 text-left font-semibold text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.id} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
                      <td className="py-4 font-medium">{u.email}</td>
                      <td className="py-4 text-muted-foreground">{u.name || "-"}</td>
                      <td className="py-4">
                        <span className="font-mono bg-muted px-2 py-1 rounded">{u.tokens}</span>
                      </td>
                      <td className="py-4">
                        <Badge variant={u.role === "admin" ? "default" : "secondary"} className={u.role === "admin" ? "bg-gradient-to-r from-violet-500 to-purple-500" : ""}>
                          {u.role}
                        </Badge>
                      </td>
                      <td className="py-4">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleToggleRole(u.id, u.role)}
                          disabled={u.id === user.id}
                          className="hover:bg-violet-500/10 hover:text-violet-500"
                        >
                          {u.role === "admin" ? "Remove Admin" : "Make Admin"}
                        </Button>
                      </td>
                    </tr>
                  ))}
                  {users.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-12 text-center text-muted-foreground">No users found</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === "keys" && (
        <div className="space-y-6">
          {/* Create Key Form */}
          <Card className="shadow-xl border-2 border-dashed border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <IconKey />
                Create New API Key
              </CardTitle>
              <CardDescription>Generate a new API key for users or system access</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateKey} className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="name">Key Name</Label>
                    <Input id="name" value={keyForm.name} onChange={(e) => setKeyForm({ ...keyForm, name: e.target.value })} placeholder="e.g., Production API Key" required className="bg-background" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="type">Key Type</Label>
                    <select id="type" value={keyForm.type} onChange={(e) => setKeyForm({ ...keyForm, type: e.target.value as "standard" | "master" | "guest" })} className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                      <option value="standard">Standard (User-linked)</option>
                      <option value="master">Master (Full access)</option>
                      <option value="guest">Guest (Limited tokens)</option>
                    </select>
                  </div>
                  {keyForm.type === "standard" && (
                    <div className="space-y-2">
                      <Label htmlFor="user_id">User</Label>
                      <select id="user_id" value={keyForm.user_id} onChange={(e) => setKeyForm({ ...keyForm, user_id: e.target.value })} className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm" required>
                        <option value="">Select user...</option>
                        {users.map((u) => (
                          <option key={u.id} value={u.id}>{u.email}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  <div className="space-y-2">
                    <Label htmlFor="token_limit">Token Limit</Label>
                    <Input id="token_limit" type="number" value={keyForm.token_limit} onChange={(e) => setKeyForm({ ...keyForm, token_limit: e.target.value })} placeholder="Unlimited" min={1} className="bg-background" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="expires_in_days">Expires (days)</Label>
                    <Input id="expires_in_days" type="number" value={keyForm.expires_in_days} onChange={(e) => setKeyForm({ ...keyForm, expires_in_days: e.target.value })} placeholder="Never" min={1} max={365} className="bg-background" />
                  </div>
                </div>
                <Button type="submit" disabled={loading} className="bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600">
                  {loading ? "Creating..." : "Generate API Key"}
                </Button>
              </form>

              {newKey && (
                <div className="mt-6 p-6 rounded-xl border-2 border-emerald-500/50 bg-emerald-500/10">
                  <div className="flex items-center gap-2 mb-3">
                    <IconCheck />
                    <p className="font-semibold text-emerald-500">API Key Created!</p>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">Copy this key now - it won&apos;t be shown again!</p>
                  <code className="block p-4 rounded-lg bg-black/50 font-mono text-sm break-all text-emerald-400">{newKey.key}</code>
                  <Button variant="outline" size="sm" className="mt-3" onClick={() => { navigator.clipboard.writeText(newKey.key); toast.success("Copied!"); }}>
                    Copy to Clipboard
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* API Keys List */}
          <Card className="shadow-xl">
            <CardHeader>
              <CardTitle>Active API Keys</CardTitle>
              <CardDescription>Manage existing API keys</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Name</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Prefix</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Type</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">User</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Usage</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Status</th>
                      <th className="pb-4 text-left font-semibold text-muted-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {apiKeys.map((k) => (
                      <tr key={k.id} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
                        <td className="py-4 font-medium">{k.name}</td>
                        <td className="py-4 font-mono text-xs bg-muted/50 rounded px-2 py-1 w-fit">{k.prefix}...</td>
                        <td className="py-4">
                          <Badge variant={k.type === "master" ? "destructive" : k.type === "guest" ? "secondary" : "default"}>
                            {k.type}
                          </Badge>
                        </td>
                        <td className="py-4 text-muted-foreground">{k.user_email || "-"}</td>
                        <td className="py-4">
                          <span className="font-mono">{k.tokens_used}</span>
                          <span className="text-muted-foreground">{k.token_limit ? ` / ${k.token_limit}` : " / ∞"}</span>
                        </td>
                        <td className="py-4">
                          <Badge variant={k.is_active ? "default" : "secondary"} className={k.is_active ? "bg-emerald-500" : ""}>
                            {k.is_active ? "Active" : "Revoked"}
                          </Badge>
                        </td>
                        <td className="py-4">
                          {k.is_active && (
                            <Button variant="ghost" size="sm" onClick={() => handleRevokeKey(k.id)} className="text-destructive hover:text-destructive hover:bg-destructive/10">
                              Revoke
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                    {apiKeys.length === 0 && (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-muted-foreground">No API keys found</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "tools" && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Test Verification Tool */}
          <Card className="shadow-xl border-t-4 border-t-blue-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <IconBeaker />
                API Test Suite
              </CardTitle>
              <CardDescription>Run predefined tests to verify API functionality</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Test Type</Label>
                <div className="grid grid-cols-3 gap-2">
                  {(["simple", "complex", "hallucination"] as const).map((type) => (
                    <button
                      key={type}
                      onClick={() => setSelectedTestType(type)}
                      className={`p-3 rounded-lg border-2 text-sm font-medium transition-all ${
                        selectedTestType === type
                          ? "border-blue-500 bg-blue-500/10 text-blue-500"
                          : "border-border hover:border-blue-500/50"
                      }`}
                    >
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  {selectedTestType === "simple" && "Basic factual claims about the Eiffel Tower"}
                  {selectedTestType === "complex" && "Multiple related claims about Einstein"}
                  {selectedTestType === "hallucination" && "Intentionally false claims for testing detection"}
                </p>
              </div>

              <Button onClick={handleRunTest} disabled={testRunning} className="w-full bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600">
                {testRunning ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                    Running Test...
                  </>
                ) : (
                  "Run Verification Test"
                )}
              </Button>

              {testResult && (
                <div className="mt-4 space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-lg bg-muted">
                    <div>
                      <p className="text-sm text-muted-foreground">Verification Score</p>
                      <p className={`text-3xl font-bold ${testResult.verification_score >= 0.7 ? "text-emerald-500" : testResult.verification_score >= 0.4 ? "text-amber-500" : "text-red-500"}`}>
                        {(testResult.verification_score * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Duration</p>
                      <p className="text-lg font-semibold">{testResult.duration_ms.toFixed(0)}ms</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="font-semibold">Claims Analyzed ({testResult.claims_found})</p>
                    {testResult.results.map((result, i) => (
                      <div key={i} className={`p-3 rounded-lg border ${result.verdict === "true" ? "border-emerald-500/30 bg-emerald-500/5" : "border-red-500/30 bg-red-500/5"}`}>
                        <div className="flex items-start gap-2">
                          <div className={`mt-0.5 ${result.verdict === "true" ? "text-emerald-500" : "text-red-500"}`}>
                            {result.verdict === "true" ? <IconCheck /> : <IconX />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm">{result.claim}</p>
                            <p className="text-xs text-muted-foreground mt-1">{result.evidence}</p>
                            <p className="text-xs mt-1">
                              <span className="text-muted-foreground">Confidence:</span>{" "}
                              <span className="font-medium">{(result.confidence * 100).toFixed(0)}%</span>
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Token Grant Tool */}
          <Card className="shadow-xl border-t-4 border-t-emerald-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <IconGift />
                Grant Tokens
              </CardTitle>
              <CardDescription>Award tokens to users for testing, promotions, or compensation</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleGrantTokens} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="grant-email">User Email</Label>
                  <Input
                    id="grant-email"
                    type="email"
                    value={grantEmail}
                    onChange={(e) => setGrantEmail(e.target.value)}
                    placeholder="user@example.com"
                    required
                    className="bg-background"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="grant-tokens">Token Amount</Label>
                  <Input
                    id="grant-tokens"
                    type="number"
                    value={grantTokens}
                    onChange={(e) => setGrantTokens(e.target.value)}
                    placeholder="100"
                    min={1}
                    max={10000}
                    required
                    className="bg-background"
                  />
                  <p className="text-xs text-muted-foreground">Enter between 1 and 10,000 tokens</p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="grant-reason">Reason (optional)</Label>
                  <Input
                    id="grant-reason"
                    value={grantReason}
                    onChange={(e) => setGrantReason(e.target.value)}
                    placeholder="e.g., Beta testing reward"
                    className="bg-background"
                  />
                </div>
                <Button type="submit" disabled={grantLoading} className="w-full bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600">
                  {grantLoading ? "Granting..." : "Grant Tokens"}
                </Button>
              </form>

              {/* Quick Grant Buttons */}
              <div className="mt-6 pt-6 border-t border-border">
                <p className="text-sm font-medium mb-3">Quick Grant Presets</p>
                <div className="grid grid-cols-3 gap-2">
                  {[10, 50, 100].map((amount) => (
                    <Button
                      key={amount}
                      variant="outline"
                      size="sm"
                      onClick={() => setGrantTokens(amount.toString())}
                      className="hover:border-emerald-500 hover:text-emerald-500"
                    >
                      +{amount}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
