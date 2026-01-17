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
  ohi_tokens: number;
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

interface AdminDashboardClientProps {
  user: User;
}

export default function AdminDashboardClient({ user }: AdminDashboardClientProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "users" | "keys">("overview");
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

  // Form state for creating keys
  const [keyForm, setKeyForm] = useState({
    name: "",
    type: "standard" as "standard" | "master" | "guest",
    user_id: "",
    token_limit: "",
    expires_in_days: "",
  });

  // Fetch users
  const fetchUsers = useCallback(async () => {
    try {
      const res = await fetch("/api/ohi/admin/users", {
        headers: {
          "X-User-Id": user.id,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users || []);
      }
    } catch (error) {
      console.error("Failed to fetch users:", error);
    }
  }, [user.id]);

  // Fetch API keys
  const fetchApiKeys = useCallback(async () => {
    try {
      const res = await fetch("/api/ohi/admin/keys", {
        headers: {
          "X-User-Id": user.id,
        },
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
      const res = await fetch("/api/ohi/admin/logs/stats", {
        headers: {
          "X-User-Id": user.id,
        },
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

    // Refresh stats every 30 seconds
    const statsInterval = setInterval(fetchLogStats, 30000);
    return () => clearInterval(statsInterval);
  }, [fetchUsers, fetchApiKeys, fetchLogStats]);

  // Connect to live logs SSE stream
  useEffect(() => {
    // Only connect when on overview tab
    if (activeTab !== "overview") {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
        setSseConnected(false);
      }
      return;
    }

    // First, fetch recent logs
    const fetchRecentLogs = async () => {
      try {
        const res = await fetch("/api/ohi/admin/logs/recent?limit=50", {
          headers: {
            "X-User-Id": user.id,
          },
        });
        if (res.ok) {
          const data = await res.json();
          setLogs(data.reverse()); // Oldest first, newest at bottom
        }
      } catch (error) {
        console.error("Failed to fetch recent logs:", error);
      }
    };

    fetchRecentLogs();

    // Then connect to SSE stream
    // Note: We need to construct the URL with auth in a way the server can accept
    // Since SSE doesn't support custom headers, we'll pass via query param
    const sseUrl = `/api/ohi/admin/logs/stream`;

    const connectSSE = () => {
      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setSseConnected(true);
        setSseError(null);
        console.log("SSE connected to live logs");
      };

      eventSource.onmessage = (event) => {
        try {
          const logEntry: LogEntry = JSON.parse(event.data);

          // Skip heartbeat messages
          if (logEntry.type === "system" && logEntry.message === "heartbeat") {
            return;
          }

          setLogs((prev) => {
            const updated = [...prev, logEntry].slice(-200); // Keep last 200 logs
            return updated;
          });

          // Auto-scroll to bottom
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

      eventSource.onerror = (error) => {
        console.error("SSE error:", error);
        setSseConnected(false);
        setSseError("Connection lost. Reconnecting...");

        // Close and try to reconnect after delay
        eventSource.close();
        eventSourceRef.current = null;

        setTimeout(() => {
          if (activeTab === "overview") {
            connectSSE();
          }
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

    if (log.method && log.path) {
      parts.push(`${log.method} ${log.path}`);
    }

    if (log.status_code) {
      parts.push(`- ${log.status_code}`);
    }

    if (log.duration_ms !== null) {
      parts.push(`(${log.duration_ms.toFixed(0)}ms)`);
    }

    if (log.key_prefix) {
      parts.push(`[${log.key_prefix}...]`);
    }

    if (log.message && !log.method) {
      parts.push(log.message);
    }

    return parts.join(" ") || log.message;
  };

  // Get log color based on level/status
  const getLogColor = (log: LogEntry): string => {
    if (log.level === "error") return "text-red-400";
    if (log.level === "warning") return "text-yellow-400";
    if (log.type === "auth") return "text-blue-400";
    if (log.type === "health") return "text-gray-500";
    if (log.status_code && log.status_code >= 500) return "text-red-400";
    if (log.status_code && log.status_code >= 400) return "text-yellow-400";
    return "text-green-400";
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

      if (keyForm.type === "standard" && keyForm.user_id) {
        body.user_id = keyForm.user_id;
      }

      if (keyForm.token_limit) {
        body.token_limit = parseInt(keyForm.token_limit, 10);
      }

      if (keyForm.expires_in_days) {
        body.expires_in_days = parseInt(keyForm.expires_in_days, 10);
      }

      const res = await fetch("/api/ohi/admin/keys", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": user.id,
        },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const data = await res.json();
        setNewKey(data);
        toast.success("API key created successfully!");
        fetchApiKeys();
        setKeyForm({
          name: "",
          type: "standard",
          user_id: "",
          token_limit: "",
          expires_in_days: "",
        });
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
      const res = await fetch(`/api/ohi/admin/keys/${keyId}`, {
        method: "DELETE",
        headers: {
          "X-User-Id": user.id,
        },
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
      const res = await fetch(`/api/ohi/admin/users/${userId}/role`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": user.id,
        },
        body: JSON.stringify({ role: newRole }),
      });

      if (res.ok) {
        toast.success(`User role updated to ${newRole}`);
        fetchUsers();
      } else {
        toast.error("Failed to update user role");
      }
    } catch (error) {
      toast.error("Failed to update user role");
      console.error(error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Admin Dashboard</h1>
          <p className="text-muted-foreground">
            Manage users, API keys, and monitor system activity
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{users.length}</div>
            <p className="text-xs text-muted-foreground">
              {users.filter((u) => u.role === "admin").length} admins
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active API Keys</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {apiKeys.filter((k) => k.is_active).length}
            </div>
            <p className="text-xs text-muted-foreground">
              {apiKeys.filter((k) => k.type === "master").length} master,{" "}
              {apiKeys.filter((k) => k.type === "guest").length} guest
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Token Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {apiKeys.reduce((sum, k) => sum + k.tokens_used, 0).toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Across all API keys
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 border-b border-border">
        {(["overview", "users", "keys"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? "border-b-2 border-primary text-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* Stats Row */}
          {logStats && (
            <div className="grid gap-4 md:grid-cols-4">
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="text-2xl font-bold">{logStats.total_requests.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">Total Requests</p>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="text-2xl font-bold text-red-500">{logStats.total_errors.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">Total Errors</p>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="text-2xl font-bold">{logStats.requests_per_minute.toFixed(1)}</div>
                  <p className="text-xs text-muted-foreground">Requests/Min</p>
                </CardContent>
              </Card>
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="text-2xl font-bold">{logStats.active_subscribers}</div>
                  <p className="text-xs text-muted-foreground">Active Viewers</p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Live Logs */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    Live API Logs
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${
                        sseConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
                      }`}
                    />
                  </CardTitle>
                  <CardDescription>
                    {sseConnected
                      ? "Real-time API activity stream"
                      : sseError || "Connecting..."}
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setLogs([])}
                  >
                    Clear
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (logContainerRef.current) {
                        logContainerRef.current.scrollTo({
                          top: logContainerRef.current.scrollHeight,
                          behavior: "smooth",
                        });
                      }
                    }}
                  >
                    Scroll to Bottom
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div
                ref={logContainerRef}
                className="h-[500px] overflow-auto rounded-md bg-black/95 p-4 font-mono text-xs"
              >
                {logs.length === 0 ? (
                  <div className="flex h-full items-center justify-center text-gray-500">
                    {sseConnected
                      ? "Waiting for API requests..."
                      : "Connecting to log stream..."}
                  </div>
                ) : (
                  logs.map((log) => (
                    <div
                      key={log.id}
                      className={`py-0.5 leading-relaxed ${getLogColor(log)}`}
                    >
                      <span className="text-gray-600">
                        [{new Date(log.timestamp).toLocaleTimeString()}]
                      </span>{" "}
                      <span className="text-gray-500">[{log.type.toUpperCase().padEnd(8)}]</span>{" "}
                      {formatLogEntry(log)}
                    </div>
                  ))
                )}
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>{logs.length} log entries</span>
                <span>
                  {sseConnected ? (
                    <span className="text-green-500">Connected</span>
                  ) : (
                    <span className="text-yellow-500">Reconnecting...</span>
                  )}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === "users" && (
        <Card>
          <CardHeader>
            <CardTitle>User Management</CardTitle>
            <CardDescription>View and manage user accounts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-3 text-left font-medium">Email</th>
                    <th className="pb-3 text-left font-medium">Name</th>
                    <th className="pb-3 text-left font-medium">Tokens</th>
                    <th className="pb-3 text-left font-medium">Role</th>
                    <th className="pb-3 text-left font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.id} className="border-b border-border/50">
                      <td className="py-3">{u.email}</td>
                      <td className="py-3">{u.name || "-"}</td>
                      <td className="py-3">{u.ohi_tokens}</td>
                      <td className="py-3">
                        <Badge
                          variant={u.role === "admin" ? "default" : "secondary"}
                        >
                          {u.role}
                        </Badge>
                      </td>
                      <td className="py-3">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleToggleRole(u.id, u.role)}
                          disabled={u.id === user.id}
                        >
                          {u.role === "admin" ? "Remove Admin" : "Make Admin"}
                        </Button>
                      </td>
                    </tr>
                  ))}
                  {users.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-muted-foreground">
                        No users found
                      </td>
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
          <Card>
            <CardHeader>
              <CardTitle>Create API Key</CardTitle>
              <CardDescription>
                Generate a new API key for users or system access
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleCreateKey} className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="name">Key Name</Label>
                    <Input
                      id="name"
                      value={keyForm.name}
                      onChange={(e) =>
                        setKeyForm({ ...keyForm, name: e.target.value })
                      }
                      placeholder="e.g., Production API Key"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="type">Key Type</Label>
                    <select
                      id="type"
                      value={keyForm.type}
                      onChange={(e) =>
                        setKeyForm({
                          ...keyForm,
                          type: e.target.value as "standard" | "master" | "guest",
                        })
                      }
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    >
                      <option value="standard">Standard (User-linked)</option>
                      <option value="master">Master (Full access)</option>
                      <option value="guest">Guest (Limited tokens)</option>
                    </select>
                  </div>
                  {keyForm.type === "standard" && (
                    <div className="space-y-2">
                      <Label htmlFor="user_id">User ID</Label>
                      <select
                        id="user_id"
                        value={keyForm.user_id}
                        onChange={(e) =>
                          setKeyForm({ ...keyForm, user_id: e.target.value })
                        }
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        required
                      >
                        <option value="">Select user...</option>
                        {users.map((u) => (
                          <option key={u.id} value={u.id}>
                            {u.email}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  <div className="space-y-2">
                    <Label htmlFor="token_limit">Token Limit (optional)</Label>
                    <Input
                      id="token_limit"
                      type="number"
                      value={keyForm.token_limit}
                      onChange={(e) =>
                        setKeyForm({ ...keyForm, token_limit: e.target.value })
                      }
                      placeholder="Leave empty for unlimited"
                      min={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="expires_in_days">
                      Expires in Days (optional)
                    </Label>
                    <Input
                      id="expires_in_days"
                      type="number"
                      value={keyForm.expires_in_days}
                      onChange={(e) =>
                        setKeyForm({ ...keyForm, expires_in_days: e.target.value })
                      }
                      placeholder="Leave empty for non-expiring"
                      min={1}
                      max={365}
                    />
                  </div>
                </div>
                <Button type="submit" disabled={loading}>
                  {loading ? "Creating..." : "Generate API Key"}
                </Button>
              </form>

              {/* Show newly created key */}
              {newKey && (
                <div className="mt-6 rounded-md border border-green-500/50 bg-green-500/10 p-4">
                  <p className="mb-2 font-medium text-green-500">
                    API Key Created Successfully!
                  </p>
                  <p className="mb-4 text-sm text-muted-foreground">
                    Copy this key now. It won&apos;t be shown again!
                  </p>
                  <code className="block rounded bg-black/50 p-3 font-mono text-sm break-all">
                    {newKey.key}
                  </code>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2"
                    onClick={() => {
                      navigator.clipboard.writeText(newKey.key);
                      toast.success("Copied to clipboard!");
                    }}
                  >
                    Copy to Clipboard
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* API Keys List */}
          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
              <CardDescription>Manage existing API keys</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="pb-3 text-left font-medium">Name</th>
                      <th className="pb-3 text-left font-medium">Prefix</th>
                      <th className="pb-3 text-left font-medium">Type</th>
                      <th className="pb-3 text-left font-medium">User</th>
                      <th className="pb-3 text-left font-medium">Usage</th>
                      <th className="pb-3 text-left font-medium">Status</th>
                      <th className="pb-3 text-left font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {apiKeys.map((k) => (
                      <tr key={k.id} className="border-b border-border/50">
                        <td className="py-3">{k.name}</td>
                        <td className="py-3 font-mono text-xs">{k.prefix}...</td>
                        <td className="py-3">
                          <Badge
                            variant={
                              k.type === "master"
                                ? "destructive"
                                : k.type === "guest"
                                ? "secondary"
                                : "default"
                            }
                          >
                            {k.type}
                          </Badge>
                        </td>
                        <td className="py-3">{k.user_email || "-"}</td>
                        <td className="py-3">
                          {k.tokens_used}
                          {k.token_limit ? ` / ${k.token_limit}` : " (unlimited)"}
                        </td>
                        <td className="py-3">
                          <Badge variant={k.is_active ? "default" : "secondary"}>
                            {k.is_active ? "Active" : "Revoked"}
                          </Badge>
                        </td>
                        <td className="py-3">
                          {k.is_active && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-destructive hover:text-destructive"
                              onClick={() => handleRevokeKey(k.id)}
                            >
                              Revoke
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                    {apiKeys.length === 0 && (
                      <tr>
                        <td colSpan={7} className="py-8 text-center text-muted-foreground">
                          No API keys found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
