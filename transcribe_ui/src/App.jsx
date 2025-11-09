// src/App.jsx
import React, { useEffect, useMemo, useState } from "react";
import { ensureApiReady, runTranscription } from "./lib/transcribe";
import {
  listModels,
  sendVerificationEmail,
  requestPasswordReset,
  resetPassword,
  verifyEmail,
} from "./lib/api";
import {
  login as apiLogin,
  registerUser as apiRegister,
  me as getCurrentUser,
  clearToken,
  authHeader,
  // token helpers (optional)
} from "./lib/api";
import "./App.css";

/* ------------------------- helpers ------------------------- */
function toMessage(e) {
  if (!e) return "Something went wrong.";
  if (typeof e === "string") return e;
  if (e.message && typeof e.message === "string") return e.message;
  if (e.detail && typeof e.detail === "string") return e.detail;
  if (e.error && typeof e.error === "string") return e.error;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

/* ------------------------- Inline Auth Panel ------------------------- */
function AuthPanel({ onAuth }) {
  const [mode, setMode] = useState(() => {
    const h = window.location.hash;
    if (h.startsWith("#/reset")) return "reset";
    return "login";
  }); // "login" | "register" | "forgot" | "reset"

  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [token, setToken] = useState(() => {
    try {
      const h = window.location.hash;
      if (h.includes("?")) {
        const qs = new URLSearchParams(h.split("?")[1]);
        return qs.get("token") || "";
      }
    } catch {}
    return "";
  });

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");

  // If opened via #/verify?token=..., verify immediately.
  useEffect(() => {
    (async () => {
      const h = window.location.hash;
      if (h.startsWith("#/verify?token=")) {
        try {
          setBusy(true);
          const qs = new URLSearchParams(h.split("?")[1]);
          const t = qs.get("token") || "";
          await verifyEmail(t);
          setMsg("Email verified. You can login now.");
          window.location.hash = "#/";
          setMode("login");
        } catch (e) {
          setError("Verification failed. Please request a new link.");
        } finally {
          setBusy(false);
        }
      }
    })();
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setMsg("");
    setBusy(true);
    try {
      if (mode === "register") {
        await apiRegister({ email, password, full_name: fullName });
        const r = await sendVerificationEmail(email);
        setMsg(r.link ? `Verification email sent.\nDev link: ${r.link}` : "Verification email sent. Check your inbox.");
        setMode("login");
        return;
      }
      if (mode === "login") {
        await apiLogin(email, password);
        onAuth?.();
        return;
      }
      if (mode === "forgot") {
        const r = await requestPasswordReset(email);
        setMsg(r.link ? `Reset email sent.\nDev link: ${r.link}` : "Reset email sent. Check your inbox.");
        setMode("login");
        return;
      }
      if (mode === "reset") {
        if (!token) throw new Error("Missing reset token.");
        await resetPassword(token, newPassword);
        setMsg("Password updated. You can login now.");
        setMode("login");
        return;
      }
    } catch (err) {
      setError(toMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel" style={{ maxWidth: 420, margin: "80px auto" }}>
      <h2 style={{ marginBottom: 12 }}>
        {mode === "login"
          ? "Login"
          : mode === "register"
          ? "Create an account"
          : mode === "forgot"
          ? "Forgot password"
          : "Reset password"}
      </h2>

      <form onSubmit={handleSubmit}>
        {mode === "register" && (
          <input
            className="box"
            type="text"
            placeholder="Full name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            style={{ width: "100%", marginBottom: 8 }}
          />
        )}

        {(mode === "login" || mode === "register" || mode === "forgot") && (
          <input
            className="box"
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ width: "100%", marginBottom: 8 }}
          />
        )}

        {(mode === "login" || mode === "register") && (
          <input
            className="box"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ width: "100%", marginBottom: 12 }}
          />
        )}

        {mode === "reset" && (
          <>
            <input
              className="box"
              type="text"
              placeholder="Reset token"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              style={{ width: "100%", marginBottom: 8 }}
            />
            <input
              className="box"
              type="password"
              placeholder="New password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              style={{ width: "100%", marginBottom: 12 }}
            />
          </>
        )}

        <button className="btn" type="submit" style={{ width: "100%" }} disabled={busy}>
          {busy
            ? "Please wait…"
            : mode === "login"
            ? "Login"
            : mode === "register"
            ? "Register"
            : mode === "forgot"
            ? "Send reset link"
            : "Reset password"}
        </button>
      </form>

      <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
        {mode !== "login" && (
          <button className="btn" onClick={() => setMode("login")} disabled={busy}>
            Back to login
          </button>
        )}
        {mode === "login" && (
          <>
            <button className="btn" onClick={() => setMode("register")} disabled={busy}>
              Create an account
            </button>
            <button className="btn" onClick={() => setMode("forgot")} disabled={busy}>
              Forgot password?
            </button>
          </>
        )}
      </div>

      {msg && (
        <div className="panel" style={{ marginTop: 12, whiteSpace: "pre-wrap" }}>
          {msg}
        </div>
      )}
      {error && (
        <div className="panel error" style={{ marginTop: 12 }}>
          {error}
        </div>
      )}
      <small style={{ display: "block", marginTop: 8, opacity: 0.6 }}>
        We’ll never share your email.
      </small>
    </div>
  );
}
/* -------------------------------------------------------------------- */

// Small hash router
function useHashRoute() {
  const [route, setRoute] = useState(window.location.hash || "#/");
  useEffect(() => {
    const onHashChange = () => setRoute(window.location.hash || "#/");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  return [route, (h) => (window.location.hash = h)];
}

function TranscriptionPage() {
  const [apiReady, setApiReady] = useState(false);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("auto");
  const [file, setFile] = useState(null);
  const [summarize, setSummarize] = useState(true);
  const [diarize, setDiarize] = useState(false);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [diarizer, setDiarizer] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // Restore from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("transcribe_state");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.model) setModel(parsed.model);
        if (typeof parsed.summarize === "boolean") setSummarize(parsed.summarize);
        if (parsed.result) setResult(parsed.result);
      } catch {}
    }
  }, []);

  // Save to localStorage whenever result/model/summarize changes
  useEffect(() => {
    const state = { model, summarize, result };
    localStorage.setItem("transcribe_state", JSON.stringify(state));
  }, [model, summarize, result]);

  // API check + model list
  useEffect(() => {
    (async () => {
      const { ok } = await ensureApiReady();
      setApiReady(ok);
      try {
        const ms = await listModels();
        const uniq = Array.from(new Set(ms));
        const withoutAuto = uniq.filter((m) => m !== "auto");
        setModels(["auto", ...withoutAuto]);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  const prettyDuration = useMemo(() => {
    if (!result?.duration_sec) return "—";
    const s = Math.max(0, Number(result.duration_sec) || 0);
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return `${m}m ${r}s`;
  }, [result?.duration_sec]);

  async function handleTranscribe() {
    setError("");
    setResult(null);
    if (!file) return setError("Choose an audio/video file.");
    setBusy(true);
    try {
      const r = await runTranscription(file, model, {
        summarize,
        diarize,
        num_speakers: diarize ? numSpeakers : undefined,
        diarizer: diarize ? diarizer : undefined,
      });
      const { session, session_id, ...clean } = r || {};
      setResult(clean);

      // Persist to local history
      try {
        const history = JSON.parse(localStorage.getItem("history_data") || "[]");
        const newItem = {
          id: clean.id ?? Date.now(),
          filename: clean.filename || file.name,
          language: clean.language || "",
          duration_sec: clean.duration_sec || 0,
          text: clean.text || "",
          summary: clean.summary || "",
          model: clean.model || model,
          created_at: clean.created_at || new Date().toISOString(),
        };
        localStorage.setItem("history_data", JSON.stringify([newItem, ...history]));
      } catch {}
    } catch (e) {
      setError(toMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-wrap">
      <header className="app-header">
        <h1>
          Transcription <span style={{ color: "#3B82F6" }}>Services</span>
        </h1>
        <span className={`badge ${apiReady ? "ok" : "bad"}`}>
          API: {apiReady ? "ready" : "offline"}
        </span>
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => (window.location.hash = "#/history")}>
          History
        </button>
      </header>

      <section className="toolbar">
        <label>
          <strong>Default model:</strong>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="btn"
            style={{ marginLeft: 8 }}
            disabled={!apiReady || busy}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={summarize}
            onChange={(e) => setSummarize(e.target.checked)}
            disabled={busy}
          />
          Generate summary
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={diarize}
            onChange={(e) => setDiarize(e.target.checked)}
            disabled={busy}
          />
          Diarize speakers
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Speakers:</span>
          <input
            type="number"
            min={2}
            max={10}
            value={numSpeakers}
            onChange={(e) => setNumSpeakers(parseInt(e.target.value || "2", 10))}
            className="box"
            style={{ width: 80 }}
            disabled={busy || !diarize}
          />
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Diarizer:</span>
          <select
            value={diarizer}
            onChange={(e) => setDiarizer(e.target.value)}
            className="btn"
            disabled={busy || !diarize}
          >
            <option value="auto">auto</option>
            <option value="basic">basic</option>
            <option value="fallback">fallback</option>
          </select>
        </label>
      </section>

      <section className="toolbar">
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setError("");
          }}
          style={{ flex: 1 }}
          disabled={busy}
        />
        <button onClick={handleTranscribe} disabled={!apiReady || !file || busy} className="btn">
          {busy ? "Transcribing…" : "Transcribe"}
        </button>
      </section>

      {error && (
        <section className="panel error">
          <strong>Error:</strong> {error}
        </section>
      )}

      {result && (
        <div className="grid">
          <section className="panel">
            <h2>Transcript</h2>
            <div className="meta">
              <span>
                <strong>Model:</strong> {result.model || model}
              </span>
              <span>
                <strong>Language:</strong> {result.language || "—"}
              </span>
              <span>
                <strong>Duration:</strong> {prettyDuration}
              </span>
            </div>
            <pre className="box" style={{ whiteSpace: "pre-wrap", margin: 0 }}>
              {result.text || "(empty)"}
            </pre>
          </section>

          <section className="panel">
            <h2>Summary</h2>
            <div className="box" style={{ marginTop: 16 }}>
              {result.summary || "(no summary)"}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function HistoryPage() {
  const [items, setItems] = useState([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  // Fetch history
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setError("");
      try {
        const res = await fetch("/api/transcriptions", { headers: authHeader() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setItems(Array.isArray(data) ? data : []);
        localStorage.setItem("history_data", JSON.stringify(Array.isArray(data) ? data : []));
      } catch (e) {
        const history = JSON.parse(localStorage.getItem("history_data") || "[]");
        setItems(history);
        setError("Showing cached history (backend unavailable).");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (i) =>
        (i.filename || "").toLowerCase().includes(q) ||
        (i.text || "").toLowerCase().includes(q) ||
        (i.summary || "").toLowerCase().includes(q)
    );
  }, [items, search]);

  async function handleDelete(id, idx) {
    const prev = items;
    const next = prev.filter((_, i) => i !== idx);
    setItems(next);
    localStorage.setItem("history_data", JSON.stringify(next));
    try {
      const res = await fetch(`/api/transcriptions/${id}`, {
        method: "DELETE",
        headers: authHeader(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      setItems(prev);
      localStorage.setItem("history_data", JSON.stringify(prev));
      alert("Failed to delete on server. Restored the item.");
    }
  }

  return (
    <div className="app-wrap">
      <header className="app-header">
        <input
          type="text"
          placeholder="Search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="box"
          style={{ flex: 1, maxWidth: 350 }}
        />
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => (window.location.hash = "#/")}>
          Back
        </button>
      </header>

      {error && (
        <section className="panel error">
          <strong>Note:</strong> {error}
        </section>
      )}

      <section className="panel">
        {filtered.length === 0 ? (
          <div className="box">(No history)</div>
        ) : (
          filtered.map((r, i) => (
            <div key={r.id ?? i} className="file-row">
              <div style={{ display: "grid", gap: 4 }}>
                <span className="file-name">{r.filename}</span>
                <small style={{ opacity: 0.7 }}>
                  {r.model || "—"} · {(r.language || "—").toUpperCase()} ·{" "}
                  {r.duration_sec ? `${Math.round(r.duration_sec)}s` : "—"}
                </small>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  className="btn"
                  onClick={() => {
                    const text = (r.text || "").trim() || "(empty)";
                    window.alert(text.length > 1000 ? text.slice(0, 1000) + "…" : text);
                  }}
                >
                  View
                </button>
                <button className="btn" onClick={() => handleDelete(r.id, i)}>
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </section>
    </div>
  );
}

// App wrapper
export default function App() {
  const [route] = useHashRoute();
  const isHistory = route === "#/history";
  const [user, setUser] = useState(undefined);

  useEffect(() => {
    getCurrentUser()
      .then((u) => setUser(u || null))
      .catch(() => setUser(null));
  }, []);

  if (user === undefined) {
    return <div style={{ padding: 24 }}>Loading…</div>;
  }

  if (!user) {
    return (
      <AuthPanel
        onAuth={() => {
          getCurrentUser().then((u) => setUser(u || null));
        }}
      />
    );
  }

  return (
    <>
      <div className="app-header" style={{ justifyContent: "flex-end", gap: 8 }}>
        <span style={{ marginRight: "auto" }}>
          Welcome, {user.full_name || user.email}
        </span>
        <button
          className="btn"
          onClick={() => {
            clearToken();
            setUser(null);
          }}
        >
          Logout
        </button>
      </div>

      {/* Unverified banner with resend button (works only if server supports it) */}
      {(user.is_verified === false || user.is_verified === 0) && (
        <div className="panel error" style={{ margin: "12px auto", maxWidth: 960 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span>Your email is not verified.</span>
            <button
              className="btn"
              onClick={async () => {
                const r = await sendVerificationEmail(user.email);
                alert(
                  r.link ? `Verification link sent.\n\nDev link: ${r.link}` : r.message || "Verification email sent."
                );
              }}
            >
              Resend verification email
            </button>
          </div>
        </div>
      )}

      <div style={{ display: isHistory ? "none" : "block" }}>
        <TranscriptionPage />
      </div>
      <div style={{ display: isHistory ? "block" : "none" }}>
        <HistoryPage />
      </div>
    </>
  );
}
