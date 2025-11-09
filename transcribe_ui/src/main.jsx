// src/main.jsx
import React from "react";
import { createRoot } from "react-dom/client";
import UploadPage from "./UploadPage.jsx";
import HistoryPage from "./HistoryPage.jsx";
import AuthPanel from "./AuthPanel.jsx";
import { getCurrentUser, logout } from "./lib/auth";
import "./index.css";

/* --------------------- Error Boundary --------------------- */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(err, info) {
    console.error("Render error:", err, info);
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, fontFamily: "monospace", color: "#b00020" }}>
          <h3>UI crashed</h3>
          <pre>{String(this.state.error.stack || this.state.error)}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ------------------------ Router -------------------------- */
function Router() {
  const [hash, setHash] = React.useState(window.location.hash || "#/");
  React.useEffect(() => {
    const onHash = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  if (hash.startsWith("#/history")) return <HistoryPage />;
  return <UploadPage />; // default
}

/* ------------------------- TopBar -------------------------- */
/* Minimal, unobtrusive strip above your existing page header */
function TopBar({ user, onLogout }) {
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 16px",
        borderBottom: "1px solid rgba(0,0,0,0.06)",
        background: "rgba(255,255,255,0.85)",
        backdropFilter: "saturate(120%) blur(4px)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      {/* Left: Welcome */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <div
          aria-hidden
          style={{
            width: 28,
            height: 28,
            borderRadius: "50%",
            background: "#3B82F6",
            color: "#fff",
            display: "grid",
            placeItems: "center",
            fontSize: 14,
            fontWeight: 600,
            flex: "0 0 auto",
          }}
          title={user?.full_name || user?.email || "You"}
        >
          {(user?.full_name || user?.email || "Y").trim().slice(0, 1).toUpperCase()}
        </div>
        <div
          style={{
            fontSize: 14,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            color: "#111827",
          }}
        >
          Welcome,&nbsp;
          <strong style={{ color: "#3B82F6" }}>
            {user?.full_name || user?.email || "You"}
          </strong>
        </div>
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Right: Logout */}
      <button
        className="btn"
        onClick={onLogout}
        style={{
          padding: "6px 14px",
          borderRadius: 8,
          border: "1px solid rgba(0,0,0,0.1)",
          background: "#ffffff",
          fontWeight: 600,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          cursor: "pointer",
        }}
      >
        Logout
      </button>
    </header>
  );
}

/* --------------------------- App --------------------------- */
function App() {
  const [user, setUser] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    (async () => {
      const me = await getCurrentUser();
      setUser(me);
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return <div style={{ padding: 24 }}>Loading…</div>;
  }

  if (!user) {
    return <AuthPanel onAuth={async () => setUser(await getCurrentUser())} />;
  }

  return (
    <>
      <TopBar
        user={user}
        onLogout={() => {
          logout();
          setUser(null);
        }}
      />
      <Router />
    </>
  );
}

/* ------------------------ Mount App ------------------------ */
const rootEl = document.getElementById("root");
createRoot(rootEl).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
