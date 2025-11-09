import React, { useMemo, useState } from "react";
import { login, register } from "./lib/auth";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function AuthPanel({ onAuth }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // --- client-side validation (JS only) ---
  const issues = useMemo(() => {
    const errs = [];
    const e = email.trim();

    if (!EMAIL_RE.test(e)) errs.push("Enter a valid email address.");
    if (password.length < 6) errs.push("Password must be at least 6 characters.");
    if (mode === "register" && fullName.trim().length < 2) {
      errs.push("Please enter your full name.");
    }
    return errs;
  }, [email, password, mode, fullName]);

  const canSubmit = issues.length === 0 && !busy;

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");

    if (!canSubmit) {
      setError(issues[0] || "Please fix the form errors.");
      return;
    }

    setBusy(true);
    try {
      const normalizedEmail = email.trim().toLowerCase();
      if (mode === "register") {
        await register(normalizedEmail, password, fullName.trim());
      }
      await login(normalizedEmail, password);
      onAuth && onAuth();
    } catch (err) {
      setError((err && err.message) || "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel" style={{ maxWidth: 420, margin: "80px auto" }}>
      <h2 style={{ marginBottom: 12 }}>
        {mode === "login" ? "Login" : "Create an account"}
      </h2>

      <form onSubmit={handleSubmit} noValidate>
        {mode === "register" && (
          <input
            className="box"
            type="text"
            placeholder="Full name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            autoComplete="name"
            style={{ width: "100%", marginBottom: 8 }}
          />
        )}

        <input
          className="box"
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          inputMode="email"
          style={{ width: "100%", marginBottom: 6 }}
        />
        {!EMAIL_RE.test(email.trim()) && email.length > 0 && (
          <div style={{ color: "#c2410c", fontSize: 12, marginBottom: 6 }}>
            Example: name@example.com
          </div>
        )}

        <input
          className="box"
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          style={{ width: "100%", marginBottom: 6 }}
        />
        {password.length > 0 && password.length < 6 && (
          <div style={{ color: "#c2410c", fontSize: 12, marginBottom: 6 }}>
            At least 6 characters.
          </div>
        )}

        {(error || issues.length > 0) && (
          <div className="panel error" style={{ margin: "8px 0 10px" }}>
            {error || issues[0]}
          </div>
        )}

        <button
          className="btn"
          type="submit"
          style={{ width: "100%" }}
          disabled={!canSubmit}
          aria-disabled={!canSubmit}
          title={!canSubmit ? (busy ? "Please wait…" : "Fix the form errors") : ""}
        >
          {busy ? "Please wait…" : mode === "login" ? "Login" : "Register"}
        </button>
      </form>

      <button
        className="btn"
        style={{ width: "100%", marginTop: 8 }}
        onClick={() => {
          setMode((m) => (m === "login" ? "register" : "login"));
          setError("");
        }}
        disabled={busy}
      >
        {mode === "login" ? "Create an account" : "Back to login"}
      </button>

      <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
        We’ll never share your email.
      </div>
    </div>
  );
}
