import React, { useEffect, useState } from "react";
import { resetPassword } from "./lib/api";

export default function ResetPassword() {
  const [token, setToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  // Read token from URL or hash
  useEffect(() => {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const hashParams = new URLSearchParams(window.location.hash.split("?")[1]);
      const t = urlParams.get("token") || hashParams.get("token") || "";
      setToken(t);
    } catch (err) {
      console.error("Token parsing failed", err);
    }
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setStatus("");

    if (!token) return setError("Invalid or missing reset token.");
    if (!newPassword) return setError("Please enter a new password.");
    if (newPassword !== confirmPassword)
      return setError("Passwords do not match.");

    try {
      setStatus("Resetting password...");
      await resetPassword(token, newPassword);
      setStatus("Password reset successful! Redirecting...");
      setDone(true);
      setTimeout(() => (window.location.href = "/#/"), 2500);
    } catch (err) {
      console.error(err);
      setError(err?.message || "Failed to reset password.");
    } finally {
      setStatus("");
    }
  }

  return (
    <div
      style={{
        maxWidth: 400,
        margin: "80px auto",
        padding: "20px",
        borderRadius: "8px",
        background: "#f9fafb",
        boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
        fontFamily: "sans-serif",
      }}
    >
      <h2 style={{ textAlign: "center", marginBottom: 20 }}>
        🔑 Reset Password
      </h2>

      {!done ? (
        <form onSubmit={handleSubmit}>
          <input
            type="password"
            placeholder="New password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            style={{
              width: "100%",
              padding: 8,
              marginBottom: 10,
              border: "1px solid #ddd",
              borderRadius: 4,
            }}
          />
          <input
            type="password"
            placeholder="Confirm new password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            style={{
              width: "100%",
              padding: 8,
              marginBottom: 10,
              border: "1px solid #ddd",
              borderRadius: 4,
            }}
          />

          <button
            type="submit"
            style={{
              width: "100%",
              padding: "10px 0",
              backgroundColor: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontWeight: "bold",
            }}
          >
            Reset Password
          </button>
        </form>
      ) : (
        <p style={{ textAlign: "center", color: "green" }}>
          Password reset successful. Redirecting…
        </p>
      )}

      {status && (
        <p style={{ color: "#2563eb", textAlign: "center", marginTop: 10 }}>
          {status}
        </p>
      )}
      {error && (
        <p style={{ color: "red", textAlign: "center", marginTop: 10 }}>
          {error}
        </p>
      )}
    </div>
  );
}
