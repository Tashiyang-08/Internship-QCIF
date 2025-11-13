// src/lib/auth.ts
// Handles login, registration, logout, and JWT token management

// Use Vite env if provided (e.g., VITE_API_BASE="http://127.0.0.1:8000"),
// otherwise fall back to "/api" (works with Vite proxy).
const API_BASE = (import.meta as any)?.env?.VITE_API_BASE || "/api";
const TOKEN_KEY = "access_token";

export interface TokenResponse {
  access_token: string;
  token_type: string; // "bearer"
}

export interface User {
  id: number;
  email: string;
  full_name?: string | null;
  is_admin?: boolean;
}

/* ----------------------------- url helper ------------------------------ */

function apiUrl(path: string): string {
  const base = API_BASE.endsWith("/") ? API_BASE.slice(0, -1) : API_BASE;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

/* ----------------------------- token utils ----------------------------- */

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {}
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

export function logout(): void {
  clearToken();
}

export function authHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Optional: decode JWT payload (for client-side expiry checks). */
export function decodeJwt(token: string): Record<string, unknown> | null {
  try {
    const [, payload] = token.split(".");
    if (!payload) return null;
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/* --------------------------------- auth -------------------------------- */

/**
 * Login with JSON body (email/password). This matches the FastAPI handler
 * you’re running and avoids the 422 errors from form-encoded payloads.
 */
export async function login(email: string, password: string): Promise<TokenResponse> {
  const res = await fetch(apiUrl("/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    throw new Error((await safeMessage(res)) || "Invalid email or password");
  }

  const data = (await res.json()) as TokenResponse;
  if (data?.access_token) setToken(data.access_token);
  return data;
}

/** Pass is_admin if you want to seed an admin (backend must allow it). */
export async function register(
  email: string,
  password: string,
  fullName?: string,
  is_admin?: boolean
): Promise<User> {
  const res = await fetch(apiUrl("/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify({
      email,
      password,
      full_name: fullName,
      ...(typeof is_admin === "boolean" ? { is_admin } : {}),
    }),
  });

  if (!res.ok) {
    throw new Error((await safeMessage(res)) || "Registration failed");
  }
  return (await res.json()) as User;
}

export async function getCurrentUser(): Promise<User | null> {
  const token = getToken();
  if (!token) return null;

  // Optional: proactively clear expired tokens client-side
  const payload = decodeJwt(token);
  const exp = typeof payload?.exp === "number" ? (payload.exp as number) : undefined;
  if (exp && Date.now() / 1000 > exp) {
    clearToken();
    return null;
  }

  const res = await fetch(apiUrl("/auth/me"), {
    headers: authHeader(),
  });

  if (res.status === 401) {
    // Token invalid/expired on server – log out.
    clearToken();
    return null;
  }
  if (!res.ok) {
    throw new Error((await safeMessage(res)) || "Failed to fetch current user");
  }

  return (await res.json()) as User;
}

/* ----------------------- password reset helpers ------------------------ */

export async function requestPasswordReset(email: string) {
  const res = await fetch(apiUrl("/auth/request-password-reset"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  // backend returns a generic success message (and link in dev)
  return res.json();
}

export async function resetPassword(token: string, new_password: string) {
  const res = await fetch(apiUrl("/auth/reset-password"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password }),
  });
  return res.json();
}

/* ------------------------------ error util ----------------------------- */

async function safeMessage(res: Response): Promise<string | null> {
  try {
    const txt = await res.text();
    if (!txt) return null;
    const j = JSON.parse(txt);
    // common FastAPI error shapes
    return j?.detail || j?.error || j?.message || null;
  } catch {
    return null;
  }
}
