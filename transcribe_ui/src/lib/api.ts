// src/lib/api.ts
import { fetchWithLog } from "./fetchWithLog";

/* ------------------------------------------------------------------ */
/* Type Definitions                                                   */
/* ------------------------------------------------------------------ */

export type Segment = {
  start: number | string;
  end: number | string;
  text: string;
  speaker?: string; // e.g., "SPEAKER_00"
};

export type TranscriptionResponse = {
  id?: number;
  filename: string;
  duration_sec?: number;
  text: string;
  language?: string;
  segments?: Segment[];
  summary?: string;
  summary_points?: string[];
  model?: string;
  latency_sec?: number;
  created_at?: string;
  error?: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: "bearer";
};

export type UserMe = {
  id: number;
  email: string;
  full_name?: string | null;
  is_admin?: boolean;
  is_verified?: boolean | 0 | 1; // tolerate various backends
};

/* ------------------------------------------------------------------ */
/* Internal helpers                                                   */
/* ------------------------------------------------------------------ */

const MODELS_CACHE_KEY = "models_cache_v1";
const AUTH_TOKEN_KEY = "auth_token_v1";

function getToken(): string | null {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(AUTH_TOKEN_KEY, token);
    else localStorage.removeItem(AUTH_TOKEN_KEY);
  } catch {}
}

export function clearToken() {
  setToken(null);
}

export function authHeader(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function buildHeaders(extra?: HeadersInit, useAuth = false): HeadersInit {
  return {
    ...(useAuth ? authHeader() : {}),
    ...(extra || {}),
  };
}

function toMessage(e: any): string {
  if (!e) return "Something went wrong.";
  if (typeof e === "string") return e;
  if (e.message && typeof e.message === "string") return e.message;
  if (e.detail && typeof e.detail === "string") return e.detail;
  if (e.error && typeof e.error === "string") return e.error;
  if (e.message && typeof e.message === "object") {
    try {
      return JSON.stringify(e.message);
    } catch {}
  }
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

/* ------------------------------------------------------------------ */
/* Core Backend API Calls                                             */
/* ------------------------------------------------------------------ */

/** Ping backend */
export async function ping() {
  try {
    const res = await fetchWithLog("/api/ping");
    return res || { ok: false };
  } catch {
    return { ok: false };
  }
}

/** Get available ASR models from the backend. */
export async function listModels(): Promise<string[]> {
  try {
    const j = (await fetchWithLog("/api/models")) as { models?: unknown };
    const models: string[] = Array.isArray(j?.models)
      ? (j.models as unknown[]).map((m) => String(m))
      : [];

    const uniq = Array.from(new Set(models.filter(Boolean)));
    const ordered = uniq.includes("auto")
      ? ["auto", ...uniq.filter((m) => m !== "auto")]
      : ["auto", ...uniq];

    try {
      localStorage.setItem(MODELS_CACHE_KEY, JSON.stringify(ordered));
    } catch {}
    return ordered;
  } catch {
    try {
      const raw = localStorage.getItem(MODELS_CACHE_KEY);
      const cached = raw ? (JSON.parse(raw) as unknown[]) : [];
      if (Array.isArray(cached)) return cached.map((m) => String(m));
    } catch {}
    return ["auto"];
  }
}

/** Upload a file for transcription. */
export async function transcribe(
  file: File,
  model: string,
  options?: {
    language?: string;
    summarize?: boolean;
    diarize?: boolean;
    num_speakers?: number;
    diarizer?: "auto" | "basic" | "fallback";
  }
): Promise<TranscriptionResponse> {
  if (!file) throw new Error("No audio file provided.");
  if (!model) throw new Error("Missing 'model' parameter.");

  const form = new FormData();
  form.append("file", file);
  form.append("model", model);

  if (options?.language && options.language !== "auto") {
    form.append("language", options.language);
  }
  if (options?.summarize) {
    form.append("summarize", "true");
  }
  if (options?.diarize) {
    form.append("diarize", "true");
    if (options.num_speakers && Number.isFinite(options.num_speakers)) {
      form.append("num_speakers", String(options.num_speakers));
    }
    form.append("diarizer", options.diarizer ?? "auto");
  }

  const res = await fetchWithLog("/api/transcribe", {
    method: "POST",
    body: form,
    headers: buildHeaders(undefined, true),
  });

  if (res && typeof res === "object" && "error" in res && (res as any).error) {
    throw new Error(toMessage(res));
  }
  return res as TranscriptionResponse;
}

/* ------------------------------------------------------------------ */
/* History API                                                        */
/* ------------------------------------------------------------------ */

export async function listTranscriptions(offset = 0, limit = 50) {
  try {
    const res = await fetchWithLog(
      `/api/transcriptions?offset=${offset}&limit=${limit}`,
      { headers: buildHeaders(undefined, true) }
    );
    return Array.isArray(res) ? res : [];
  } catch (e) {
    console.error("listTranscriptions failed:", e);
    return [];
  }
}

export async function getTranscription(id: number) {
  try {
    const res = await fetchWithLog(`/api/transcriptions/${id}`, {
      headers: buildHeaders(undefined, true),
    });
    return res || null;
  } catch (e) {
    console.error("getTranscription failed:", e);
    throw e;
  }
}

export async function deleteTranscription(id: number) {
  try {
    const res = await fetchWithLog(`/api/transcriptions/${id}`, {
      method: "DELETE",
      headers: buildHeaders(undefined, true),
    });
    return res || { ok: true };
  } catch (e) {
    console.error("deleteTranscription failed:", e);
    throw e;
  }
}

/* ------------------------------------------------------------------ */
/* Auth API                                                           */
/* ------------------------------------------------------------------ */

export async function registerUser(opts: {
  email: string;
  password: string;
  full_name?: string;
  is_admin?: boolean;
}): Promise<UserMe> {
  const res = await fetchWithLog("/api/auth/register", {
    method: "POST",
    headers: buildHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(opts),
  });

  if ((res as any)?.detail) throw new Error(toMessage(res));
  return res as UserMe;
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const body = new URLSearchParams({ username: email, password });

  const res = await fetchWithLog("/api/auth/login", {
    method: "POST",
    headers: buildHeaders({ "Content-Type": "application/x-www-form-urlencoded" }),
    body,
  });

  const errMsg = toMessage(res);
  if (!res || !("access_token" in res)) {
    throw new Error(errMsg || "Invalid credentials");
  }

  const tr = res as TokenResponse;
  if (tr?.access_token) setToken(tr.access_token);
  return tr;
}

export async function me(): Promise<UserMe> {
  const res = await fetchWithLog("/api/auth/me", {
    headers: buildHeaders(undefined, true),
  });

  if ((res as any)?.detail) throw new Error(toMessage(res));
  return res as UserMe;
}

/** Optional: send verification email (works only if backend provides the endpoint). */
export async function sendVerificationEmail(email: string): Promise<{ message?: string; link?: string }> {
  try {
    const res = await fetchWithLog("/api/auth/send-verification", {
      method: "POST",
      headers: buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ email }),
    });
    if ((res as any)?.detail) throw new Error(toMessage(res));
    return res as any;
  } catch (e) {
    return { message: "Email verification not enabled on server." };
  }
}

/** Optional: verify email with token (works only if backend provides the endpoint). */
export async function verifyEmail(token: string): Promise<{ message?: string }> {
  try {
    const res = await fetchWithLog("/api/auth/verify-email", {
      method: "POST",
      headers: buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ token }),
    });
    if ((res as any)?.detail) throw new Error(toMessage(res));
    return res as any;
  } catch (e) {
    throw new Error("Verification failed.");
  }
}

/** Request password reset (backend endpoint required). */
export async function requestPasswordReset(email: string): Promise<{ message?: string; link?: string }> {
  const res = await fetchWithLog("/api/auth/request-password-reset", {
    method: "POST",
    headers: buildHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ email }),
  });
  if ((res as any)?.detail) throw new Error(toMessage(res));
  return res as any;
}

/** Reset password using token (backend endpoint required). */
export async function resetPassword(token: string, new_password: string): Promise<{ message?: string }> {
  const res = await fetchWithLog("/api/auth/reset-password", {
    method: "POST",
    headers: buildHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ token, new_password }),
  });
  if ((res as any)?.detail) throw new Error(toMessage(res));
  return res as any;
}

/* ------------------------------------------------------------------ */
/* Local Session Helpers                                              */
/* ------------------------------------------------------------------ */

const SESSION_KEY = "tsession-id";

export function newSession() {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {}
  return id;
}

export function getSession(): string {
  try {
    let id = localStorage.getItem(SESSION_KEY) || "";
    if (!id) id = newSession();
    return id;
  } catch {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }
}
