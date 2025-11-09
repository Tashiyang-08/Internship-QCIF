export type FetchOptions = RequestInit & {
  okStatuses?: number[]; // extra HTTP codes to treat as OK
};

export async function fetchWithLog(url: string, opts: FetchOptions = {}) {
  const { okStatuses = [], ...rest } = opts;

  // IMPORTANT: use the URL exactly as passed in.
  // Your Vite proxy already maps "/api" -> backend.
  const fullUrl = url;

  const method = rest.method || "GET";
  console.debug(`[fetch] ${method} ${fullUrl}`);

  let res: Response;
  try {
    res = await fetch(fullUrl, rest);
  } catch (e: any) {
    console.error("[fetch] network error:", fullUrl, e);
    throw new Error(`Network error while calling ${fullUrl}: ${e?.message || String(e)}`);
  }

  const ct = res.headers.get("content-type") || "";
  const isOk = res.ok || okStatuses.includes(res.status);

  let payload: any = null;
  try {
    payload = ct.includes("application/json") ? await res.json() : await res.text();
    if (typeof payload === "string" && !payload.trim()) payload = null;
  } catch (err) {
    console.warn("[fetch] failed to parse response:", err);
  }

  if (!isOk) {
    console.error("[fetch] http error", res.status, fullUrl, payload);
    const detail =
      typeof payload === "string"
        ? payload
        : payload?.detail || payload?.error || JSON.stringify(payload);
    throw new Error(`HTTP ${res.status} at ${fullUrl}: ${detail}`);
  }

  console.debug("[fetch] ok", res.status, fullUrl, payload);
  return payload;
}
