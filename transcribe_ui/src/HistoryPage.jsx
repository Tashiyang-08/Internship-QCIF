// src/HistoryPage.jsx
import React from "react";
import {
  listTranscriptions,
  getTranscription,
  deleteTranscription,
} from "./lib/api";
import "./App.css";

const CACHE_KEY = "history_cache_v1";

export default function HistoryPage() {
  const [items, setItems] = React.useState(() => {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });
  const [query, setQuery] = React.useState("");
  const [selected, setSelected] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState("");

  React.useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr("");
      try {
        const rows = await listTranscriptions(0, 200);
        const normalized = (rows || []).map((r) => ({
          id: r?.id,
          filename: r?.filename || "unknown",
          duration_sec:
            typeof r?.duration_sec === "number"
              ? Math.max(0, Math.round(r.duration_sec))
              : null,
          created_at: r?.created_at || r?.ts || r?.time || null,
          language: r?.language || "",
          model: r?.model || "",
        }));
        if (!alive) return;
        setItems(normalized);
        localStorage.setItem(CACHE_KEY, JSON.stringify(normalized));
      } catch (e) {
        if (!alive) return;
        setErr(
          e?.message ? `Failed to load history: ${e.message}` : "Failed to load history."
        );
        // keep cached items so the UI still renders
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  // Robust local time formatter with timezone label
  function formatLocalTime(ts, { hour12 = true } = {}) {
    if (!ts) return "";
    try {
      let safe = String(ts).trim();
      if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}/.test(safe)) {
        safe = safe.replace(" ", "T") + "Z";
      }
      const d = new Date(safe);
      if (isNaN(d.getTime())) return String(ts);
      return new Intl.DateTimeFormat(undefined, {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12,
        timeZoneName: "short",
      }).format(d);
    } catch {
      return String(ts) || "";
    }
  }

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (it) =>
        (it.filename || "").toLowerCase().includes(q) ||
        formatLocalTime(it.created_at).toLowerCase().includes(q)
    );
  }, [items, query]);

  async function onView(id) {
    if (!id) return;
    setLoading(true);
    setErr("");
    try {
      const rec = await getTranscription(id);
      setSelected(rec || null);
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onDelete(id) {
    if (!id) return;
    if (!window.confirm("Delete this transcription permanently?")) return;
    setErr("");
    try {
      await deleteTranscription(id);
      setItems((prev) => {
        const next = prev.filter((x) => x.id !== id);
        localStorage.setItem(CACHE_KEY, JSON.stringify(next));
        return next;
      });
      if (selected?.id === id) setSelected(null);
    } catch (e) {
      setErr(e?.message || String(e));
    }
  }

  return (
    <div className="history-wrap">
      <div className="history-header">
        <input
          className="search"
          placeholder="Search history…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn" onClick={() => (window.location.hash = "#/")}>
          Back
        </button>
      </div>

      {err && (
        <div className="panel error" style={{ marginBottom: 12 }}>
          {err}
        </div>
      )}

      <div className="history-grid">
        {/* Left list */}
        <div className="panel">
          {loading && !items.length ? (
            <div className="muted">Loading…</div>
          ) : filtered.length ? (
            filtered.map((it) => (
              <div
                key={it.id ?? Math.random()}
                className={`hist-row ${selected?.id === it.id ? "active" : ""}`}
              >
                <div className="hist-main">
                  <div className="file-name">{it.filename}</div>
                  <div className="muted small">
                    {formatLocalTime(it.created_at, { hour12: true })}
                    {typeof it.duration_sec === "number" && <> • {it.duration_sec}s</>}
                  </div>
                </div>
                <div className="hist-actions">
                  <button className="btn" onClick={() => onView(it.id)}>View</button>
                  <button className="btn danger" onClick={() => onDelete(it.id)}>Delete</button>
                </div>
              </div>
            ))
          ) : (
            <div className="muted">(no matches)</div>
          )}
        </div>

        {/* Right preview */}
        <div className="panel">
          {!selected ? (
            <div className="muted">(select a transcription)</div>
          ) : (
            <>
              <h3 className="mb8">{selected.filename || "—"}</h3>
              <div className="meta mb12">
                {formatLocalTime(selected.created_at, { hour12: true })}
                {selected.duration_sec ? <> • {Math.round(selected.duration_sec)}s</> : null}
                {selected.language ? <> • {selected.language}</> : null}
                {selected.model ? <> • {selected.model}</> : null}
              </div>

              <h4 className="mb6">Transcript</h4>
              <div className="box mb16" style={{ whiteSpace: "pre-wrap" }}>
                {selected.text || "(empty)"}
              </div>

              <h4 className="mb6">Summary</h4>
              <div className="box" style={{ whiteSpace: "pre-wrap" }}>
                {selected.summary || "(no summary)"}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
