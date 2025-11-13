// src/UploadPage.jsx
import React from "react";
import { listModels, transcribe, transcribeSrt, getSession, newSession } from "./lib/api";
import "./App.css";

const STATE_KEY = "ts_ui_state_v3";

/* ---------- helpers to render [Speaker XX] with colors ---------- */
const palette = ["#2563EB","#059669","#DC2626","#7C3AED","#D97706","#0EA5E9","#16A34A","#EA580C"];
function colorForTag(tagText) {
  const m = /^\s*\[(?:Speaker\s+)?(\d{1,2})\]\s*/i.exec(tagText);
  if (m) {
    const idx = parseInt(m[1], 10) || 0;
    return palette[idx % palette.length];
  }
  let h = 0;
  for (let i = 0; i < tagText.length; i++) h = (h * 31 + tagText.charCodeAt(i)) >>> 0;
  return palette[h % palette.length];
}
function renderTaggedTranscript(text) {
  if (!text) return "(empty)";
  return text.split("\n").map((line, i) => {
    const m = /^\s*(\[[^\]]+\])\s*(.*)$/.exec(line);
    if (!m) return <div key={i} style={{ whiteSpace: "pre-wrap" }}>{line}</div>;
    const tag = m[1], rest = m[2] || "";
    const color = colorForTag(tag);
    return (
      <div key={i} style={{ whiteSpace: "pre-wrap" }}>
        <span style={{ display: "inline-block", minWidth: 120, fontWeight: 600, color }}>{tag}</span>{" "}
        <span>{rest}</span>
      </div>
    );
  });
}
/* ------------------------------------------------------------------ */

function makeObjectUrl(blobOrFile) {
  return URL.createObjectURL(blobOrFile);
}
function revokeUrl(url) {
  try { url && URL.revokeObjectURL(url); } catch {}
}

function baseName(name) {
  return (name || "audio").replace(/\.[a-z0-9]+$/i, "");
}

/** Serializable subset for localStorage */
function serializeFiles(files) {
  return files.map((f) => ({
    id: f.id,
    name: f.file ? f.file.name : f.name || "",
    status: f.status,
    chosenModel: f.chosenModel || null,
    res: f.res
      ? {
          id: f.res.id,
          filename: f.res.filename,
          language: f.res.language,
          duration_sec: f.res.duration_sec,
          text: f.res.text,
          summary: f.res.summary,
          model: f.res.model,
          created_at: f.res.created_at,
        }
      : null,
    err: f.err || "",
    // downloads
    downloads: {
      txtUrl: f.downloads?.txtUrl || null,
      srtUrl: f.downloads?.srtUrl || null,
      // originalUrl is regenerated when file present
    },
  }));
}
function reviveFiles(saved) {
  return (saved || []).map((f) => ({
    id: f.id,
    file: null, // cannot revive raw File
    name: f.name,
    status: f.status || "idle",
    chosenModel: f.chosenModel || null,
    res: f.res || null,
    err: f.err || "",
    downloads: {
      txtUrl: f.downloads?.txtUrl || null,
      srtUrl: f.downloads?.srtUrl || null,
      originalUrl: null, // rebuilt when file is re-added
    },
  }));
}

export default function UploadPage() {
  const [hydrated, setHydrated] = React.useState(false);

  const [apiOk, setApiOk] = React.useState(true);
  const [models, setModels] = React.useState(["auto"]);
  const [defaultModel, setDefaultModel] = React.useState("auto");
  const [language, setLanguage] = React.useState("auto");
  const [summarize, setSummarize] = React.useState(true);

  // diarization controls
  const [diarize, setDiarize] = React.useState(true);
  const [numSpeakers, setNumSpeakers] = React.useState(3);
  const [diarizer, setDiarizer] = React.useState("auto");

  const [files, setFiles] = React.useState([]);
  const [activeId, setActiveId] = React.useState(null);
  const [log, setLog] = React.useState(["(idle)"]);

  const [sessionId, setSessionId] = React.useState(getSession());

  // Restore on mount
  React.useEffect(() => {
    (async () => {
      try {
        const ms = await listModels();
        setModels(ms?.length ? ms : ["auto"]);
      } catch {
        setApiOk(false);
      }

      try {
        const raw = localStorage.getItem(STATE_KEY);
        if (raw) {
          const saved = JSON.parse(raw);
          if (saved.sessionId) setSessionId(saved.sessionId);
          if (saved.language) setLanguage(saved.language);
          if (saved.defaultModel) setDefaultModel(saved.defaultModel);
          if (typeof saved.summarize === "boolean") setSummarize(saved.summarize);
          if (typeof saved.diarize === "boolean") setDiarize(saved.diarize);
          if (saved.numSpeakers != null) setNumSpeakers(saved.numSpeakers);
          if (saved.diarizer) setDiarizer(saved.diarizer);

          const revived = reviveFiles(saved.files);
          if (revived.length) {
            setFiles(revived);
            setActiveId(saved.activeId || revived[0]?.id || null);
          }
          if (saved.log?.length) setLog(saved.log);
        }
      } catch {}
      setHydrated(true);
    })();
  }, []);

  // Persist after hydration only
  React.useEffect(() => {
    if (!hydrated) return;
    const state = {
      sessionId,
      language,
      defaultModel,
      summarize,
      diarize,
      numSpeakers,
      diarizer,
      files: serializeFiles(files),
      activeId,
      log,
    };
    localStorage.setItem(STATE_KEY, JSON.stringify(state));
  }, [hydrated, sessionId, language, defaultModel, summarize, diarize, numSpeakers, diarizer, files, activeId, log]);

  function addFiles(list) {
    const arr = Array.from(list || []).map((f) => {
      const id = `${Date.now()}-${f.name}-${Math.random().toString(36).slice(2, 6)}`;
      return {
        id,
        file: f,
        name: f.name,
        chosenModel: null,
        status: "idle",
        res: null,
        err: "",
        downloads: {
          originalUrl: makeObjectUrl(f),
          txtUrl: null,
          srtUrl: null,
        },
      };
    });
    setFiles((prev) => [...prev, ...arr]);
    if (!activeId && arr[0]) setActiveId(arr[0].id);
  }

  function removeFile(id) {
    setFiles((prev) => {
      const it = prev.find((x) => x.id === id);
      if (it?.downloads?.originalUrl) revokeUrl(it.downloads.originalUrl);
      if (it?.downloads?.txtUrl) revokeUrl(it.downloads.txtUrl);
      if (it?.downloads?.srtUrl) revokeUrl(it.downloads.srtUrl);
      return prev.filter((x) => x.id !== id);
    });
    if (activeId === id) {
      const next = files.find((f) => f.id !== id);
      setActiveId(next ? next.id : null);
    }
  }

  // Create/update TXT link whenever transcript changes
  React.useEffect(() => {
    const it = files.find((f) => f.id === activeId);
    if (!it || !it.res?.text) return;
    // build a fresh txt URL
    const txtBlob = new Blob([it.res.text], { type: "text/plain;charset=utf-8" });
    const url = makeObjectUrl(txtBlob);
    setFiles((prev) =>
      prev.map((x) => {
        if (x.id !== it.id) return x;
        if (x.downloads?.txtUrl) revokeUrl(x.downloads.txtUrl);
        return { ...x, downloads: { ...(x.downloads || {}), txtUrl: url } };
      })
    );
    // revoke on unmount or when content changes again
    return () => revokeUrl(url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, JSON.stringify(files.find((f) => f.id === activeId)?.res?.text || "")]);

  async function runOne(item) {
    const chosen = item.chosenModel || defaultModel || "auto";
    setFiles((prev) =>
      prev.map((x) => (x.id === item.id ? { ...x, status: "running", err: "" } : x))
    );
    setLog((l) => [`▶ Transcribing: ${item.name || item.file?.name}`, ...l]);
    try {
      const res = await transcribe(item.file, chosen, {
        language,
        summarize,
        diarize,
        num_speakers: diarize ? numSpeakers : undefined,
        diarizer: diarize ? diarizer : undefined,
      });
      setFiles((prev) =>
        prev.map((x) =>
          x.id === item.id
            ? {
                ...x,
                status: "done",
                res,
                name: x.name || x.file?.name || res.filename || "audio",
              }
            : x
        )
      );
      setActiveId(item.id);
      setLog((l) => [`✓ Done: ${item.name || item.file?.name} (model: ${res.model || chosen})`, ...l]);
    } catch (e) {
      setFiles((prev) =>
        prev.map((x) => (x.id === item.id ? { ...x, status: "error", err: String(e) } : x))
      );
      setLog((l) => [`✗ Failed: ${item.name || item.file?.name} → ${String(e)}`, ...l]);
    }
  }

  async function runAll() {
    for (const it of files) {
      if (it.status === "idle" || it.status === "error") {
        // eslint-disable-next-line no-await-in-loop
        await runOne(it);
      }
    }
  }

  function clearQueue() {
    // revoke object URLs
    files.forEach((f) => {
      revokeUrl(f.downloads?.originalUrl);
      revokeUrl(f.downloads?.txtUrl);
      revokeUrl(f.downloads?.srtUrl);
    });
    setFiles([]);
    setActiveId(null);
    setLog(["(idle)"]);
  }

  function startNewSession() {
    const id = newSession();
    setSessionId(id);
    clearQueue();
    localStorage.removeItem(STATE_KEY);
  }

  const activeItem = files.find((f) => f.id === activeId) || files[0] || null;
  const currentFileModel = activeItem?.chosenModel ?? defaultModel ?? "auto";
  const onChangeActiveFileModel = (val) => {
    if (!activeItem) return;
    setFiles((prev) =>
      prev.map((x) => (x.id === activeItem.id ? { ...x, chosenModel: val || null } : x))
    );
  };

  // ---- Download actions shown in OUTPUT panel ----
  async function onDownloadSrt(item) {
    try {
      if (!item?.file) {
        alert("Re-add the file to this session to export SRT.");
        return;
      }
      const chosen = item.chosenModel || defaultModel || "auto";
      const blob = await transcribeSrt(item.file, chosen, {
        language,
        diarize,
        num_speakers: diarize ? numSpeakers : undefined,
        diarizer: diarize ? diarizer : undefined,
      });
      const url = makeObjectUrl(blob);
      setFiles((prev) =>
        prev.map((x) => {
          if (x.id !== item.id) return x;
          if (x.downloads?.srtUrl) revokeUrl(x.downloads.srtUrl);
          return { ...x, downloads: { ...(x.downloads || {}), srtUrl: url } };
        })
      );
    } catch (e) {
      alert("SRT export failed: " + String(e));
    }
  }

  function DownloadLinks({ item }) {
    const base = baseName(item?.name || item?.file?.name || "transcript");
    const txtReady = Boolean(item?.downloads?.txtUrl);
    const srtReady = Boolean(item?.downloads?.srtUrl);
    const canOriginal = Boolean(item?.downloads?.originalUrl);

    return (
      <div className="box" style={{ display: "grid", gap: 8 }}>
        <div style={{ fontWeight: 600 }}>Downloads</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {/* Transcript TXT */}
          <a
            className={`btn ${!txtReady ? "disabled" : ""}`}
            href={txtReady ? item.downloads.txtUrl : undefined}
            download={`${base}.txt`}
            onClick={(e) => { if (!txtReady) e.preventDefault(); }}
            title={txtReady ? "Download transcript (.txt)" : "Transcript not available yet"}
          >
            Transcript (.txt)
          </a>

          {/* SRT (generate or download if ready) */}
          {srtReady ? (
            <a
              className="btn"
              href={item.downloads.srtUrl}
              download={`${base}.srt`}
              title="Download SRT subtitles"
            >
              Subtitles (.srt)
            </a>
          ) : (
            <button
              className="btn"
              onClick={() => onDownloadSrt(item)}
              disabled={!item?.file}
              title={!item?.file ? "Re-add the file to export SRT" : "Generate SRT"}
            >
              Generate SRT
            </button>
          )}

          {/* Original file */}
          <a
            className={`btn ${!canOriginal ? "disabled" : ""}`}
            href={canOriginal ? item.downloads.originalUrl : undefined}
            download={item?.file?.name || `${base}.wav`}
            onClick={(e) => { if (!canOriginal) e.preventDefault(); }}
            title={canOriginal ? "Download original audio/video" : "Not available for history items"}
          >
            Original file
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="app-wrap">
      <header className="app-header">
        <h2>
          Transcription <span style={{ color: "#3B82F6" }}>Services</span>
        </h2>
        <span className={`badge ${apiOk ? "ok" : "bad"}`}>API: {apiOk ? "ready" : "offline"}</span>
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => document.getElementById("file-input").click()}>
          + Add files
        </button>
        <input
          id="file-input"
          type="file"
          accept="audio/*,video/*"
          multiple
          hidden
          onChange={(e) => addFiles(e.target.files)}
        />
        <button className="btn" onClick={runAll} disabled={!files.length}>
          Transcribe all
        </button>
        <button className="btn" onClick={clearQueue}>Clear</button>
        <button className="btn" onClick={startNewSession}>New session</button>
        <button className="btn" onClick={() => (window.location.hash = "#/history")}>History</button>
      </header>

      <div className="toolbar">
        <label>Language:&nbsp;
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="auto">Auto-detect</option>
            <option value="en">English</option>
          </select>
        </label>
        <label style={{ marginLeft: 12 }}>Default model:&nbsp;
          <select value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)}>
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label style={{ marginLeft: 12 }}>
          <input type="checkbox" checked={summarize} onChange={(e) => setSummarize(e.target.checked)} />
          &nbsp;Generate summary
        </label>

        <label style={{ marginLeft: 24 }}>
          <input type="checkbox" checked={diarize} onChange={(e) => setDiarize(e.target.checked)} />
          &nbsp;Speaker diarization
        </label>
        <label style={{ marginLeft: 12 }}>
          Speakers:&nbsp;
          <input
            type="number"
            min={0}
            max={10}
            value={numSpeakers}
            onChange={(e) => setNumSpeakers(parseInt(e.target.value || "0", 10))}
            className="box"
            style={{ width: 80 }}
            disabled={!diarize}
          />
        </label>
        <label style={{ marginLeft: 12 }}>
          Diarizer:&nbsp;
          <select value={diarizer} onChange={(e) => setDiarizer(e.target.value)} disabled={!diarize}>
            <option value="auto">auto</option>
            <option value="basic">basic</option>
            <option value="fallback">fallback</option>
          </select>
        </label>
      </div>

      <main className="grid">
        <section className="panel">
          <h3>Selected files</h3>
          {!files.length && <div style={{ color: "#666" }}>(none)</div>}
          {files.map((it) => {
            const label = it.file?.name || it.name || "(unsaved)";
            const tag = " • model: " + (it.chosenModel || defaultModel || "auto");
            return (
              <div
                key={it.id}
                className={`file-row ${activeId === it.id ? "active" : ""}`}
                onClick={() => setActiveId(it.id)}
              >
                <div className="file-name">
                  {label}<span style={{ color: "#888" }}>{tag}</span>
                </div>
                <div className="file-actions">
                  <button
                    className="btn"
                    onClick={(e) => { e.stopPropagation(); runOne(it); }}
                    disabled={it.status === "running" || !it.file}
                    title={!it.file && it.res ? "Re-add file to re-run" : ""}
                  >
                    Transcribe
                  </button>
                  <button className="btn" onClick={(e) => { e.stopPropagation(); removeFile(it.id); }}>
                    Remove
                  </button>
                </div>
              </div>
            );
          })}

          <h4 style={{ marginTop: 16 }}>Activity log</h4>
          <pre className="box" style={{ minHeight: 140 }}>{log.slice(0, 200).join("\n")}</pre>
        </section>

        <section className="panel">
          {activeItem ? (
            <>
              <h3>{activeItem.file?.name || activeItem.name}</h3>
              <div className="meta">
                Language: {activeItem?.res?.language || language}
                {" • "}
                Duration: {activeItem?.res?.duration_sec ? Math.round(activeItem.res.duration_sec) + "s" : "—"}
              </div>
              <div style={{ marginBottom: 10 }}>
                <label>
                  Model for this file:&nbsp;
                  <select value={currentFileModel} onChange={(e) => onChangeActiveFileModel(e.target.value)}>
                    <option value="">(use default: {defaultModel})</option>
                    {models.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </label>
              </div>

              <h4>Transcript</h4>
              <div className="box" style={{ minHeight: 160 }}>
                {activeItem?.res?.text
                  ? renderTaggedTranscript(activeItem.res.text)
                  : (activeItem.file ? "(not transcribed yet)" : "(not transcribed in this session — re-add file)")}
              </div>

              {/* NEW: Downloads shown in the output panel */}
              <div style={{ marginTop: 10 }}>
                <DownloadLinks item={activeItem} />
              </div>

              <h4>Summary</h4>
              <div className="box" style={{ minHeight: 80, whiteSpace: "pre-wrap" }}>
                {activeItem?.res?.summary || "(no summary)"}
              </div>
              <div className="meta" style={{ marginTop: 8 }}>
                Used model: <b>{activeItem?.res?.model || activeItem.chosenModel || defaultModel}</b>
              </div>
              {activeItem?.err && (
                <div className="panel error" style={{ marginTop: 12 }}>
                  <b>Error:</b> {activeItem.err}
                </div>
              )}
            </>
          ) : (
            <div style={{ color: "#666" }}>(select a file)</div>
          )}
        </section>
      </main>
    </div>
  );
}
