// src/lib/transcribe.ts
import { listModels, ping, transcribe, TranscriptionResponse } from "./api";

/** Ping backend + fetch models (returns ok=false if either fails) */
export async function ensureApiReady(): Promise<{ ok: boolean; models: string[] }> {
  try {
    await ping();
    const models = await listModels();
    const uniq = Array.from(new Set(models));
    const withoutAuto = uniq.filter((m) => m && m !== "auto");
    const ordered = uniq.includes("auto") ? ["auto", ...withoutAuto] : withoutAuto;
    return { ok: true, models: ordered };
  } catch (err) {
    console.error("API not ready:", err);
    return { ok: false, models: [] };
  }
}

/** Submit transcription with safe defaults (model/diarization/language) */
export async function runTranscription(
  file: File,
  model?: string,
  opts?: {
    language?: string;
    summarize?: boolean;
    diarize?: boolean;
    num_speakers?: number | null; // 0/null/undefined -> AUTO
    diarizer?: "auto" | "basic" | "fallback";
  }
): Promise<TranscriptionResponse> {
  if (!file) throw new Error("No file provided");

  let models: string[] = [];
  try {
    models = await listModels();
  } catch (e) {
    console.warn("Failed to fetch model list; using fallback.", e);
  }

  const uniq = Array.from(new Set(models));
  const hasAuto = uniq.includes("auto");
  const fallback = hasAuto ? "auto" : uniq[0] || "faster-whisper-small";

  const selectedModel =
    model && uniq.length > 0 && uniq.includes(model)
      ? model
      : fallback;

  /** ✅ Speaker count AUTO logic */
  const numSpeakers =
    typeof opts?.num_speakers === "number" && opts.num_speakers > 0
      ? opts.num_speakers
      : undefined; // backend auto-mode

  const safeOpts = {
    language: opts?.language,
    summarize: Boolean(opts?.summarize),
    diarize: Boolean(opts?.diarize),
    diarizer: opts?.diarizer || "auto",
    num_speakers: numSpeakers, // ✅ only send number if >0
  };

  return transcribe(file, selectedModel, safeOpts);
}
