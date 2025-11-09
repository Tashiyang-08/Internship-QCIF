// src/lib/transcribe.ts
import { listModels, ping, transcribe, TranscriptionResponse } from "./api";

/**
 * Pings the backend and fetches available models.
 * Returns ok=false if either step fails.
 */
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

/**
 * Runs transcription with a safe model choice.
 * - Uses explicit model if valid; otherwise "auto" when available.
 * - Diarization:
 *    - diarize=true and num_speakers === undefined => AUTO speaker count on backend
 *    - diarize=true and num_speakers is number     => fixed count
 */
export async function runTranscription(
  file: File,
  model?: string,
  opts?: {
    language?: string;
    summarize?: boolean;
    diarize?: boolean;
    num_speakers?: number | null; // leave null/undefined for AUTO
    diarizer?: "auto" | "basic" | "fallback";
  }
): Promise<TranscriptionResponse> {
  if (!file) {
    throw new Error("No file provided");
  }

  let models: string[] = [];
  try {
    models = await listModels();
  } catch (e) {
    console.warn("Failed to fetch models; using fallback.", e);
  }

  const uniq = Array.from(new Set(models));
  const hasAuto = uniq.includes("auto");
  const preferredFallback = hasAuto ? "auto" : (uniq[0] || "faster-whisper-small");

  const selectedModel =
    model && uniq.length > 0 && uniq.includes(model) ? model : preferredFallback;

  const safeOpts = {
    language: opts?.language,
    summarize: Boolean(opts?.summarize),
    diarize: Boolean(opts?.diarize),
    diarizer: opts?.diarizer || "auto",
    num_speakers:
      typeof opts?.num_speakers === "number" && opts?.num_speakers > 0
        ? opts?.num_speakers
        : undefined, // AUTO when undefined
  };

  return transcribe(file, selectedModel, safeOpts);
}
