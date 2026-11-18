import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnalysisJob,
  AnalysisScope,
  AssistantContext,
  AssistantHealth,
  ConditionSources,
  FrameDetail,
  FolderListing,
  StudiesFolder,
  MethodKey,
  ObservationsPayload,
  ParticipantCard,
  ProposalsPayload,
  ReplayPayload,
  ReportStatus,
  RiskPeriodsPayload,
  SessionOverview,
  Timeline,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `Request failed with ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* the body was not JSON; the status message stands */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    signal,
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `Request failed with ${response.status}`;
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === "string") detail = payload.detail;
      else if (Array.isArray(payload?.detail)) detail = payload.detail.map((item: { msg?: string }) => item.msg ?? "Invalid input").join(" ");
    } catch {
      /* the status message stands */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  studiesFolder: (signal?: AbortSignal) => get<StudiesFolder>("/api/studies-folder", signal),
  browseFolders: (path: string, signal?: AbortSignal) =>
    get<FolderListing>(`/api/studies-folder/browse?path=${encodeURIComponent(path)}`, signal),
  selectStudiesFolder: (path: string) => post<StudiesFolder>("/api/studies-folder", { path }),
  participants: (search: string, signal?: AbortSignal) =>
    get<{ count: number; participants: ParticipantCard[] }>(
      `/api/participants?search=${encodeURIComponent(search)}`,
      signal,
    ),
  session: (sessionId: string, signal?: AbortSignal) =>
    get<SessionOverview>(`/api/sessions/${encodeURIComponent(sessionId)}`, signal),
  timeline: (sessionId: string, method: MethodKey, signal?: AbortSignal, person?: string | null) =>
    get<Timeline>(
      `/api/sessions/${encodeURIComponent(sessionId)}/timeline?method=${method}` +
        (person ? `&person=${encodeURIComponent(person)}` : ""),
      signal,
    ),
  frame: (
    sessionId: string,
    frame: number,
    method: MethodKey,
    person: string | null,
    signal?: AbortSignal,
  ) =>
    get<FrameDetail>(
      `/api/sessions/${encodeURIComponent(sessionId)}/frames/${frame}?method=${method}` +
        (person ? `&person=${encodeURIComponent(person)}` : ""),
      signal,
    ),
  frameImage: (
    sessionId: string,
    frame: number,
    camera: number,
    region: string,
    width = 720,
    overlay = true,
    person: string | null = null,
  ) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/frames/${frame}/cameras/${camera}.jpg` +
    `?region=${encodeURIComponent(region)}&width=${width}&overlay=${overlay}` +
    (person ? `&person=${encodeURIComponent(person)}` : ""),
  videoUrl: (sessionId: string, camera: number) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/video/${camera}`,
  videoStatus: (sessionId: string, camera: number, signal?: AbortSignal) =>
    get<{ state: string; detail: string; ready: boolean }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/video/${camera}/status`,
      signal,
    ),
  startAnalysis: (
    scope: AnalysisScope,
    options: {
      participantKey?: string;
      studyFolder?: string;
      taskFolder?: string;
      samplingFps?: number | null;
      rerun?: boolean;
    } = {},
    signal?: AbortSignal,
  ) =>
    post<AnalysisJob>(
      "/api/analysis/jobs",
      { scope, samplingFps: 10, rerun: false, ...options },
      signal,
    ),
  analysisJob: (jobId: string, signal?: AbortSignal) =>
    get<AnalysisJob>(`/api/analysis/jobs/${encodeURIComponent(jobId)}`, signal),
  activeAnalysisJobs: (signal?: AbortSignal) =>
    get<{ jobs: AnalysisJob[] }>("/api/analysis/jobs?active=true", signal),
  replay: (
    sessionId: string,
    method: MethodKey,
    camera: number,
    person: string | null,
    anchorFrame: number | null,
    signal?: AbortSignal,
  ) =>
    get<ReplayPayload>(
      `/api/sessions/${encodeURIComponent(sessionId)}/replay?method=${method}` +
        `&camera=${camera}&maxFrames=360` +
        (person ? `&person=${encodeURIComponent(person)}` : "") +
        (anchorFrame !== null ? `&anchorFrame=${anchorFrame}` : ""),
      signal,
    ),
  observations: (
    sessionId: string,
    method: MethodKey,
    page: number,
    band: string,
    person: string | null,
    signal?: AbortSignal,
  ) =>
    get<ObservationsPayload>(
      `/api/sessions/${encodeURIComponent(sessionId)}/observations?method=${method}` +
        `&page=${page}&limit=60&band=${encodeURIComponent(band)}` +
        (person ? `&person=${encodeURIComponent(person)}` : ""),
      signal,
    ),
  riskPeriods: (
    sessionId: string,
    method: MethodKey,
    person: string | null,
    signal?: AbortSignal,
  ) =>
    get<RiskPeriodsPayload>(
      `/api/sessions/${encodeURIComponent(sessionId)}/risk-periods?method=${method}` +
        (person ? `&person=${encodeURIComponent(person)}` : ""),
      signal,
    ),
  frameDataExport: (
    sessionId: string,
    method: MethodKey,
    band = "all",
    person: string | null = null,
  ) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/exports/frame-calculations.csv` +
    `?method=${method}&band=${encodeURIComponent(band)}` +
    (person ? `&person=${encodeURIComponent(person)}` : ""),
  assistantHealth: (signal?: AbortSignal, sessionId?: string) =>
    get<AssistantHealth>("/api/assistant/health" + (sessionId ? `?sessionId=${encodeURIComponent(sessionId)}` : ""), signal),
  assistantContext: (sessionId: string, signal?: AbortSignal, method: MethodKey = "reba") =>
    get<AssistantContext>(
      `/api/sessions/${encodeURIComponent(sessionId)}/assistant/context?method=${method}`,
      signal,
    ),
  conditions: (sessionId: string, signal?: AbortSignal) =>
    get<ConditionSources>(`/api/sessions/${encodeURIComponent(sessionId)}/conditions`, signal),
  proposals: (sessionId: string, signal?: AbortSignal) =>
    get<ProposalsPayload>(`/api/sessions/${encodeURIComponent(sessionId)}/proposals`, signal),
  decideProposal: (
    sessionId: string,
    decision: "accept" | "reject",
    proposal: {condition: string; proposalId?: string; value?: unknown; note?: string},
  ) =>
    post<ProposalsPayload>(
      `/api/sessions/${encodeURIComponent(sessionId)}/proposals/${decision}`,
      proposal,
    ),
  generateReport: (sessionId: string, signal?: AbortSignal) =>
    post<ReportStatus>(`/api/sessions/${encodeURIComponent(sessionId)}/report`, {}, signal),
  reportDownloadUrl: (sessionId: string) =>
    `/api/sessions/${encodeURIComponent(sessionId)}/report/download`,
};

/**
 * Read a server-sent event stream of JSON objects.
 *
 * The assistant answers over SSE because a turn that reads three tools and opens
 * two frames takes tens of seconds, and watching it work is the audit trail
 * rather than a spinner. `fetch` is used instead of `EventSource` because the
 * turn is a POST carrying the question and the conversation so far.
 */
export async function readEventStream(
  path: string,
  body: unknown,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    let detail = `Request failed with ${response.status}`;
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === "string") detail = payload.detail;
    } catch {
      /* the status message stands */
    }
    throw new ApiError(detail, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Events are separated by a blank line; a partial one stays in the buffer
    // until the rest of it arrives.
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const chunk = buffer.slice(0, split).trim();
      buffer = buffer.slice(split + 2);
      if (chunk.startsWith("data:")) {
        const payload = chunk.slice(5).trim();
        if (payload) {
          try {
            onEvent(JSON.parse(payload) as Record<string, unknown>);
          } catch {
            /* a malformed event must not abort the rest of the stream */
          }
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }
}

/**
 * Wait for a browser-playable copy of a recording.
 *
 * The captured files are MPEG-4 Part 2, which no browser decodes, so the server
 * transcodes an H.264 proxy on first request. Polling here means the panel can
 * say "converting" instead of showing a video element that silently errors.
 */
export function useVideoReady(sessionId: string, camera: number) {
  const [state, setState] = useState<{ ready: boolean; state: string; detail: string }>({
    ready: false,
    state: "checking",
    detail: "",
  });

  useEffect(() => {
    let live = true;
    let timer = 0;
    const controller = new AbortController();

    const poll = async () => {
      try {
        const result = await api.videoStatus(sessionId, camera, controller.signal);
        if (!live) return;
        setState(result);
        if (!result.ready && result.state === "building") {
          timer = window.setTimeout(poll, 2000);
        }
      } catch (error) {
        if (!live || controller.signal.aborted) return;
        setState({
          ready: false,
          state: "failed",
          detail: error instanceof Error ? error.message : "Playback is unavailable",
        });
      }
    };

    setState({ ready: false, state: "checking", detail: "" });
    void poll();
    return () => {
      live = false;
      controller.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [sessionId, camera]);

  return state;
}

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/**
 * Fetch on mount and whenever the key changes, cancelling the in-flight request.
 *
 * `keepPrevious` is what makes the workspace feel connected rather than reloaded:
 * while a new frame is loading the previous one stays on screen instead of
 * collapsing to a spinner, so switching frames reads as an update rather than a
 * navigation.
 */
export function useAsync<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  options: { keepPrevious?: boolean } = {},
): Async<T> & { reload: () => void } {
  const { keepPrevious = false } = options;
  const [state, setState] = useState<Async<T>>({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    setState((previous) => ({
      data: keepPrevious ? previous.data : null,
      error: null,
      loading: true,
    }));
    loaderRef
      .current(controller.signal)
      .then((data) => {
        if (live) setState({ data, error: null, loading: false });
      })
      .catch((error: unknown) => {
        if (!live || controller.signal.aborted) return;
        const message =
          error instanceof Error ? error.message : "Something went wrong loading this view";
        setState((previous) => ({
          data: keepPrevious ? previous.data : null,
          error: message,
          loading: false,
        }));
      });
    return () => {
      live = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { ...state, reload };
}

/** Delay a rapidly changing value so scrubbing does not fire a request per pixel. */
export function useDebounced<T>(value: T, delay = 140): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return settled;
}
