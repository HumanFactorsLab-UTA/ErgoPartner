import { ApiError, get, post } from "./api";
import type { AnalysisJob, MethodKey } from "./types";

export type CameraMode = "single" | "multi";
export const multiRoles = ["front", "side", "iso"] as const;
export type CameraRole = "camera" | typeof multiRoles[number];
export type Profile = "standing_light_work" | "standing_manual_handling" | "seated_workstation" | null;
export interface AssessmentSetup {
  visionModel?: string | null;
  cameraMode: CameraMode;
  studyName: string; studyFolder: string | null; participantId: string; taskName: string;
  heightCm: number | null; primaryMethod: MethodKey; samplingFps: number;
  loadKg: number | null; coupling: "good" | "fair" | "poor" | "unacceptable" | null;
  assumptionProfile: Profile; shockForce: boolean | null;
  leftArmSupport: boolean | null; rightArmSupport: boolean | null;
  leftWristEndRange: boolean | null; rightWristEndRange: boolean | null;
  rulaLoadType: "intermittent" | "static_or_repeated" | null;
  footageReviewed: boolean; synchronizationConfirmed: boolean; conditionsReviewed: boolean;
}
export interface AssessmentVideo {
  id: string; name: string; path: string; role: CameraRole; fps: number;
  frames: number; durationSec: number; width: number; height: number;
}
export interface AssessmentDraft {
  grouped?: boolean;
  existing?: TaskIdentity;
  id: string; source: "video" | "live" | "existing"; studiesRoot: string;
  createdAt: string; updatedAt: string; step: number; setup: AssessmentSetup;
  videos: AssessmentVideo[]; selectedTake: string | null;
  availableVideos?: AssessmentVideo[];
  takes: { id: string; createdAt: string; durationSec: number; videos: AssessmentVideo[];
    manifest?: { placement_review?: PlacementStatus | null } }[];
  published: TaskIdentity | null; analysis: { jobId: string; sessionId: string } | null;
  job: AnalysisJob | null; status: "draft" | "processing" | "failed" | "ready" | "interrupted";
}
export interface Preferences {
  visionModel?: string | null;
  primaryMethod: MethodKey; samplingFps: number; assumptionProfile: Profile; cameraFps: 24 | 30 | 60;
}
export interface TaskIdentity { participantKey: string; taskFolder: string }
export interface TaskGroup { studyFolder?: string; studyName?: string; participantId?: string; taskName: string }
export type TrashTarget =
  | {kind: "draft"; draftId: string}
  | ({kind: "task"} & TaskIdentity)
  | {kind: "participant"; participantKey: string};
export interface TrashItem {
  id: string; kind: TrashTarget["kind"]; label: string; deletedAt: string; participantKey?: string; taskCount: number;
}
export interface AssessmentOptions {
  studies: { folder: string; name: string }[];
  profiles: Record<string, Record<string, unknown>>;
  preferences: Preferences;
}
export interface CaptureStatus {
  assessmentId: string | null; preview_active: boolean; recording: boolean;
  elapsed_sec: number; synchronized_frame_sets: number; dropped_frame_sets: number;
  cameras: Record<string, { label: string; actual_fps: number; running: boolean; error: string | null }>;
}
export type PlacementChoice = "ready" | "adjust" | "unclear";
export interface PlacementCamera {
  id: string;
  state: "warming" | "attention" | "clear" | "unavailable";
  samples: number;
  regions: { name: string; status: string }[];
  issues: string[];
  decision: { value: PlacementChoice } | null;
  visual: null | {
    state: "checking" | "ready" | "unavailable" | "expired"; checkId: string; frames: number[];
    detail?: string; capturedAt: string; model?: string;
    findings?: { label: string; visibility: string; cue: string; suggestion: string; frames: number[] }[];
  };
}
export type PlacementStatus = { preview: string; cameras: Record<string, PlacementCamera>; noClearView?: string[] };
export interface AssessmentReview {
  reviewer: string; findings: string; actions: string; reviewed: boolean;
  reviewedAt: string | null; updatedAt: string | null; stale: boolean;
}
const base = (id: string) => `/api/assessments/${encodeURIComponent(id)}`;
export const workflow = {
  list: (signal?: AbortSignal) => get<{ assessments: AssessmentDraft[] }>("/api/assessments", signal),
  create: (source: AssessmentDraft["source"], participantKey?: string, taskFolder?: string, cameraMode?: CameraMode, group?: TaskGroup) =>
    post<AssessmentDraft>("/api/assessments", { source, participantKey, taskFolder, cameraMode, group }),
  trash: (signal?: AbortSignal) => get<{items: TrashItem[]}>("/api/assessment-trash", signal),
  moveToTrash: (target: TrashTarget) => post<unknown>("/api/assessment-trash", target),
  restore: (id: string) => post<{restored: boolean}>(`/api/assessment-trash/${encodeURIComponent(id)}/restore`, {}),
  draft: (id: string, signal?: AbortSignal) => get<AssessmentDraft>(base(id), signal),
  save: (draft: AssessmentDraft, step = draft.step) => post<AssessmentDraft>(base(draft.id), { setup: draft.setup, step, roles: draft.videos.map(v => v.role) }),
  selectFootage: (draft: AssessmentDraft, cameraMode: CameraMode, videoIds?: string[]) =>
    post<AssessmentDraft>(base(draft.id), {
      setup: { ...draft.setup, cameraMode, footageReviewed: false, synchronizationConfirmed: false },
      step: draft.step, videoIds,
    }),
  options: (signal?: AbortSignal) => get<AssessmentOptions>("/api/assessment-options", signal),
  preferences: (value: Preferences) => post<Preferences>("/api/assessment-preferences", value),
  upload: async (id: string, file: File, signal?: AbortSignal) => {
    const response = await fetch(`${base(id)}/videos?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file, signal, headers: { "Content-Type": "application/octet-stream" } });
    if (!response.ok) { const body = await response.json(); throw new ApiError(body.detail ?? "The video could not be added.", response.status); }
    return response.json() as Promise<AssessmentDraft>;
  },
  remove: (id: string, video: string) => post<AssessmentDraft>(`${base(id)}/videos/${video}/remove`, {}),
  videoStatus: (id: string, video: string, signal?: AbortSignal) => get<{ ready: boolean; state: string; detail: string }>(`${base(id)}/videos/${video}/status`, signal),
  videoUrl: (id: string, video: string) => `${base(id)}/videos/${video}/play`,
  analyze: (id: string) => post<AssessmentDraft>(`${base(id)}/analyze`, {}),
  capture: (signal?: AbortSignal) => get<CaptureStatus>("/api/capture/status", signal),
  devices: () => post<{ devices: number[] }>("/api/capture/devices", {}),
  preview: (id: string, cameras: { deviceId: number; role: CameraRole }[]) => post<CaptureStatus>(`${base(id)}/capture/preview`, { cameras }),
  startCapture: (id: string) => post<CaptureStatus>(`${base(id)}/capture/start`, {}),
  stopCapture: (id: string) => post<AssessmentDraft>(`${base(id)}/capture/stop`, {}),
  closeCapture: (id: string) => post<CaptureStatus>(`${base(id)}/capture/close`, {}),
  captureImage: (id: string, device: string, tick: number) => `${base(id)}/capture/${device}.jpg?t=${tick}`,
  placement: (id: string, signal?: AbortSignal) => get<PlacementStatus>(`${base(id)}/capture/placement`, signal),
  inspectPlacement: (id: string, device: string, reviewId: string) => post<PlacementStatus>(`${base(id)}/capture/placement/${device}/inspect`, { reviewId }),
  decidePlacement: (id: string, device: string, reviewId: string, choice: PlacementChoice) => post<PlacementStatus>(`${base(id)}/capture/placement/${device}/decision`, { reviewId, choice }),
  placementImage: (id: string, device: string, check: string, frame: number) => `${base(id)}/capture/placement/${device}/checks/${check}/${frame}.jpg`,
  selectTake: (id: string, take: string) => post<AssessmentDraft>(`${base(id)}/takes/${take}`, {}),
  review: (id: string, signal?: AbortSignal) => get<AssessmentReview>(`/api/sessions/${encodeURIComponent(id)}/review`, signal),
  saveReview: (id: string, value: AssessmentReview) => post<AssessmentReview>(`/api/sessions/${encodeURIComponent(id)}/review`, value),
};

export const profileLabels: Record<string, string> = {
  standing_light_work: "Standing light work", standing_manual_handling: "Standing manual handling", seated_workstation: "Seated workstation",
};
export const roleLabels: Record<CameraRole, string> = { camera: "Camera", front: "Front", side: "Side", iso: "Oblique (ISO)" };
export function studyDisplayName(name: string | null | undefined, folder = "") {
  const value = name || folder;
  const stamp = /^study_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}$/i.exec(value);
  if (stamp) {
    const [,year,month,day,hour,minute] = stamp;
    const date = new Date(Number(year), Number(month)-1, Number(day), Number(hour), Number(minute));
    return `Study on ${date.toLocaleString(undefined, {dateStyle:"medium",timeStyle:"short"})}`;
  }
  return name || folder.replace(/_/g, " ").trim();
}

export function reviewLink(sessionId: string, fallback = "reba") {
  let method = fallback === "rula" ? "rula" : "reba";
  try { const saved = localStorage.getItem(`posture-method:${sessionId}`); if (saved === "reba" || saved === "rula") method = saved; } catch { /* Storage may be disabled. */ }
  return `/s/${sessionId}/${method}`;
}
