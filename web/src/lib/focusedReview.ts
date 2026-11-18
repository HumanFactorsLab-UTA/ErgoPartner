import { get, post } from "./api";
import type { MethodKey } from "./types";

export const checks = ["grip", "armSupport", "visibility"] as const;
export type Check = (typeof checks)[number];
export const checkLabels: Record<Check, string> = {
  grip: "Grip", armSupport: "Arm support", visibility: "Visibility",
};
export type Answers = Record<Check, string>;
export interface ReviewMoment {
  frame: number;
  person: string;
  camera: number;
  method: MethodKey;
}
export interface ReviewImage extends ReviewMoment {
  imageIndex: number;
  imageUrl: string;
  sec: number;
  sourceFrame: number;
  view: string;
  region?: string | null;
}
export interface ReviewCase extends ReviewMoment {
  caseId: string;
  cameraMode: "single" | "multi";
  evidence: ReviewImage[];
  questions: { key: Check; prompt: string; options: { value: string; label: string }[] }[];
  blindEligible: boolean;
  prediction: null | {
    id: string; model: string; promptVersion: string;
    answers: Record<Check, { value: string; reason: string; evidence: ReviewImage[] }>;
  };
  reference: { answers: Answers } | null;
  decision: { answers: Answers } | null;
}
export interface Count {
  n: number;
  correct: number;
  accuracy: number | null;
}
interface CheckMetric extends Count {
  unclearCases: number;
  correctAbstentions: number;
  falseDefinites: number;
  unnecessaryAbstentions: number;
}
export type Metrics = Record<Check, CheckMetric>;
export interface Evaluation {
  sessionCount?: number;
  skippedSessions?: number;
  caseCount: number;
  labelledCases: number;
  pairedCases: number;
  assistedCases: number;
  metrics: Metrics;
  model: string;
  promptVersion: string;
  byVisibility: Record<string, {caseCount: number; metrics: Metrics}>;
  byCameraMode: Record<string, {caseCount: number; metrics: Metrics}>;
}
const base = (id: string) => `/api/sessions/${encodeURIComponent(id)}/focused-review`;
export const focusedReview = {
  case: (id: string, moment: ReviewMoment, signal?: AbortSignal) => {
    const params = new URLSearchParams({
      frame: String(moment.frame), camera: String(moment.camera),
      method: moment.method, person: moment.person,
    });
    return get<ReviewCase>(`${base(id)}?${params}`, signal);
  },
  save: (id: string, action: "predict" | "decision" | "reference", review: ReviewCase, answers: Answers | undefined, signal: AbortSignal) => {
    const { caseId, frame, camera, method, person } = review;
    const predictionId = action === "decision" ? review.prediction?.id : undefined;
    return post<ReviewCase>(`${base(id)}/${action}`, {caseId, frame, camera, method, person, answers, predictionId}, signal);
  },
  samples: (id: string, person: string, signal?: AbortSignal) =>
    get<{ frames: {frame: number; sec: number}[] }>(`${base(id)}/samples?person=${encodeURIComponent(person)}`, signal),
  evaluation: (id?: string, signal?: AbortSignal) => get<Evaluation>(
    id ? `${base(id)}/evaluation` : "/api/focused-review/evaluation", signal),
};
