export type Method = "REBA" | "RULA";
export type MethodKey = "reba" | "rula";
export type BandStatus = "good" | "warning" | "serious" | "critical";

export interface Band {
  label: string;
  action: string;
  status: BandStatus;
  lower: number;
  upper: number;
  range: string;
  describesRisk?: boolean;
}

export interface TrialCard {
  updatedAt?: string;
  taskId: number | null;
  name: string;
  folder: string;
  description: string | null;
  primaryTool: string | null;
  cameras: number | null;
  samplings: (number | null)[];
  status: "analysed" | "not_analysed" | "incomplete";
  reviewStatus?: "reviewed" | "needs_review";
  sessionId: string | null;
  durationSec: number | null;
  recordedAt: string | null;
  reba: MethodHeadline | null;
  rula: MethodHeadline | null;
}

export interface MethodHeadline {
  maximum: number | null;
  band: Band | null;
  observations?: number | null;
  percentile90?: number | null;
  durationWeightedMean?: number | null;
  peakFrame?: number | null;
  peakSec?: number | null;
}

export interface ParticipantCard {
  key: string;
  participantId: string;
  label: string;
  recordedAt: string | null;
  study: string;
  studyName?: string | null;
  age: number | null;
  sex: string | null;
  heightCm: number | null;
  weightKg: number | null;
  dominantHand: string | null;
  trialCount: number;
  analysedCount: number;
  trials: TrialCard[];
}

export interface SessionOverview {
  sessionId: string;
  durationSec: number | null;
  processedFrames: number | null;
  poseFrames: number | null;
  cameraCount: number | null;
  trackingMode: string | null;
  calibration: string | null;
  softwareVersion: string | null;
  runtimeSec: number | null;
  methods: Record<MethodKey, MethodHeadline>;
  rnle: {
    liftingIndex: number | null;
    recommendedWeightKg: number | null;
    validity: string | null;
    missing: string[];
    invalid: string[];
  };
  warnings: string[];
  people: string[];
  sampling: number | null;
  samplingLabel: string;
  participant: {
    key: string;
    participantId: string;
    label: string;
    recordedAt: string | null;
    age: number | null;
    sex: string | null;
    dominantHand: string | null;
    heightCm: number | null;
  };
  trial: {
    taskId: number | null;
    name: string;
    folder: string;
    description: string | null;
    primaryTool: string | null;
  };
  cameras: { camera_index: number; path: string; name: string }[];
}

export interface SeriesPoint {
  /** frame number */
  f: number;
  /** timestamp in seconds */
  t: number;
  /** score */
  s: number;
  /** band status */
  b: BandStatus | "none";
}

export interface Driver {
  region: string;
  label: string;
  component: string;
  kind: "posture" | "task";
  score: number;
  maximum: number;
  severity: number;
  pointsAdded: number;
  measurement: string | null;
  valueDeg: number | null;
}

export interface Episode {
  index: number;
  startSec: number;
  endSec: number;
  durationSec: number;
  startFrame: number;
  endFrame: number;
  peakScore: number;
  peakFrame: number;
  peakSec: number;
  observations: number;
  band: Band | null;
  timeRange: string;
  drivers: Driver[];
}

export interface BodyLoad {
  region: string;
  label: string;
  component: string;
  kind: "posture" | "task";
  maximum: number;
  worstScore: number;
  worstPoints: number;
  meanPoints: number;
  percentOfTask: number;
  percentRaised: number;
  peakFrame: number | null;
  peakSec: number | null;
  measurement: string | null;
  peakValueDeg: number | null;
}

export interface BandShare {
  label: string;
  range: string;
  status: BandStatus;
  action: string;
  durationSec: number;
  percent: number;
  observations: number;
}

export interface Timeline {
  sessionId: string;
  method: Method;
  person: string;
  scaleMaximum: number;
  bands: Band[];
  actionThreshold: number;
  verdict: {
    scored: boolean;
    headline: string;
    detail: string;
    peakScore: number | null;
    peakFrame: number | null;
    peakSec: number | null;
    peakClock: string;
    peakBand: Band | null;
    typicalScore: number | null;
    typicalBand: Band | null;
    scoredDurationSec: number;
    timeAtActionSec: number;
    percentAtAction: number;
    episodeCount: number;
    leadingRegion: string | null;
    leadingLabel: string | null;
    leadingPoints: number | null;
    leadingTaskLabel: string | null;
    leadingTaskPoints: number | null;
  };
  series: { count: number; drawn: number; points: SeriesPoint[] };
  shares: BandShare[];
  episodes: Episode[];
  posture: BodyLoad[];
  taskFactors: BodyLoad[];
}

export interface FrameCamera {
  cameraIndex: number;
  role: string;
  sourceFrame: number | null;
  timestampSec: number | null;
  syncDeltaSec: number | null;
}

export interface FrameDetail {
  sessionId: string;
  method: Method;
  person: string;
  frame: number;
  timestampSec: number | null;
  clock: string;
  score: number | null;
  band: Band | null;
  selectedSide: string | null;
  selectionReason: string | null;
  riskLevel: string | null;
  actionLevel: string | null;
  components: Record<string, number>;
  sides: Record<
    "left" | "right",
    {
      score: number | null;
      validity: string | null;
      components: Record<string, number>;
      missing: string[];
      invalid: string[];
    }
  >;
  drivers: Driver[];
  angles: Record<string, number>;
  angleSources: Record<string, string>;
  anglesByCamera?: Record<string, Record<string, number>>;
  quality: {
    meanVisibility: number | null;
    cameraView: string | null;
    wristSource: string | null;
    status: string | null;
    warnings: string[];
  };
  cameras: FrameCamera[];
  offset: number;
}

export type AnalysisScope = "trial" | "participant" | "study" | "all";
export type AnalysisJobStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "completed_with_errors"
  | "failed";

export interface AnalysisJobOutcome {
  label: string;
  ok: boolean;
  error: string | null;
  outputDir?: string | null;
  participantKey?: string;
  taskFolder?: string;
  sessionId?: string | null;
}

export interface AnalysisJob {
  jobId: string;
  scope: AnalysisScope;
  status: AnalysisJobStatus;
  done: boolean;
  total: number;
  completed: number;
  failed: number;
  skippedCount?: number;
  progressPercent: number;
  currentLabel: string | null;
  current?: {
    status: string;
    stage: string;
    message: string;
    progressFrame: number;
    totalFrames: number;
    elapsedSec: number;
    etaSec: number;
  } | null;
  outcomes: AnalysisJobOutcome[];
  sessionIds?: string[];
  message?: string | null;
}

export interface ReplayPoint {
  id: number;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface ReplayFrame {
  f: number;
  t: number;
  sourceFrame: number | null;
  score: number | null;
  band: Band | null;
  points: ReplayPoint[];
}

export interface ReplayPayload {
  sessionId: string;
  method: Method;
  person: string | null;
  cameraIndex: number;
  sourceCount: number;
  sampledCount: number;
  anchorFrame: number | null;
  anchorIncluded: boolean;
  coordinateSpace: string;
  imageWidth: number;
  imageHeight: number;
  aspectRatio: number;
  dimensionsSource: string;
  visibilityThreshold: number;
  connections: [number, number][];
  frames: ReplayFrame[];
}

export interface ObservationRow {
  frame: number;
  timestampSec: number;
  clock: string;
  score: number | null;
  band: Band | null;
  selectedSide: string | null;
  validity: string | null;
  meanVisibility: number | null;
  components: Record<string, number>;
  angles: Record<string, number>;
}

export interface ObservationsPayload {
  sessionId: string;
  method: Method;
  total: number;
  page: number;
  limit: number;
  pages: number;
  rows: ObservationRow[];
}

export interface RiskPeriod {
  index: number;
  startFrame: number;
  endFrame: number;
  startSec: number;
  endSec: number;
  durationSec: number;
  observations: number;
  peakScore: number | null;
  peakFrame: number;
  band: Band | null;
}

export interface RiskPeriodsPayload {
  sessionId: string;
  method: Method;
  durationSec: number;
  periods: RiskPeriod[];
}

/* ------------------------------------------------------------------ */
/* Assistant                                                           */
/* ------------------------------------------------------------------ */

export interface AssistantHealth {
  selection?: string | null;
  ok: boolean;
  state: "ready" | "unreachable" | "model_missing" | "disabled" | "unavailable";
  detail: string;
  endpoint?: string;
  model?: string;
  available?: string[];
  chatModel?: string;
  sceneModel?: string;
  promptVersion?: string;
  /** True when the model endpoint is on this machine, so no footage leaves it. */
  local?: boolean;
}

/** A moment the assistant referenced, rendered as a control that seeks the workspace. */
export interface AssistantCitation {
  frame: number | null;
  sec: number | null;
  method: MethodKey | null;
  region: string | null;
  camera?: number | null;
  person?: string | null;
  kind?: "observed" | "computed";
  view?: string | null;
  overlay?: boolean;
}

export interface ReviewSelection {
  method: MethodKey;
  frame: number | null;
  person: string | null;
  region: string | null;
  camera: number;
  timestampSec: number | null;
}

export interface SceneEvidence {
  frame: number;
  sec: number;
  camera: number;
  person?: string | null;
  view?: string;
}

export interface AssistantToolCall {
  name: string;
  arguments: Record<string, unknown>;
  status?: "running" | "done";
  frames?: number[];
  images?: number;
}

export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
  citations?: AssistantCitation[];
  toolCalls?: AssistantToolCall[];
  audit?: AuditReport;
  /** Set while a reply is still streaming, so the composer can lock. */
  streaming?: boolean;
  error?: string | null;
}

/**
 * How many independent readings of the recording produced the same answer.
 *
 * This replaces the model's own stated confidence as the thing to trust: it is
 * measured across passes over different frames rather than asserted by the same
 * forward pass that produced the reading.
 */
export interface Agreement {
  passes: number;
  agreeing: number;
  fraction: number;
  band: "unanimous" | "majority" | "split" | "contested";
  alternatives: unknown[];
}

export interface SceneProposal {
  proposalId?: string;
  evidence?: SceneEvidence[];
  condition: string;
  value?: unknown;
  value_range?: [number, number];
  confidence: "high" | "medium" | "low";
  rationale: string;
  evidence_frames: number[];
  agreement?: Agreement;
  contested?: boolean;
  status?: "pending" | "accepted" | "rejected";
  decidedValue?: unknown;
  decidedAt?: string | null;
  note?: string | null;
}

export interface SceneCard {
  findings?: {fact: string; visibility: string; category: string; evidence: SceneEvidence[]; passes: number; readings: number}[];
  schema: string;
  sessionId: string;
  builtAt: string;
  model: string;
  promptVersion: string;
  passes?: number;
  frames_sampled: number[];
  frameSelection?: {
    method: string;
    coverageVsUniform: number;
    note: string;
  };
  perPass?: {
    pass: number;
    frames: number[];
    coverage: number;
    activity_class?: string | null;
  }[];
  task_inference: {
    summary?: string;
    activity_class?: string;
    confidence?: string;
    agreement?: Agreement;
    alternative_summaries?: string[];
  };
  workstation: { objects?: string[]; tools?: string[]; surfaces?: string[] };
  condition_proposals: SceneProposal[];
  segments: { start_sec: number; end_sec: number; label: string }[];
  observations: string[];
  disclaimer: string;
}

/** A number the answer stated, and whether the evidence the model saw held it. */
export interface AuditClaim {
  value: number;
  written: string;
  context: string;
  traced: boolean;
  matched: number | null;
}

export interface AuditReport {
  total: number;
  traced: number;
  untraced: AuditClaim[];
  note: string;
}

export interface AssistantContext {
  sessionId: string;
  suggestions: string[];
  history: {
    id: string;
    at: string;
    question: string;
    answer: string;
    citations: AssistantCitation[];
    toolCalls?: AssistantToolCall[];
    audit?: AuditReport;
  }[];
  scene: SceneCard | null;
  sceneBuilt: boolean;
  proposals: ProposalsPayload;
}

export interface ReportStatus {
  exists: boolean;
  sessionId: string;
  filename?: string;
  sizeKb?: number;
}

export interface ProposalsPayload {
  available: boolean;
  sessionId?: string;
  proposals: SceneProposal[];
  decisions?: Record<string, unknown>;
  note?: string;
}

export interface ConditionSources {
  observed: string[];
  derived: string[];
  assumed: string[];
  unknown: string[];
  conflicts: unknown[];
  confirmFirst: string[];
  assumedValues: Record<string, unknown>;
  notObservableFromPose: string[];
  note: string;
}
export type StudiesFolder = { path: string; exists: boolean };
export type FolderListing = {
  path: string;
  parent: string | null;
  drives: string[];
  studyCount: number;
  folders: { name: string; path: string; isStudy: boolean }[];
};
