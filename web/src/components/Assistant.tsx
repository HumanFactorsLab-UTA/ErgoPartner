import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconAlert, IconArrowRight, IconInfo, IconPerson, IconSearch } from "./Icons";
import { Disclosure, Skeleton } from "./ui";
import { api, post, readEventStream, useAsync } from "../lib/api";
import { VisionModelSelect } from "./VisionModelSelect";
import { clock, humanise } from "../lib/format";
import type {
  Agreement,
  AssistantCitation,
  AssistantMessage,
  AssistantToolCall,
  AuditReport,
  MethodKey,
  ReviewSelection,
  SceneCard,
  SceneEvidence,
  SceneProposal,
} from "../lib/types";

/** What each tool is doing, in the reviewer's language rather than the model's. */
const TOOL_LABELS: Record<string, string> = {
  session_overview: "Reading the recording record",
  method_summary: "Reading the score summary",
  episodes: "Ranking the risk episodes",
  frame: "Reading one frame's calculation",
  body_regions: "Comparing body-region load",
  task_factors: "Reading the task conditions",
  risk_periods: "Listing action-band periods",
  observations: "Scanning frame calculations",
  condition_sources: "Checking measured against assumed",
  scene_card: "Reading the visual scene card",
  method_reference: "Looking up the method documentation",
  look_at: "Looking at the footage",
  watch: "Watching a stretch of the recording",
  cameras: "Checking the camera views",
};

/**
 * Inline register tags, e.g. `[computed · frame 825 · 0:28]`, `[observed · saw
 * frame 825]`, `[judgment]`.
 *
 * The register is the safety design, so it has to be visible rather than parsed
 * away. Rendering it as a chip is what lets a reviewer see at a glance which
 * sentences are measurements and which are the model's opinion.
 *
 * The optional backticks are not decoration: the model frequently emits the tag
 * as inline code, and without swallowing them the chip renders wrapped in stray
 * punctuation.
 */
const TAG = /`?\[(computed|observed|judgment)([^\]]*)\]`?/gi;

function TaggedText({ text, citations, onCitation }: { text: string; citations: AssistantCitation[]; onCitation: (citation: AssistantCitation) => void }) {
  const parts = useMemo(() => {
    const out: { kind: "text" | "tag"; value: string; register?: string; citation?: AssistantCitation }[] = [];
    let last = 0;
    TAG.lastIndex = 0;
    for (let match = TAG.exec(text); match !== null; match = TAG.exec(text)) {
      if (match.index > last) out.push({ kind: "text", value: text.slice(last, match.index) });
      const frame = /\bframe\s*[#:=]?\s*(\d+)\b/i.exec(match[2] ?? "");
      const camera = /\bcamera\s*[#:=]?\s*(\d+)\b/i.exec(match[2] ?? "");
      const method = /\b(rula|reba)\b/i.exec(match[2] ?? "");
      const register = match[1].toLowerCase();
      const matches = frame ? citations.filter(item =>
        item.frame === Number(frame[1]) && item.kind === register
        && (!camera || item.camera === Number(camera[1]))
        && (!method || item.method === method[1].toLowerCase())) : [];
      out.push({
        kind: "tag",
        value: (match[2] ?? "").replace(/^[\s·.-]+/, "").trim(),
        register,
        citation: matches.length === 1 ? matches[0] : undefined,
      });
      last = match.index + match[0].length;
    }
    if (last < text.length) out.push({ kind: "text", value: text.slice(last) });
    return out;
  }, [text, citations]);

  return (
    <>
      {parts.map((part, index) =>
        part.kind === "text" ? (
          <span key={index}>{part.value}</span>
        ) : part.citation ? (
          <button
            key={index}
            type="button"
            className="regtag"
            data-register={part.register}
            onClick={() => onCitation(part.citation!)}
            title={`Go to frame ${part.citation.frame}`}
          >
            {part.register}
            {part.value ? <span className="regtag__detail">{part.value}</span> : null}
          </button>
        ) : (
          <span key={index} className="regtag" data-register={part.register}>
            {part.register}
            {part.value ? <span className="regtag__detail">{part.value}</span> : null}
          </span>
        ),
      )}
    </>
  );
}

function ToolTrail({ calls }: { calls: AssistantToolCall[] }) {
  if (!calls.length) return null;
  return (
    <div className="tooltrail">
      {calls.map((call, index) => (
        <span key={`${call.name}-${index}`} className="tooltrail__item" data-status={call.status}>
          {call.status === "running" ? <span className="tooltrail__spin" aria-hidden="true" /> : null}
          {TOOL_LABELS[call.name] ?? call.name}
          {call.frames?.length ? <em> · frame {call.frames[0]}</em> : null}
        </span>
      ))}
    </div>
  );
}

function CitationRow({
  citations,
  onCitation,
  cameraLabel,
}: {
  citations: AssistantCitation[];
  onCitation: (citation: AssistantCitation) => void;
  cameraLabel: (index: number) => string;
}) {
  const verified = citations.filter(item => item.frame !== null && (item.kind === "computed" || item.kind === "observed"));
  if (!verified.length) return null;
  return (
    <div className="citerow">
      <span className="micro muted">Evidence</span>
      {verified.map((citation, index) => (
          <button
            key={index}
            type="button"
            className="cite"
            onClick={() => onCitation(citation)}
          >
            {citation.sec !== null ? clock(citation.sec) : `frame ${citation.frame}`}
            {citation.method ? ` · ${citation.method.toUpperCase()}` : ""}
            {citation.camera != null ? ` · ${cameraLabel(citation.camera)}` : ""}
            {citation.person ? ` · ${citation.person}` : ""}
            {citation.region ? <span> · {humanise(citation.region)}</span> : null}
            <IconArrowRight size={11} />
          </button>
        ),
      )}
    </div>
  );
}

/**
 * How many independent readings agreed.
 *
 * This is deliberately given more visual weight than the model's own stated
 * confidence, which sits beside it in muted text. One is measured across passes
 * over different frames; the other is the same forward pass grading itself.
 */
function AgreementBadge({ agreement }: { agreement?: Agreement }) {
  if (!agreement || !agreement.passes) return null;
  return (
    <span className="agree" data-band={agreement.band}>
      <span className="agree__pips" aria-hidden="true">
        {Array.from({ length: agreement.passes }, (_, index) => (
          <span key={index} data-on={index < agreement.agreeing} />
        ))}
      </span>
      {agreement.agreeing} of {agreement.passes} readings
    </span>
  );
}

function readableProse(value: string) {
  return value.replace(/\s*\u2014\s*/g, ", ");
}

function SceneLinks({ evidence, onEvidence, cameraLabel }: {
  evidence?: SceneEvidence[];
  onEvidence: (source: SceneEvidence) => void;
  cameraLabel: (index: number) => string;
}) {
  return evidence?.map(source => (
    <button type="button" className="cite" key={`${source.frame}-${source.camera}-${source.person}-${source.view}`} onClick={() => onEvidence(source)}>
      Frame {source.frame} · {cameraLabel(source.camera)}{source.person && ` · ${source.person}`}
    </button>
  ));
}

function ProposalCard({
  proposal,
  onDecide,
  onEvidence,
  cameraLabel,
  busy,
}: {
  proposal: SceneProposal;
  onEvidence: (source: SceneEvidence) => void;
  cameraLabel: (index: number) => string;
  onDecide: (decision: "accept" | "reject", proposal: SceneProposal) => void;
  busy: boolean;
}) {
  const value =
    proposal.value_range !== undefined
      ? `${proposal.value_range[0]}–${proposal.value_range[1]} kg`
      : String(proposal.value);
  const decided = proposal.status && proposal.status !== "pending";
  const contested = Boolean(proposal.contested);

  return (
    <article
      className="proposal"
      data-status={proposal.status ?? "pending"}
      data-contested={contested}
    >
      <div className="proposal__head">
        <div>
          <p className="micro muted">{humanise(proposal.condition)}</p>
          <strong>{value}</strong>
        </div>
        <AgreementBadge agreement={proposal.agreement} />
      </div>
      <p className="small">{readableProse(proposal.rationale)}</p>
      <SceneLinks evidence={proposal.evidence} onEvidence={onEvidence} cameraLabel={cameraLabel} />
      <p className="micro muted">
        {proposal.confidence} self-reported confidence
        {proposal.evidence_frames?.length
          ? ` · seen at frame ${proposal.evidence_frames.slice(0, 4).join(", ")}`
          : ""}
      </p>

      {contested && !decided ? (
        <p className="proposal__contested">
          <IconAlert size={12} /> The readings disagreed
          {proposal.agreement?.alternatives?.length
            ? `. Other readings suggested ${proposal.agreement.alternatives.map(String).join(", ")}`
            : ""}
          . Check the frames before accepting this.
        </p>
      ) : null}

      {decided ? (
        <p className="proposal__decided">
          {proposal.status === "accepted" ? "Accepted" : "Rejected"}. Decision recorded for the next
          analysis run.
        </p>
      ) : (
        <div className="proposal__actions">
          <button
            type="button"
            className={`btn btn--sm ${contested ? "btn--ghost" : "btn--primary"}`}
            disabled={busy}
            onClick={() => onDecide("accept", proposal)}
          >
            Accept as entered
          </button>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            disabled={busy}
            onClick={() => onDecide("reject", proposal)}
          >
            Reject
          </button>
        </div>
      )}
    </article>
  );
}

/**
 * Every figure in the answer, checked against the evidence the model was given.
 *
 * Shown even when everything traced, because "8 of 8 figures traced" is the
 * reassurance that makes the one time it says 7 of 8 mean something.
 */
function ProvenanceBar({ audit }: { audit?: AuditReport }) {
  if (!audit || !audit.total) return null;
  const clean = audit.untraced.length === 0;
  return (
    <div className="provenance" data-clean={clean}>
      <span className="provenance__count">
        {audit.traced} of {audit.total} numbers matched the assessment or tool results
      </span>
      {clean ? null : (
        <ul className="provenance__list">
          {audit.untraced.map((claim, index) => (
            <li key={index}>
              <strong>{claim.written}</strong>. Check this value against the cited evidence
              <span className="provenance__context">…{claim.context}…</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SceneCardPanel({
  sessionId,
  scene,
  proposals,
  onBuild,
  onEvidence,
  cameraLabel,
  building,
  buildStatus,
  onDecide,
  deciding,
}: {
  sessionId: string;
  scene: SceneCard | null;
  proposals: SceneProposal[];
  onBuild: () => void;
  onEvidence: (source: SceneEvidence) => void;
  cameraLabel: (index: number) => string;
  building: boolean;
  buildStatus: string;
  onDecide: (decision: "accept" | "reject", proposal: SceneProposal) => void;
  deciding: boolean;
}) {
  if (!scene) {
    return (
      <div className="scenecard scenecard--empty">
        <p className="insight-card__kicker">Visual scene reading</p>
        <h3>What the pose model could not see</h3>
        <p className="small muted">
          Review visible handholds and arm support across sampled frames. You can confirm or reject
          each proposed condition. Load mass and shock force require a measurement or task record.
        </p>
        <button type="button" className="btn btn--primary" onClick={onBuild} disabled={building}>
          {building ? buildStatus || "Reading the recording…" : "Read this recording"}
        </button>
        {building ? (
          <p className="micro muted">
            Sampling frames across the trial and interpreting them. About a minute.
          </p>
        ) : null}
      </div>
    );
  }

  const inference = scene.task_inference ?? {};
  const objects = scene.workstation?.objects ?? [];
  const pending = proposals.filter((item) => (item.status ?? "pending") === "pending");

  return (
    <div className="scenecard">
      <div className="scenecard__head">
        <div>
          <p className="insight-card__kicker">Visual activity classification</p>
          <h3>{inference.activity_class ? humanise(inference.activity_class) : "Scene reading"}</h3>
        </div>
        {/* Agreement across readings, not the model's own word for it. Falls back
            to the stated confidence only for cards built before passes existed. */}
        {inference.agreement ? (
          <AgreementBadge agreement={inference.agreement} />
        ) : inference.confidence ? (
          <span className="badge badge--neutral">{inference.confidence} confidence</span>
        ) : null}
      </div>

      {inference.summary ? <p className="small">{readableProse(inference.summary)}</p> : null}
      {inference.alternative_summaries?.length ? (
        <details className="small muted">
          <summary>Compare descriptions from other readings</summary>
          <ul>{inference.alternative_summaries.map((summary, i) => <li key={i}>{summary}</li>)}</ul>
        </details>
      ) : null}

      <p className="scenecard__disclaimer">
        <IconInfo size={13} /> Confirm visible conditions against the frames before using them in a rerun.
      </p>

      {objects.length ? (
        <div className="scenecard__objects">
          {objects.map((object) => (
            <span key={object} className="chip">
              {object}
            </span>
          ))}
        </div>
      ) : null}

      {scene.frames_sampled?.length ? (
        <p className="scenecard__method">
          Read <strong>{scene.frames_sampled.length} frames</strong>
          {scene.passes && scene.passes > 1 ? (
            <>
              {" "}
              across <strong>{scene.passes} readings of this recording</strong>
            </>
          ) : null}
          . Frames were chosen by posture coverage rather than clock time
          {scene.frameSelection ? (
            <>
              {" "}
              (<strong>{scene.frameSelection.coverageVsUniform}×</strong> the representativeness
              of evenly spaced sampling)
            </>
          ) : null}
          . {scene.model}
        </p>
      ) : null}

      {proposals.length ? (
        <div className="scenecard__proposals">
          <p className="micro muted">
            {pending.length
              ? `${pending.length} condition${pending.length === 1 ? "" : "s"} the analysis had to assume`
              : "All proposals decided"}
          </p>
          {proposals.map((proposal) => (
            <ProposalCard
              key={proposal.proposalId ?? proposal.condition}
              proposal={proposal}
              onEvidence={onEvidence}
              cameraLabel={cameraLabel}
              onDecide={onDecide}
              busy={deciding}
            />
          ))}
        </div>
      ) : (
        <p className="small muted">
          The footage provided insufficient evidence to propose changes to the recorded assumptions.
        </p>
      )}

      {scene.findings?.length ? (
        <Disclosure summary="Visual findings" hint={`${scene.findings.length} noted`}>
          <ul className="limitation-list">{scene.findings.map((finding, index) => <li key={index}>
            {readableProse(finding.fact)} <span className="micro muted">{finding.visibility}</span>
            <SceneLinks evidence={finding.evidence} onEvidence={onEvidence} cameraLabel={cameraLabel} />
          </li>)}</ul>
        </Disclosure>
      ) : scene.observations?.length ? (
        <Disclosure summary="Other observations" hint={`${scene.observations.length} noted`}>
          <ul className="limitation-list">
            {scene.observations.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </Disclosure>
      ) : null}

      <button type="button" className="btn btn--ghost btn--sm" onClick={onBuild} disabled={building}>
        {building ? buildStatus || "Re-reading…" : "Read again"}
      </button>
      <span className="micro muted" style={{ display: "block", marginTop: "var(--s-2)" }}>
        Session {sessionId.split("__").slice(1, 3).join(" · ")}
      </span>
    </div>
  );
}

export function Assistant({
  sessionId,
  method,
  selection,
  cameraLabel,
  onCitation,
  onFocusedReview,
  layout = "wide",
  onClose,
}: {
  sessionId: string;
  method: MethodKey;
  selection: ReviewSelection;
  cameraLabel: (index: number) => string;
  onCitation: (citation: AssistantCitation) => void;
  onFocusedReview?: () => void;
  /**
   * ``rail`` stacks the conversation and the scene reading behind a two-way
   * switch instead of placing them side by side. A docked rail is about 380px;
   * two columns in that is two unusable columns.
   */
  layout?: "wide" | "rail";
  onClose?: () => void;
}) {
  const [pane, setPane] = useState<"chat" | "scene">("chat");
  const health = useAsync((signal) => api.assistantHealth(signal, sessionId), [sessionId]);
  const [changingModel, setChangingModel] = useState(false);
  const [modelError, setModelError] = useState("");
  const context = useAsync((signal) => api.assistantContext(sessionId, signal, method), [sessionId, method]);

  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<AssistantToolCall[]>([]);
  const [scene, setScene] = useState<SceneCard | null>(null);
  const [proposals, setProposals] = useState<SceneProposal[]>([]);
  const [building, setBuilding] = useState(false);
  const [buildStatus, setBuildStatus] = useState("");
  const [deciding, setDeciding] = useState(false);
  const [sceneError, setSceneError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const thread = useRef<HTMLDivElement | null>(null);
  const restored = useRef(false);

  async function changeModel(model: string | null) {
    setChangingModel(true);
    setModelError("");
    try {
      await post(`/api/sessions/${encodeURIComponent(sessionId)}/assistant/model`, { model });
      health.reload();
      context.reload();
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "The model selection could not be saved.");
    } finally {
      setChangingModel(false);
    }
  }

  useEffect(() => {
    if (!context.data || context.data.sessionId !== sessionId) return;
    setScene(context.data.scene);
    setProposals(context.data.proposals?.proposals ?? []);
    if (!restored.current) {
      restored.current = true;
      const prior = (context.data.history ?? []).flatMap<AssistantMessage>(turn => [
        { role: "user", content: turn.question },
        { role: "assistant", content: turn.answer, citations: turn.citations, audit: turn.audit, toolCalls: turn.toolCalls },
      ]);
      setMessages(active => [...prior, ...active]);
    }
  }, [context.data, sessionId]);

  useEffect(() => {
    restored.current = false;
    setMessages([]);
    setDraft("");
    setLive([]);
  }, [sessionId]);

  useEffect(() => () => abort.current?.abort(), []);

  // Keep the newest reply in view while it streams, but never fight a reader who
  // has scrolled up to re-read something.
  useEffect(() => {
    const node = thread.current;
    if (!node) return;
    const nearBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 160;
    if (nearBottom) node.scrollTop = node.scrollHeight;
  }, [messages, live]);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy) return;
      setDraft("");
      setBusy(true);
      setLive([]);

      const history = messages
        .filter((item) => !item.error)
        .map((item) => ({ role: item.role, content: item.content }));

      setMessages((current) => [
        ...current,
        { role: "user", content: trimmed },
        { role: "assistant", content: "", streaming: true },
      ]);

      const controller = new AbortController();
      abort.current = controller;

      const patch = (change: Partial<AssistantMessage>) =>
        setMessages((current) => {
          const next = [...current];
          const last = next.length - 1;
          if (last >= 0 && next[last].role === "assistant") {
            next[last] = { ...next[last], ...change };
          }
          return next;
        });

      try {
        await readEventStream(
          `/api/sessions/${encodeURIComponent(sessionId)}/assistant/messages`,
          { question: trimmed, history, selection },
          (event) => {
            const kind = String(event.type);
            if (kind === "delta") {
              const text = String(event.text ?? "");
              setMessages((current) => {
                const next = [...current];
                const last = next.length - 1;
                if (last >= 0 && next[last].role === "assistant") {
                  next[last] = { ...next[last], content: next[last].content + text };
                }
                return next;
              });
            } else if (kind === "reset") {
              // The model drafted an answer, then decided to check a tool. That
              // draft is about to be rewritten and often carries figures it is
              // about to correct, so it is discarded rather than shown.
              patch({ content: "" });
            } else if (kind === "tool") {
              const call: AssistantToolCall = {
                name: String(event.name),
                arguments: (event.arguments ?? {}) as Record<string, unknown>,
                status: event.status as "running" | "done",
                frames: (event.frames ?? []) as number[],
              };
              setLive((current) => {
                const index = current.findIndex(
                  (item) => item.name === call.name && item.status === "running",
                );
                if (call.status === "done" && index >= 0) {
                  const next = [...current];
                  next[index] = call;
                  return next;
                }
                return [...current, call];
              });
            } else if (kind === "done") {
              patch({
                content: String(event.text ?? ""),
                citations: (event.citations ?? []) as AssistantCitation[],
                toolCalls: (event.toolCalls ?? []) as AssistantToolCall[],
                audit: event.audit as AuditReport | undefined,
                streaming: false,
              });
            } else if (kind === "error") {
              patch({ streaming: false, error: String(event.detail ?? "The assistant failed.") });
            }
          },
          controller.signal,
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          patch({
            streaming: false,
            error: error instanceof Error ? error.message : "The assistant could not be reached.",
          });
        }
      } finally {
        setBusy(false);
        setLive([]);
        abort.current = null;
      }
    },
    [busy, messages, sessionId, selection],
  );

  const build = useCallback(async () => {
    if (building) return;
    setBuilding(true);
    setSceneError("");
    setBuildStatus("Sampling frames…");
    const controller = new AbortController();
    try {
      await readEventStream(
        `/api/sessions/${encodeURIComponent(sessionId)}/scene`,
        {},
        (event) => {
          const kind = String(event.type);
          if (kind === "status") setBuildStatus(String(event.detail ?? ""));
          else if (kind === "done") {
            setScene(event.scene as SceneCard);
            void api.proposals(sessionId).then((payload) => setProposals(payload.proposals ?? [])).catch(error => setSceneError(error.message));
          } else if (kind === "error") setSceneError(String(event.detail ?? "Could not read the recording."));
        },
        controller.signal,
      );
    } catch (error) {
      setSceneError(error instanceof Error ? error.message : "Could not read the recording");
    } finally {
      setBuilding(false);
    }
  }, [building, sessionId]);

  const decide = useCallback(
    async (decision: "accept" | "reject", {condition, proposalId}: SceneProposal) => {
      setDeciding(true);
      setSceneError("");
      try {
        const payload = await api.decideProposal(sessionId, decision, {condition, proposalId});
        setProposals(payload.proposals ?? []);
      } catch (error) {
        setSceneError(error instanceof Error ? error.message : "Could not update this proposal.");
        context.reload();
      } finally {
        setDeciding(false);
      }
    },
    [sessionId, context.reload],
  );

  const suggestions = context.data?.suggestions ?? [];
  const offline = health.data && !health.data.ok;
  const rail = layout === "rail";
  const pendingProposals = proposals.filter(
    (item) => (item.status ?? "pending") === "pending",
  ).length;

  return (
    <div className="assistant" data-layout={layout}>
      {rail ? (
        <header className="railhead">
          <div className="railhead__switch" role="tablist" aria-label="Assistant panes">
            <button
              type="button"
              role="tab"
              aria-selected={pane === "chat"}
              onClick={() => setPane("chat")}
            >
              Ask
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={pane === "scene"}
              onClick={() => setPane("scene")}
            >
              Scene
              {pendingProposals ? <span className="railhead__dot">{pendingProposals}</span> : null}
            </button>
          </div>
          <button
            type="button"
            className="railhead__close"
            onClick={onClose}
            aria-label="Close the assistant"
            title="Close"
          >
            ×
          </button>
        </header>
      ) : null}

      <div className="assistant__main" hidden={rail && pane !== "chat"}>
        <details><summary>Model settings</summary>
          <VisionModelSelect sessionId={sessionId} editingSessionDefault value={health.data?.selection ?? null}
            disabled={busy || building || changingModel || health.loading} onChange={model => void changeModel(model)} />
          {modelError && <p role="alert" className="workflow-error">{modelError}</p>}
        </details>
        {onFocusedReview && <button type="button" className="btn btn--ghost btn--sm" onClick={onFocusedReview}>
          Review grip and arm support
        </button>}
        <div className="assistant__status">
          {health.loading ? (
            <span className="micro muted">Checking the model…</span>
          ) : offline ? (
            <span className="assistant__offline">
              <IconAlert size={13} /> {health.data?.detail}
            </span>
          ) : (
            <span className="micro muted">
              <span className="dot dot--ok" aria-hidden="true" /> {health.data?.model} ·{" "}
              {health.data?.local ? "on this machine" : "remote endpoint"}
            </span>
          )}
        </div>

        <div className="assistant__thread" ref={thread}>
          {messages.length === 0 ? (
            <div className="assistant__intro">
              <IconPerson size={22} />
              <h3>Ask about this assessment</h3>
              <p className="small muted">
                Ask what drove a score or what the worker was doing at the selected moment.
                Open an evidence link to review the calculation or inspect the source frame.
              </p>
            </div>
          ) : null}

          {messages.map((message, index) => (
            <div key={index} className={`bubble bubble--${message.role}`}>
              {message.role === "assistant" ? (
                <>
                  {message.toolCalls?.length ? <ToolTrail calls={message.toolCalls} /> : null}
                  {message.error ? (
                    <p className="bubble__error">
                      <IconAlert size={13} /> {message.error}
                    </p>
                  ) : (
                    <div className="bubble__body">
                      <TaggedText text={readableProse(message.content)} citations={message.citations ?? []} onCitation={onCitation} />
                      {message.streaming ? <span className="caret" aria-hidden="true" /> : null}
                    </div>
                  )}
                  <CitationRow citations={message.citations ?? []} onCitation={onCitation} cameraLabel={cameraLabel} />
                  <ProvenanceBar audit={message.audit} />
                </>
              ) : (
                <div className="bubble__body">{message.content}</div>
              )}
            </div>
          ))}

          {busy && live.length ? <ToolTrail calls={live} /> : null}
        </div>

        {suggestions.length && messages.length === 0 ? (
          <div className="assistant__chips">
            {suggestions.map((question) => (
              <button
                key={question}
                type="button"
                className="chip chip--action"
                onClick={() => void ask(question)}
                disabled={busy || Boolean(offline)}
              >
                <IconSearch size={12} />
                {question}
              </button>
            ))}
          </div>
        ) : null}

        <p className="micro muted">Reviewing {method.toUpperCase()}
          {selection.frame != null && ` · frame ${selection.frame}`}
          {selection.person && ` · ${selection.person}`}{` · ${cameraLabel(selection.camera)}`}
          {selection.region && ` · ${humanise(selection.region)}`}
        </p>
        <form
          className="assistant__composer"
          onSubmit={(event) => {
            event.preventDefault();
            void ask(draft);
          }}
        >
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void ask(draft);
              }
            }}
            placeholder={
              offline
                ? "Assessment chat is unavailable"
                : `Ask about this ${method.toUpperCase()} assessment…`
            }
            rows={2}
            disabled={busy || Boolean(offline)}
          />
          <button
            type="submit"
            className="btn btn--primary"
            disabled={busy || !draft.trim() || Boolean(offline)}
          >
            {busy ? "Working…" : "Ask"}
          </button>
        </form>
        <p className="micro muted assistant__foot">
          Computed labels identify assessment data. Observed labels identify visual evidence.
          Judgment labels identify interpretation.
        </p>
      </div>

      <aside className="assistant__side" hidden={rail && pane !== "scene"}>
        {sceneError && <p className="workflow-error" role="alert">{sceneError}</p>}
        {context.loading ? (
          <Skeleton height={280} radius="var(--r-xl)" />
        ) : (
          <SceneCardPanel
            sessionId={sessionId}
            scene={scene}
            proposals={proposals}
            onBuild={() => void build()}
            onEvidence={source => onCitation({...source, method: null, region: null, kind: "observed"})}
            cameraLabel={cameraLabel}
            building={building}
            buildStatus={buildStatus}
            onDecide={(decision, proposal) => void decide(decision, proposal)}
            deciding={deciding}
          />
        )}
      </aside>
    </div>
  );
}
