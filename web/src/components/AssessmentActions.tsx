import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { workflow, type TrashTarget } from "../lib/workflow";

export function ActionMenu({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  return (
    <div
      className="workflow-menu"
      ref={host}
      onBlur={(event) => {
        if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node))
          setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          setOpen(false);
          trigger.current?.focus();
        }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          if (!open) {
            setOpen(true);
            requestAnimationFrame(() =>
              host.current
                ?.querySelector<HTMLElement>(
                  ".workflow-menu__panel button:not(:disabled), .workflow-menu__panel a",
                )
                ?.focus(),
            );
          } else {
            const actions = Array.from(
              host.current?.querySelectorAll<HTMLElement>(
                ".workflow-menu__panel button:not(:disabled), .workflow-menu__panel a",
              ) ?? [],
            );
            if (actions.length) {
              const i = actions.indexOf(document.activeElement as HTMLElement);
              actions[
                (i + (event.key === "ArrowDown" ? 1 : -1) + actions.length) % actions.length
              ]?.focus();
            }
          }
        }
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="btn btn--ghost workflow-menu__trigger"
        aria-label={label}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        <span aria-hidden="true">•••</span>
      </button>
      {open && (
        <div id={id} className="workflow-menu__panel" role="group" aria-label={label}>
          {children}
        </div>
      )}
    </div>
  );
}

export interface TrashSelection {
  target: TrashTarget;
  label: string;
  taskCount?: number;
}
export function TrashDialog({
  selection,
  onClose,
  onMoved,
}: {
  selection: TrashSelection;
  onClose: () => void;
  onMoved: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const el = dialog.current;
    el?.showModal();
    return () => el?.close();
  }, []);
  async function move() {
    setBusy(true);
    setError("");
    try {
      await workflow.moveToTrash(selection.target);
      onMoved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not move this item to Trash.");
      setBusy(false);
    }
  }
  return (
    <dialog
      className="workflow-dialog"
      ref={dialog}
      aria-labelledby={title}
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onClose();
      }}
    >
      <div className="stack stack-4">
        <h2 id={title}>Move {selection.label} to Trash?</h2>
        <p>
          {selection.target.kind === "participant"
            ? `This includes ${selection.taskCount ?? 0} ${selection.taskCount === 1 ? "task" : "tasks"}. `
            : selection.target.kind === "draft"
              ? "This moves the saved setup. "
              : ""}
          You can restore this item from Trash. Original recordings stay on disk.
        </p>
        {error && (
          <p role="alert" className="workflow-error">
            {error}
          </p>
        )}
        <div className="row row-2 wrap">
          <button type="button" className="btn" autoFocus disabled={busy} onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn--danger"
            disabled={busy}
            onClick={() => void move()}
          >
            {busy ? "Moving…" : "Move to Trash"}
          </button>
        </div>
      </div>
    </dialog>
  );
}
