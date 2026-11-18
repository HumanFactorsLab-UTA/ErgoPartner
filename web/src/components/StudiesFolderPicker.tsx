import { useEffect, useRef, useState } from "react";
import { api, useAsync } from "../lib/api";
import type { FolderListing } from "../lib/types";
import { IconArrowRight, IconFolder } from "./Icons";

function FolderBrowser({ initialPath, onClose, onSelected }: {
  initialPath: string;
  onClose: () => void;
  onSelected: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const pending = useRef<AbortController | null>(null);
  const [address, setAddress] = useState(initialPath);
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function browse(path: string) {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setError(null);
    try {
      const next = await api.browseFolders(path, controller.signal);
      if (controller.signal.aborted) return;
      setListing(next);
      setAddress(next.path);
    } catch (err) {
      if (!controller.signal.aborted) {
        setListing(null);
        setError(err instanceof Error ? err.message : "Could not open this folder.");
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  useEffect(() => {
    dialog.current?.showModal();
    void browse(initialPath);
    return () => pending.current?.abort();
  }, [initialPath]);

  async function select() {
    if (!listing) return;
    setSaving(true);
    setError(null);
    try {
      await api.selectStudiesFolder(listing.path);
      onSelected();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save this folder.");
      setSaving(false);
    }
  }

  return (
    <dialog ref={dialog} className="folder-browser" aria-labelledby="folder-browser-title"
      onCancel={(event) => { event.preventDefault(); if (!saving) onClose(); }}>
      <div className="stack stack-4">
        <div>
          <p className="eyebrow">Study library</p>
          <h2 id="folder-browser-title">Choose studies folder</h2>
          <p className="folder-browser__hint">Choose the folder containing your studies. Recordings stay on this computer.</p>
        </div>
        <form className="folder-browser__address" onSubmit={(event) => { event.preventDefault(); void browse(address); }}>
          <label className="field">
            <IconFolder />
            <input aria-label="Folder path" value={address} disabled={saving}
              onChange={(event) => setAddress(event.target.value)} autoFocus />
          </label>
          <button className="btn btn--ghost" disabled={loading || saving || !address.trim()}>Go</button>
        </form>
        <div className="row row-2 wrap">
          <button className="btn btn--ghost btn--sm" type="button" disabled={loading || saving || !listing?.parent}
            onClick={() => listing?.parent && void browse(listing.parent)}>Up one folder</button>
          {listing?.drives.map((drive) => (
            <button className="btn btn--ghost btn--sm" type="button" key={drive} disabled={loading || saving}
              aria-label={`Open drive ${drive}`} onClick={() => void browse(drive)}>{drive}</button>
          ))}
        </div>
        {error && <p role="alert" className="folder-browser__error">{error}</p>}
        <div className="folder-browser__folders" aria-busy={loading}>
          {loading ? <p role="status">Opening folder…</p> : listing?.folders.length ? (
            <ul aria-label="Folders">
              {listing.folders.map((folder) => (
                <li key={folder.path}>
                  <button type="button" disabled={saving} onClick={() => void browse(folder.path)}
                    aria-label={`Open folder ${folder.name}`}>
                    <IconFolder size={18} /><span>{folder.name}</span>
                    {folder.isStudy && <small>Study</small>}<IconArrowRight size={14} />
                  </button>
                </li>
              ))}
            </ul>
          ) : listing ? <p>No subfolders here.</p> : <p>Enter a folder path above to continue.</p>}
        </div>
        <p className="folder-browser__hint" aria-live="polite">
          {listing && !loading ? listing.studyCount
            ? `${listing.studyCount} ${listing.studyCount === 1 ? "study" : "studies"} found in this folder.`
            : "No studies in this folder. Look for the parent folder that contains your studies."
            : "Folders with a study configuration are marked Study."}
        </p>
        <div className="folder-browser__actions">
          <button type="button" className="btn btn--ghost" disabled={saving} onClick={onClose}>Cancel</button>
          <button type="button" className="btn btn--primary" onClick={() => void select()}
            disabled={saving || loading || !listing?.studyCount || address !== listing.path}>
            {saving ? "Saving…" : "Use this folder"}
          </button>
        </div>
      </div>
    </dialog>
  );
}

export function StudiesFolderPicker({ disabled, onChanged }: { disabled: boolean; onChanged: () => void }) {
  const { data, error, reload } = useAsync((signal) => api.studiesFolder(signal), []);
  const [open, setOpen] = useState(false);
  return (
    <section className="studies-folder" aria-label="Studies folder">
      <IconFolder size={22} />
      <div className="studies-folder__path">
        <strong>Studies folder</strong>
        <span>{data?.path ?? error ?? "Loading folder…"}</span>
        {data && !data.exists && <small>This folder is unavailable. Browse to choose another.</small>}
        {disabled && <small>You can change folders when the current analysis finishes.</small>}
      </div>
      <button type="button" className="btn btn--ghost" disabled={disabled} onClick={() => setOpen(true)}>
        <IconFolder /> Browse folder
      </button>
      {open && <FolderBrowser initialPath={data?.path ?? ""} onClose={() => setOpen(false)}
        onSelected={() => { setOpen(false); reload(); onChanged(); }} />}
    </section>
  );
}
