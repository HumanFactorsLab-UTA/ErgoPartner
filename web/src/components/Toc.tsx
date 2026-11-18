import { useEffect, useState } from "react";

export interface TocEntry {
  id: string;
  label: string;
}

/**
 * Highlights whichever section is currently in the reading position.
 *
 * IntersectionObserver alone reports "this section is somewhere on screen",
 * which on a long page means three of them at once and a table of contents that
 * flickers between them. Tracking the *topmost section that has passed the
 * reading line* gives one unambiguous answer, and matches what a reader would
 * say they are looking at.
 */
export function useScrollSpy(ids: string[], offset = 140): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  // The caller builds this array inline, so a fresh reference arrives on every
  // render. Keying the effect on the joined ids stops it tearing down and
  // rebuilding its listeners on renders that changed nothing.
  const key = ids.join("|");

  useEffect(() => {
    const sections = key ? key.split("|") : [];
    if (sections.length === 0) return;

    let scheduled = 0;
    let timer = 0;

    const measure = () => {
      scheduled = 0;
      timer = 0;
      let current = sections[0];
      for (const id of sections) {
        const node = document.getElementById(id);
        if (!node) continue;
        if (node.getBoundingClientRect().top - offset <= 0) current = id;
      }
      // At the very bottom the last section may never cross the line, so the
      // final entry wins once the page cannot scroll further.
      const atEnd = window.innerHeight + window.scrollY >= document.body.scrollHeight - 2;
      setActive(atEnd ? sections[sections.length - 1] : current);
    };

    // requestAnimationFrame is the right throttle while the tab is visible and
    // is never called while it is hidden, which would leave the highlight stale
    // on return. The timeout is the floor that keeps it correct either way.
    const onScroll = () => {
      if (scheduled || timer) return;
      scheduled = window.requestAnimationFrame(measure);
      timer = window.setTimeout(measure, 120);
    };

    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      if (scheduled) window.cancelAnimationFrame(scheduled);
      if (timer) window.clearTimeout(timer);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [key, offset]);

  return active;
}

export function Toc({ entries }: { entries: TocEntry[] }) {
  const ids = entries.map((entry) => entry.id);
  const active = useScrollSpy(ids);

  return (
    <nav className="toc" aria-label="On this page">
      <p className="toc__title">On this page</p>
      {entries.map((entry) => (
        <a
          key={entry.id}
          href={`#${entry.id}`}
          data-active={entry.id === active}
          onClick={(event) => {
            event.preventDefault();
            const node = document.getElementById(entry.id);
            if (!node) return;
            node.scrollIntoView({ behavior: "smooth", block: "start" });
            history.replaceState(null, "", `#${entry.id}`);
          }}
        >
          {entry.label}
        </a>
      ))}
    </nav>
  );
}
