import { Fragment, useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useParams } from "react-router-dom";
import {
  IconFolder,
  IconGauge,
  IconLibrary,
  IconPerson,
} from "./Icons";
import type { MethodKey, SessionOverview } from "../lib/types";
import { CaptureNotice } from "./WorkflowParts";

export interface Crumb {
  label: string;
  to?: string;
}

let hasShownWordmark = false;

function BrandLink() {
  const [animate] = useState(() => !hasShownWordmark);
  useEffect(() => { hasShownWordmark = true; }, []);
  return (
    <Link to="/" className="sidebar__brand" aria-label="ErgoPartner home">
      <picture>
        <source media="(prefers-reduced-motion: reduce)" srcSet="/brand/ergopartner-horizontal.svg" />
        <img src={animate ? "/brand/ergopartner-reveal.svg" : "/brand/ergopartner-horizontal.svg"}
          alt="ErgoPartner" width="671" height="128" />
      </picture>
    </Link>
  );
}

/**
 * The workflow navigator.
 *
 * Global destinations are always available. Assessment-specific navigation
 * appears once a recording is loaded, keeping the entry and setup screens focused.
 */
function Sidebar({
  session,
  method,
}: {
  session: SessionOverview | null;
  method: MethodKey | null;
}) {
  return (
    <aside className="sidebar">
      <BrandLink />

      {session ? (
        <div className="sidebar__context">
          <div className="sidebar__contextrow">
            <span className="sidebar__ctxlabel">Participant</span>
            <span className="sidebar__ctxvalue" title={session.participant.label}>
              {session.participant.participantId}
            </span>
          </div>
          <div className="sidebar__contextrow">
            <span className="sidebar__ctxlabel">Recording</span>
            <span className="sidebar__ctxvalue" title={session.trial.name}>
              {session.trial.name}
            </span>
          </div>
          <div className="sidebar__contextrow">
            <span className="sidebar__ctxlabel">Analysis</span>
            <span className="sidebar__ctxvalue">
              {method ? method.toUpperCase() : "Not chosen"}
            </span>
          </div>
        </div>
      ) : null}

      <nav className="sidebar__nav" aria-label="Workflow">
        <div className="navgroup">
          <p className="navgroup__title">Workspace</p>
          <NavLink to="/" end className="navitem">
            <IconPerson className="navitem__icon" />
            Home
          </NavLink>
          <NavLink to="/assessments" className="navitem">
            <IconLibrary className="navitem__icon" />
            Assessments
          </NavLink>
          <NavLink to="/overview" className="navitem">
            <IconGauge className="navitem__icon" />
            Study overview
          </NavLink>
          <NavLink to="/settings" className="navitem">
            <IconFolder className="navitem__icon" />
            Settings
          </NavLink>
        </div>

      </nav>
    </aside>
  );
}

function Breadcrumbs({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav className="crumbs" aria-label="Breadcrumb">
      {crumbs.map((crumb, index) => {
        const last = index === crumbs.length - 1;
        return (
          <Fragment key={`${crumb.label}-${index}`}>
            {index > 0 ? (
              <span className="crumbs__sep" aria-hidden="true">
                /
              </span>
            ) : null}
            {crumb.to && !last ? (
              <Link to={crumb.to}>{crumb.label}</Link>
            ) : (
              <span className={last ? "crumbs__current" : undefined}>{crumb.label}</span>
            )}
          </Fragment>
        );
      })}
    </nav>
  );
}

export function Shell({
  children,
  crumbs = [],
  session = null,
  method = null,
  actions,
  wide = false,
  rail = null,
}: {
  children: ReactNode;
  crumbs?: Crumb[];
  session?: SessionOverview | null;
  method?: MethodKey | null;
  actions?: ReactNode;
  wide?: boolean;
  /**
   * A panel docked to the right of the content, for tools that act *on* what is
   * on screen rather than replacing it. The assistant lives here: its answers
   * move the timeline and the frame panel, which it cannot do from a tab that
   * hides them.
   */
  rail?: ReactNode;
}) {
  const params = useParams();
  return (
    <div className="shell" data-rail={rail ? "open" : "closed"}>
      <Sidebar session={session} method={method} />
      <div className="main">
        <header className="topbar">
          <Breadcrumbs crumbs={crumbs} />
          <span className="topbar__spacer" />
          {actions}
        </header>
        <CaptureNotice />
        <main
          className={`page${wide ? " page--wide" : ""}`}
          // Keyed so switching session or method remounts the page and replays
          // its enter animation. The tab is deliberately not in the key: moving
          // between tabs is navigation within one screen, not a new one.
          key={`${params.sessionId ?? "root"}-${params.method ?? "index"}`}
        >
          {children}
        </main>
      </div>
      {rail ? <aside className="rail">{rail}</aside> : null}
    </div>
  );
}
