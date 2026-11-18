import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { IconArrowRight, IconPause, IconPlay, IconRecord, IconVideo } from "../components/Icons";
import { CaptureNotice, DraftList } from "../components/WorkflowParts";
import "../styles/home.css";

let hasVisitedHome = false;

export default function Home() {
  const [paused, setPaused] = useState(() => {
    try {
      return localStorage.getItem("ergopartner-home-motion") === "paused";
    } catch {
      return false;
    }
  });
  const [showEntrance] = useState(() => !hasVisitedHome && !paused);
  useEffect(() => {
    hasVisitedHome = true;
  }, []);

  function toggleMotion() {
    const next = !paused;
    setPaused(next);
    try {
      localStorage.setItem("ergopartner-home-motion", next ? "paused" : "playing");
    } catch {
      /* Animation remains controllable without browser storage. */
    }
  }

  return (
    <div className="home-canvas" data-paused={paused} data-entrance={showEntrance}>
      <div className="home-backdrop" aria-hidden="true">
        <div className="home-brand-art">
          <img src="/brand/ergopartner-symbol.svg" alt="" />
        </div>
      </div>
      <header className="home-nav">
        <Link to="/" aria-label="ErgoPartner home">
          <img src="/brand/ergopartner-horizontal.svg" alt="ErgoPartner" width="671" height="128" />
        </Link>
        <nav aria-label="Workspace">
          <Link to="/assessments">Assessments</Link>
          <Link to="/overview">Study overview</Link>
          <Link to="/settings">Settings</Link>
        </nav>
      </header>
      <CaptureNotice />
      <main className="home-main">
        <section className="home-hero" aria-labelledby="home-title">
          <div className="home-intro">
            <h1 id="home-title">
              Assess a<br />
              work task.
            </h1>
            <p>Record a task or review footage you already have.</p>
            <div className="home-actions">
              <Link className="btn btn--primary btn--lg" to="/new?source=live">
                <IconRecord size={18} />
                Record live
              </Link>
              <Link className="btn btn--lg" to="/new?source=video">
                <IconVideo size={18} />
                Analyze recorded video
              </Link>
            </div>
          </div>
          <button
            type="button"
            className="home-motion"
            onClick={toggleMotion}
            aria-label={paused ? "Play logo animation" : "Pause logo animation"}
          >
            {paused ? <IconPlay size={15} /> : <IconPause size={15} />}
          </button>
        </section>
        <section className="home-recent" aria-labelledby="home-recent-title">
          <div className="workflow-heading">
            <h2 id="home-recent-title">Continue an assessment</h2>
            <Link to="/assessments">
              All assessments <IconArrowRight size={15} />
            </Link>
          </div>
          <DraftList limit={4} />
        </section>
      </main>
    </div>
  );
}
