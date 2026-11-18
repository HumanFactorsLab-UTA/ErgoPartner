import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";

import Landing from "./screens/Landing";
import Library from "./screens/Library";
import AnalysisPicker from "./screens/AnalysisPicker";
import Workspace from "./screens/Workspace";
import Home from "./screens/Home";
import NewAssessment from "./screens/NewAssessment";
import AssessmentSetup from "./screens/AssessmentSetup";
import Settings from "./screens/Settings";
import Research from "./screens/Research";
import NotFound from "./screens/NotFound";

const router = createBrowserRouter([
  { path: "/", element: <Home /> },
  { path: "/overview", element: <Landing /> },
  { path: "/new", element: <NewAssessment /> },
  { path: "/record", element: <Navigate to="/new?source=live" replace /> },
  { path: "/library", element: <Navigate to="/assessments" replace /> },
  { path: "/assessments", element: <Library /> },
  { path: "/assessments/:assessmentId", element: <AssessmentSetup /> },
  { path: "/research", element: <Research /> },
  { path: "/settings", element: <Settings /> },
  { path: "/s/:sessionId", element: <AnalysisPicker /> },
  { path: "/s/:sessionId/:method", element: <Workspace /> },
  { path: "*", element: <NotFound /> },
]);

const container = document.getElementById("root");
if (!container) throw new Error("The application root element is missing from index.html");

createRoot(container).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
