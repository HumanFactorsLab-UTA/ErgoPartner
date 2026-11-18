import { Link } from "react-router-dom";
import { Shell } from "../components/Shell";
import { EmptyState } from "../components/ui";

export default function NotFound() {
  return (
    <Shell crumbs={[{ label: "Dashboard", to: "/" }, { label: "Not found" }]}>
      <EmptyState
        title="That page does not exist"
        detail="The link may be from an older version of the application, or the recording it pointed at has been removed."
        action={
          <Link className="btn btn--primary" to="/">
            Back to the dashboard
          </Link>
        }
      />
    </Shell>
  );
}
