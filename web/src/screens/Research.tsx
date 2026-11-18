import { TestResults } from "../components/FocusedReview";
import { Shell } from "../components/Shell";

export default function Research() {
  return (
    <Shell crumbs={[{ label: "Settings", to: "/settings" }, { label: "Research evaluation" }]}>
      <section className="assessment-setup stack stack-5">
        <h1>Research evaluation</h1>
        <h2>Visual review agreement</h2>
        <TestResults />
      </section>
    </Shell>
  );
}
