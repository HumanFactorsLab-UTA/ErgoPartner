// Run the reviewer controls against an isolated, repeatable set of image readings.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= "0";
const { chromium } = await import("playwright");
const root = fileURLToPath(new URL("../", import.meta.url));
const entry = `
  import React, {useState} from 'react';
  import {createRoot} from 'react-dom/client';
  import {FocusedReview} from '/src/components/FocusedReview.tsx';
  import '/src/styles/tokens.css';
  import '/src/styles/base.css';
  import '/src/styles/app.css';
  function Test() {
    const [frame, setFrame] = useState(10);
    const [camera, setCamera] = useState(0);
    return React.createElement(FocusedReview, {sessionId:'test', method:'rula', person:'worker-A',
      frame, camera, cameras:[{index:0,label:'Front'},{index:1,label:'Side'}],
      onFrame:setFrame,onCamera:setCamera});
  }
  createRoot(document.getElementById('root')).render(React.createElement(Test));
`;
const server = await createServer({
  root, logLevel: "error", server: {host:"127.0.0.1", port:0}, plugins: [{
    name: "focused-review-check",
    resolveId: id => id === "/focused-browser-test.js" ? id : undefined,
    load: id => id === "/focused-browser-test.js" ? entry : undefined,
  }],
});
await server.listen();
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  const cases = new Map();
  const requests = [];
  const options = values => values.map(([value, label]) => ({value, label}));
  const questions = [
    {key:"grip",prompt:"Where is the grasp?",options:options([["object_grasp","Object grip"],["no_grasp","Hands free"],["unclear","Cannot tell"]])},
    {key:"armSupport",prompt:"Is the arm supported?",options:options([["contact","Resting contact"],["clear_gap","Visible gap"],["unclear","Cannot tell"]])},
    {key:"visibility",prompt:"Can the contact areas be seen?",options:options([["clear","Clear view"],["partial","Partly visible"],["unclear","Neither area can be judged"]])},
  ];
  let staleDecision = true;
  const html = await server.transformIndexHtml("/focused-test",
    '<!doctype html><html><head><meta name="viewport" content="width=device-width"></head><body><div id="root"></div><script type="module" src="/focused-browser-test.js"></script></body></html>');
  await page.route("**/focused-test", route => route.fulfill({contentType:"text/html",body:html}));
  await page.route("**/api/**", async route => {
    const req = route.request();
    const url = new URL(req.url());
    const resource = url.pathname.split("/").at(-1);
    const json = data => route.fulfill({json:data});
    if (resource.endsWith(".svg")) return route.fulfill({contentType:"image/svg+xml",
      body:'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#567"/></svg>'});
    if (resource === "samples") return json({frames:[{frame:10,sec:1},{frame:20,sec:2}]});
    if (resource === "evaluation") {
      const metric = {n:1, correct:0,accuracy:0,unclearCases:1,correctAbstentions:0,falseDefinites:1,unnecessaryAbstentions:0};
      const metrics = Object.fromEntries(questions.map(({key}) => [key,metric]));
      return json({caseCount:2,labelledCases:1,pairedCases:1,assistedCases:1,metrics,model:"test",promptVersion:"v1",
        byVisibility:{unclear:{caseCount:1,metrics}},byCameraMode:{multi:{caseCount:1,metrics}}});
    }
    const body = req.method() === "POST" ? req.postDataJSON() : Object.fromEntries(url.searchParams);
    const frame = Number(body.frame), camera = Number(body.camera);
    const id = `${body.person}-${frame}-${camera}`;
    if (!cases.has(id)) {
      const target = {frame, camera, person:body.person, method:body.method};
      const evidence = [0, 1].map(i => ({...target,frame:frame+i,sourceFrame:frame+i,sec:(frame+i)/10,
        view:"raw",imageIndex:i+1, imageUrl:`/api/${id}/${i}.svg`}));
      cases.set(id,{...target,caseId:id,questions,evidence,cameraMode:"multi",blindEligible:true,reference:null,prediction:null,decision:null});
    }
    const record = cases.get(id);
    if (req.method() === "POST") {
      assert.equal(body.caseId, id);
      requests.push({resource,...body});
      if (resource === "reference") {
        assert.equal(record.blindEligible, true);
        record.reference = {answers:body.answers};
      } else if (resource === "predict") {
        record.blindEligible = false;
        record.prediction = {id:"test-prediction",model:"test",promptVersion:"v1",answers:Object.fromEntries(
          questions.map(q => [q.key,{value:q.options[0].value,reason:"A visible cue.",evidence:[record.evidence[1]]}]))};
      } else if (resource === "decision") {
        assert.equal(body.predictionId, "test-prediction");
        if (staleDecision) {
          staleDecision = false;
          return route.fulfill({status:409,json:{detail:"This evidence changed. Review the current reading."}});
        }
        record.decision = {answers:body.answers};
      }
    }
    return json(record);
  });
  await page.goto(new URL("/focused-test", server.resolvedUrls.local[0]).href);
  const button = name => page.getByRole("button",{name,exact:true});
  await button("Blind test").click();
  await page.getByRole("group",{name:"Visibility",exact:true}).getByLabel("Neither area can be judged").check();
  assert.deepEqual(requests, []);
  await button("Save independent labels").click();
  await page.getByText("Independent labels locked").waitFor();
  assert.deepEqual(requests[0].answers, {grip:"unclear",armSupport:"unclear",visibility:"unclear"});
  await button("Compare with AI").click();
  await page.getByText("AI · Object grip.").waitFor();
  assert.deepEqual(requests.map(action => action.resource), ["reference","predict"]);
  await page.getByRole("group",{name:"Grip",exact:true}).getByRole("button",{name:"Frame 11"}).click();
  await button("Enlarge Frame 11").click();
  await page.getByRole("dialog").waitFor();
  await page.keyboard.press("Escape");
  await button("View test results").click();
  await page.getByRole("columnheader",{name:"Overconfident calls"}).waitFor();
  await page.getByLabel("Show results for").selectOption("visibility-unclear");

  await page.getByLabel("Test moment").selectOption("20");
  await button("AI-assisted review").click();
  await page.getByRole("combobox",{name:"Review camera",exact:true}).selectOption("1");
  await button("Get AI suggestions").click();
  await page.getByRole("group",{name:"Grip",exact:true}).getByLabel("Hands free").check();
  await button("Confirm answers").click();
  await page.getByRole("alert").filter({hasText:"evidence changed"}).waitFor();
  await page.getByRole("group",{name:"Grip",exact:true}).getByLabel("Hands free").check();
  await button("Confirm answers").click();
  await page.getByRole("status").filter({hasText:"Review saved"}).waitFor();
  assert.equal(requests.at(-1).camera, 1);
  assert.equal(requests.at(-1).answers.grip,"no_grasp");
  await button("Blind test").click();
  await page.getByText(/suggestions for this moment have already been revealed/).waitFor();
  assert.equal(await button("Save independent labels").count(),0);
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  assert.deepEqual(pageErrors, []);
  console.log("Passed blind labeling, reveal, source image, metrics, stale confirmation, correction, camera selection, and mobile checks.");
} finally {
  await browser?.close();
  await server.close();
}
