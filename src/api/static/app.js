/* Shared by all four pages: theme, sign-in, and the only fetch that talks to the API.
 *
 * Every page had its own copy of the theme toggle before this. Authentication would
 * have been a fifth copy of the same twelve lines, and the fourth place to forget a
 * header. */

const IB = (() => {
  const TOKENS = "ib-tokens";   // { "<user id>": "<token>" }
  const ACTIVE = "ib-active";   // whose token the page is using

  /* The seeded personas, offered as one click each. The shared demo password used to
   * be pre-filled in the form and printed underneath it — which made signing in a
   * button press with no sign-in, and put a credential on the front page of a product
   * whose entire claim is that access is enforced. The password still works in the
   * form below for anyone who wants to type it; it is just not advertised. */
  const DEMO_PASSWORD = "demo";
  const DEMO = [
    { email: "alex.tan@aurelia.example", name: "Alex Tan", role: "support agent",
      sees: "support and all-staff material. Refused anything restricted." },
    { email: "marcus.lim@aurelia.example", name: "Marcus Lim", role: "compliance officer",
      sees: "restricted compliance material, and the three officer-only pages." },
  ];

  const read = (key, fallback) => {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (_) { return fallback; }
  };
  const write = (key, value) => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) {}
  };

  const tokens = () => read(TOKENS, {});
  const active = () => read(ACTIVE, null);
  const tokenFor = (id) => tokens()[String(id)] ?? null;
  const signedIn = () => Object.keys(tokens());

  function remember(id, token, { activate = true } = {}) {
    write(TOKENS, { ...tokens(), [String(id)]: token });
    if (activate) write(ACTIVE, String(id));
  }

  function forget(id) {
    const held = tokens();
    delete held[String(id)];
    write(TOKENS, held);
    // Clear the active slot rather than promoting whoever is left. This runs on a
    // 401, and silently continuing as a DIFFERENT person is the worst answer to an
    // expired token: the next question would be asked, and audited, as someone the
    // user did not choose. Switching identity is an explicit act (`activate`).
    if (String(active()) === String(id)) write(ACTIVE, null);
  }

  /* Switch to an identity already held. Explicit, always — `remember` keeps the
   * first one active when a second is added, so this is the only way to change who
   * a question is asked as. */
  function activate(id) {
    if (!tokenFor(id)) return false;
    write(ACTIVE, String(id));
    return true;
  }

  function forgetAll() {
    write(TOKENS, {});
    write(ACTIVE, null);
  }

  /* One door to the API. `as` picks whose token to send — the side-by-side
   * comparison is the only caller that ever passes it, because showing what two
   * people see means holding two people's tokens, not asking on their behalf. */
  async function api(path, { as = null, ...options } = {}) {
    const token = tokenFor(as ?? active());
    const headers = { ...(options.headers || {}) };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (options.body) headers["Content-Type"] = "application/json";
    const response = await fetch(path, { ...options, headers });
    if (response.status === 401) {
      // The token is gone, expired or was never good. Drop it rather than let every
      // later call fail the same way with no explanation.
      if (as ?? active()) forget(as ?? active());
      gate();
    }
    return response;
  }

  async function login(email, password) {
    const response = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!response.ok) throw new Error("That email and password do not match.");
    const { access_token } = await response.json();
    // Checked: an unchecked read here stored the token under the key "undefined"
    // when /auth/me failed, and every later call sent no Authorization at all.
    const who = await fetch("/auth/me", { headers: { Authorization: `Bearer ${access_token}` } });
    if (!who.ok) throw new Error("Signed in, but could not read the account. Try again.");
    const me = await who.json();
    remember(me.id, access_token);
    return me;
  }

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* The sign-in panel, shown in place of the page when nobody is signed in. */
  function gate({ force = false, keepActive = null } = {}) {
    // `active()`, not just `signedIn()`: a 401 clears the active slot while leaving
    // other tokens held, and without this the page would render signed-in-looking
    // chrome with no identity behind it.
    if (signedIn().length && active() && !force) return false;
    const main = document.querySelector("main");
    if (!main || main.dataset.gated) return true;
    main.dataset.gated = "1";
    main.innerHTML = `
      <h1>${keepActive ? "Add an identity" : "Sign in"}</h1>
      <p class="lede">${keepActive
        ? `You stay signed in as user ${esc(keepActive)}. Holding a second person's token
           is what lets the side-by-side comparison show what each of them sees — there
           is no way to ask on someone else's behalf.`
        : `Every question and every report is answered for the person asking. There is no
           way in without a token, and the token says only who you are — what you may see
           is read from the database on each request.`}</p>
      <div class="personas">
        <span class="section">Sign in as</span>
        ${DEMO.map((who) => `
          <button class="demo-id" type="button" data-email="${esc(who.email)}">
            <span class="name">${esc(who.name)}</span>
            <span class="role">${esc(who.role)}</span>
            <span class="says">${esc(who.sees)}</span>
          </button>`).join("")}
      </div>

      <details class="own"${keepActive ? " open" : ""}>
        <summary>Sign in with an email and password</summary>
        <form class="panel ask" id="ib-signin" style="max-width:420px">
          <label class="label" for="ib-email">Email</label>
          <input id="ib-email" type="email" autocomplete="username" placeholder="you@aurelia.example">
          <label class="label" for="ib-password" style="margin-top:10px">Password</label>
          <input id="ib-password" type="password" autocomplete="current-password">
          <div class="controls" style="margin-top:14px">
            <button type="submit">Sign in</button>
          </div>
        </form>
      </details>
      <p id="ib-error" class="signin-error" role="alert"></p>

      <div class="guide">
        <span class="section">What the four pages are</span>
        <dl>
          <dt>Ask</dt>
          <dd>Ask a question across Confluence, Jira, Slack and Drive. You get an answer
            you are allowed to see, or one refusal. <b>Anyone.</b></dd>
          <dt>Gaps</dt>
          <dd>Where the permission model is wrong. One half is documents people wanted and
            could not see — fix the grant. The other is questions nothing answers — write
            the page. <b>Compliance officer.</b></dd>
          <dt>Audit</dt>
          <dd>Every request, hash-chained. The only place the real reason for a refusal can
            be read back, and reading it is itself recorded. <b>Compliance officer.</b></dd>
          <dt>Sources</dt>
          <dd>What is connected, how fresh it is, and who currently may read what. Revoke a
            grant and the next question reflects it. <b>Compliance officer.</b></dd>
        </dl>
        <p>Three of the four are officer-only, so signing in as the support agent and
          clicking around will get you refused — that is the product working, not a fault.
          Hold both identities to see each side.</p>
      </div>`;
    const enter = async (email, password) => {
      try {
        await login(email, password);
        // The first identity stays the active one: adding somebody must not quietly
        // change who the next question is asked as.
        if (keepActive) write(ACTIVE, String(keepActive));
        location.reload();
      } catch (error) {
        document.getElementById("ib-error").textContent = error.message;
      }
    };
    main.querySelector(".personas").addEventListener("click", (e) => {
      const pick = e.target.closest("[data-email]");
      if (pick) enter(pick.dataset.email, DEMO_PASSWORD);
    });
    document.getElementById("ib-signin").addEventListener("submit", (e) => {
      e.preventDefault();
      enter(document.getElementById("ib-email").value,
            document.getElementById("ib-password").value);
    });
    return true;
  }

  /* Who is signed in, who else is held, and a way out — in the app bar.
   *
   * A switcher, not a label. Holding two identities and being unable to become the
   * second one was a dead end you could walk into by following the instructions: the
   * officer-only pages say "add that identity", and adding it left you refused,
   * because `remember` deliberately keeps the first one active. Adding somebody must
   * not silently change who the next question is asked as — but there has to be a
   * way to say so on purpose, and this is it. */
  async function identity() {
    const bar = document.querySelector(".bar .spacer");
    if (!bar || !signedIn().length) return null;
    const response = await api("/auth/me");
    if (!response.ok) return null;
    const me = await response.json();

    // Every held identity, so switching is one click and the roster is visible.
    // Roles come from each token's own /auth/me — the server decides what a caller
    // is, never the page (CLAUDE.md §4a).
    const others = signedIn().filter((id) => String(id) !== String(me.id));
    const roles = await Promise.all(others.map(async (id) => {
      const r = await api("/auth/me", { as: id });
      return r.ok ? { id, ...(await r.json()) } : null;
    }));

    const chip = document.createElement("div");
    chip.className = "identity";
    chip.innerHTML = `
      <button class="persona on" type="button" disabled
        title="You are asking as this person">
        <span class="who">${esc(me.id)}</span><span class="role">${esc(me.role)}</span>
      </button>
      ${roles.filter(Boolean).map((who) => `
        <button class="persona" type="button" data-be="${esc(who.id)}"
          title="Ask as ${esc(who.role)} user ${esc(who.id)} instead">
          <span class="who">${esc(who.id)}</span><span class="role">${esc(who.role)}</span>
        </button>`).join("")}
      <button class="copy" type="button" id="ib-add">Add identity</button>
      <button class="copy" type="button" id="ib-out">Sign out${
        signedIn().length > 1 ? " all" : ""}</button>`;
    bar.after(chip);

    // Say which pages will refuse you before you click them.
    if (me.role !== "compliance") {
      for (const link of document.querySelectorAll("nav a[data-officer]")) {
        link.classList.add("locked");
        link.title = "Compliance officers only — switch to that identity in the header";
        // A tooltip is not read to a keyboard or a screen reader, and the lock is only a picture.
        link.insertAdjacentHTML("beforeend", '<span class="sr"> — compliance officers only</span>');
      }
    }
    chip.addEventListener("click", (e) => {
      const swap = e.target.closest("[data-be]");
      if (!swap) return;
      activate(swap.dataset.be);
      location.reload();
    });
    // Everything, not just the active one. It used to forget the active identity and
    // promote whoever was left, so "Sign out" left you signed in as someone else.
    document.getElementById("ib-out").addEventListener("click", () => {
      forgetAll();
      location.reload();
    });
    document.getElementById("ib-add").addEventListener("click", addIdentity);
    return me;
  }

  /* Sign in as a SECOND person without giving up the first. The side-by-side
   * comparison needs two tokens, and there is no honest shortcut: seeing what
   * someone else sees means being able to authenticate as them. Uses the same form
   * as the gate rather than window.prompt, which some embedded views refuse outright
   * and which cannot show an error next to the field that caused it. */
  function addIdentity() {
    gate({ force: true, keepActive: active() });
  }

  function theme() {
    try {
      const saved = localStorage.getItem("ib-theme");
      if (saved) document.documentElement.dataset.theme = saved;
    } catch (_) {}
    const button = document.getElementById("theme");
    if (!button) return;
    button.addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme
        ? root.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("ib-theme", root.dataset.theme); } catch (_) {}
    });
  }

  /* Returns false when the page should stop and let the gate have the screen. */
  function start() {
    theme();
    if (gate()) return false;
    identity();
    return true;
  }

  return { api, login, remember, forget, forgetAll, activate, tokenFor,
           signedIn, active, addIdentity, start, esc };
})();
