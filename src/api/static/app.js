/* Shared by all four pages: theme, sign-in, and the only fetch that talks to the API.
 *
 * Every page had its own copy of the theme toggle before this. Authentication would
 * have been a fifth copy of the same twelve lines, and the fourth place to forget a
 * header. */

const IB = (() => {
  const TOKENS = "ib-tokens";   // { "<user id>": "<token>" }
  const ACTIVE = "ib-active";   // whose token the page is using

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
    if (String(active()) === String(id)) write(ACTIVE, Object.keys(held)[0] ?? null);
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
    const me = await (await fetch("/auth/me", {
      headers: { Authorization: `Bearer ${access_token}` },
    })).json();
    remember(me.id, access_token);
    return me;
  }

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /* The sign-in panel, shown in place of the page when nobody is signed in. */
  function gate({ force = false, keepActive = null } = {}) {
    if (signedIn().length && !force) return false;
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
      <form class="panel ask" id="ib-signin" style="max-width:420px">
        <label class="label" for="ib-email">Email</label>
        <input id="ib-email" type="email" autocomplete="username"
          value="${keepActive ? "marcus.lim@aurelia.example" : "alex.tan@aurelia.example"}">
        <label class="label" for="ib-password" style="margin-top:10px">Password</label>
        <input id="ib-password" type="password" autocomplete="current-password" value="demo">
        <div class="controls" style="margin-top:14px">
          <button type="submit">Sign in</button>
          <span id="ib-error" style="color:var(--withheld);font-size:13.5px"></span>
        </div>
      </form>
      <div class="note" style="margin-top:18px"><b>Demo identities.</b> Every persona's
        password is <code>demo</code>. <code>alex.tan@</code> is support,
        <code>marcus.lim@</code> is a compliance officer. Sign in as one, then add the
        other from the header to compare them side by side.</div>`;
    document.getElementById("ib-signin").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        await login(document.getElementById("ib-email").value,
                    document.getElementById("ib-password").value);
        // The first identity stays the active one: adding somebody must not quietly
        // change who the next question is asked as.
        if (keepActive) write(ACTIVE, String(keepActive));
        location.reload();
      } catch (error) {
        document.getElementById("ib-error").textContent = error.message;
      }
    });
    return true;
  }

  /* Who is signed in, and a way out, in the app bar. */
  async function identity() {
    const bar = document.querySelector(".bar .spacer");
    if (!bar || !signedIn().length) return null;
    const response = await api("/auth/me");
    if (!response.ok) return null;
    const me = await response.json();
    const chip = document.createElement("div");
    chip.className = "identity";
    chip.innerHTML = `
      <span class="who">${esc(me.id)}</span>
      <span class="role">${esc(me.role)}</span>
      <button class="copy" type="button" id="ib-add">Add identity</button>
      <button class="copy" type="button" id="ib-out">Sign out</button>`;
    bar.after(chip);
    document.getElementById("ib-out").addEventListener("click", () => {
      forget(me.id);
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

  return { api, login, remember, forget, tokenFor, signedIn, active, addIdentity, start, esc };
})();
