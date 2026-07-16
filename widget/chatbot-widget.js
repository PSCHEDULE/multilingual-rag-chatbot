/**
 * Multilingual RAG Chatbot embed widget (M6).
 * Modes: floating (default) | inline
 * Uses POST + fetch streaming (SSE contract: docs/sse-contract.md)
 */
(function (global) {
  "use strict";

  var STYLE_ID = "mrc-widget-styles";

  function css() {
    return [
      ".mrc-root{font-family:system-ui,-apple-system,sans-serif;font-size:14px;color:#0f172a;}",
      ".mrc-fab{position:fixed;right:20px;bottom:20px;z-index:99999;width:56px;height:56px;border-radius:50%;",
      "border:none;background:#2563eb;color:#fff;font-size:22px;cursor:pointer;box-shadow:0 8px 24px rgba(37,99,235,.35);}",
      ".mrc-panel{position:fixed;right:20px;bottom:88px;z-index:99999;width:360px;max-width:calc(100vw - 24px);",
      "height:520px;max-height:calc(100vh - 120px);background:#fff;border:1px solid #e2e8f0;border-radius:16px;",
      "box-shadow:0 16px 48px rgba(15,23,42,.18);display:flex;flex-direction:column;overflow:hidden;}",
      ".mrc-panel.mrc-inline{position:relative;right:auto;bottom:auto;width:100%;height:480px;box-shadow:none;}",
      ".mrc-header{padding:12px 14px;background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;font-weight:600;}",
      ".mrc-sub{font-weight:400;font-size:12px;opacity:.9;margin-top:2px;}",
      ".mrc-msgs{flex:1;overflow:auto;padding:12px;background:#f8fafc;}",
      ".mrc-msg{margin:0 0 10px;padding:10px 12px;border-radius:12px;max-width:90%;line-height:1.45;white-space:pre-wrap;}",
      ".mrc-msg.user{background:#dbeafe;margin-left:auto;}",
      ".mrc-msg.bot{background:#fff;border:1px solid #e2e8f0;}",
      ".mrc-sources{font-size:12px;color:#475569;margin-top:8px;}",
      ".mrc-sources summary{cursor:pointer;}",
      ".mrc-inputrow{display:flex;gap:8px;padding:10px;border-top:1px solid #e2e8f0;background:#fff;}",
      ".mrc-inputrow input{flex:1;border:1px solid #cbd5e1;border-radius:10px;padding:10px 12px;}",
      ".mrc-inputrow button{border:none;background:#2563eb;color:#fff;border-radius:10px;padding:0 14px;cursor:pointer;}",
      ".mrc-inputrow button:disabled{opacity:.5;cursor:not-allowed;}",
      ".mrc-md strong{font-weight:700;}",
      ".mrc-md code{background:#f1f5f9;padding:1px 4px;border-radius:4px;font-size:12px;}",
    ].join("");
  }

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = css();
    document.head.appendChild(s);
  }

  function escapeHtml(t) {
    return String(t)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Minimal markdown: **bold**, `code`, newlines */
  function renderMarkdown(text) {
    var html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function parseSseChunk(buffer, onEvent) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop() || "";
    for (var i = 0; i < parts.length; i++) {
      var block = parts[i];
      if (!block.trim()) continue;
      var eventName = "message";
      var dataLines = [];
      var lines = block.split("\n");
      for (var j = 0; j < lines.length; j++) {
        var line = lines[j];
        if (line.indexOf("event:") === 0) eventName = line.slice(6).trim();
        else if (line.indexOf("data:") === 0) dataLines.push(line.slice(5).trim());
      }
      var dataRaw = dataLines.join("\n");
      var data = {};
      try {
        data = dataRaw ? JSON.parse(dataRaw) : {};
      } catch (e) {
        data = { text: dataRaw };
      }
      onEvent(eventName, data);
    }
    return rest;
  }

  function createUI(opts) {
    ensureStyles();
    var root = document.createElement("div");
    root.className = "mrc-root";

    var panel = document.createElement("div");
    panel.className = "mrc-panel" + (opts.mode === "inline" ? " mrc-inline" : "");
    if (opts.mode !== "inline") panel.style.display = "none";

    panel.innerHTML =
      '<div class="mrc-header">Support Chat<div class="mrc-sub">Multilingual RAG · <span class="mrc-lang">auto</span></div></div>' +
      '<div class="mrc-msgs"></div>' +
      '<div class="mrc-inputrow"><input type="text" placeholder="Ask a question..." /><button type="button">Send</button></div>';

    root.appendChild(panel);

    var fab = null;
    if (opts.mode !== "inline") {
      fab = document.createElement("button");
      fab.className = "mrc-fab";
      fab.type = "button";
      fab.setAttribute("aria-label", "Open chat");
      fab.textContent = "💬";
      fab.addEventListener("click", function () {
        panel.style.display = panel.style.display === "none" ? "flex" : "none";
      });
      root.appendChild(fab);
    } else {
      panel.style.display = "flex";
    }

    var mount = opts.mountEl || document.body;
    mount.appendChild(root);
    return {
      root: root,
      panel: panel,
      msgs: panel.querySelector(".mrc-msgs"),
      input: panel.querySelector("input"),
      button: panel.querySelector("button"),
      langEl: panel.querySelector(".mrc-lang"),
    };
  }

  function MultilingualChatbot(options) {
    this.opts = options || {};
    this.apiBase = (this.opts.apiBase || "").replace(/\/$/, "");
    this.mode = this.opts.mode || "floating";
    this.locale = this.opts.locale || null;
    this.sessionId = this.opts.sessionId || null;
    this.ui = null;
  }

  MultilingualChatbot.prototype.mount = function (selector) {
    var el = selector ? document.querySelector(selector) : null;
    this.ui = createUI({ mode: this.mode, mountEl: el || document.body });
    var self = this;
    this.ui.button.addEventListener("click", function () {
      self.send();
    });
    this.ui.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") self.send();
    });
    return this;
  };

  MultilingualChatbot.prototype.addMessage = function (role, text, sources) {
    var div = document.createElement("div");
    div.className = "mrc-msg " + (role === "user" ? "user" : "bot");
    if (role === "user") {
      div.textContent = text;
    } else {
      div.classList.add("mrc-md");
      div.innerHTML = renderMarkdown(text || "");
      if (sources && sources.length) {
        var det = document.createElement("details");
        det.className = "mrc-sources";
        var sum = document.createElement("summary");
        sum.textContent = "Sources (" + sources.length + ")";
        det.appendChild(sum);
        var ul = document.createElement("ul");
        sources.forEach(function (s) {
          var li = document.createElement("li");
          li.textContent =
            (s.title || s.source || "doc") +
            (s.score != null ? " · " + Number(s.score).toFixed(3) : "");
          ul.appendChild(li);
        });
        det.appendChild(ul);
        div.appendChild(det);
      }
    }
    this.ui.msgs.appendChild(div);
    this.ui.msgs.scrollTop = this.ui.msgs.scrollHeight;
    return div;
  };

  MultilingualChatbot.prototype.send = async function () {
    var text = (this.ui.input.value || "").trim();
    if (!text) return;
    this.ui.input.value = "";
    this.ui.button.disabled = true;
    this.addMessage("user", text);
    var botEl = this.addMessage("bot", "…");
    var acc = "";
    var sources = [];
    var self = this;

    try {
      var resp = await fetch(this.apiBase + "/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message: text,
          session_id: this.sessionId,
          language: this.locale,
        }),
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        buffer = parseSseChunk(buffer, function (eventName, data) {
          if (eventName === "meta") {
            if (data.session_id) self.sessionId = data.session_id;
            if (data.language && self.ui.langEl) self.ui.langEl.textContent = data.language;
          } else if (eventName === "token") {
            acc += data.text || "";
            botEl.innerHTML = renderMarkdown(acc);
            botEl.classList.add("mrc-md");
          } else if (eventName === "sources") {
            sources = (data.items || []).slice();
          } else if (eventName === "done") {
            if (data.session_id) self.sessionId = data.session_id;
            if (data.answer && !acc) acc = data.answer;
          } else if (eventName === "error") {
            acc += "\n[error] " + (data.message || "unknown");
          }
        });
      }
      botEl.innerHTML = renderMarkdown(acc || "(no response)");
      botEl.classList.add("mrc-md");
      if (sources.length) {
        // re-render with sources
        var parent = botEl.parentNode;
        parent.removeChild(botEl);
        self.addMessage("bot", acc || "(no response)", sources);
      }
    } catch (err) {
      botEl.textContent = "Failed to reach API: " + (err && err.message ? err.message : err);
    } finally {
      this.ui.button.disabled = false;
      this.ui.msgs.scrollTop = this.ui.msgs.scrollHeight;
    }
  };

  MultilingualChatbot.init = function (options) {
    var w = new MultilingualChatbot(options || {});
    var mount = (options && options.mount) || null;
    w.mount(mount);
    return w;
  };

  // Auto-init from script data attributes
  function autoInit() {
    var scripts = document.getElementsByTagName("script");
    var me = scripts[scripts.length - 1];
    if (!me || !me.getAttribute) return;
    if (me.getAttribute("data-mrc-autoload") === "false") return;
    var api = me.getAttribute("data-api-base");
    if (!api && me.src) {
      try {
        api = new URL(me.src).origin;
      } catch (e) {
        api = "";
      }
    }
    if (me.getAttribute("data-api-base") != null || me.getAttribute("data-mode")) {
      MultilingualChatbot.init({
        apiBase: api || "",
        mode: me.getAttribute("data-mode") || "floating",
        locale: me.getAttribute("data-locale") || null,
      });
    }
  }

  global.MultilingualChatbot = MultilingualChatbot;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoInit);
  } else {
    autoInit();
  }
})(typeof window !== "undefined" ? window : globalThis);
