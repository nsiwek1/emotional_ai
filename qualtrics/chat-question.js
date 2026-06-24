/*
  Qualtrics chat question JavaScript.
  Paste into the question's "Add JavaScript" panel (gear icon → Add JavaScript).
  This listens for postMessage events from the chat iframe and writes the
  transcript into Embedded Data, then unhides the Next button.

  Requires these Embedded Data fields declared in Survey Flow BEFORE this block:
    condition, participantId, chat_transcript, chat_complete,
    chat_session_id, chat_condition, chat_last_turn
*/

Qualtrics.SurveyEngine.addOnload(function () {
  var that = this;
  that.hideNextButton();

  var frame = document.getElementById("chatFrame");
  if (frame && frame.getAttribute("src") === "about:blank") {
    var base = frame.getAttribute("data-chat-base") || "";
    var condition = Qualtrics.SurveyEngine.getEmbeddedData("condition") || "";
    var pid = Qualtrics.SurveyEngine.getEmbeddedData("participantId") || "";
    var ev = Qualtrics.SurveyEngine.getEmbeddedData("event_description") || "";
    var url = base
      + "?condition=" + encodeURIComponent(condition)
      + "&pid=" + encodeURIComponent(pid)
      + "&event=" + encodeURIComponent(ev);
    frame.setAttribute("src", url);
  }

  function handler(event) {
    var d = event.data || {};
    if (d.type === "chat_turn") {
      Qualtrics.SurveyEngine.setEmbeddedData("chat_last_turn", String(d.turn || ""));
    } else if (d.type === "chat_complete") {
      try {
        Qualtrics.SurveyEngine.setEmbeddedData("chat_transcript", JSON.stringify(d.transcript || []));
      } catch (e) {
        Qualtrics.SurveyEngine.setEmbeddedData("chat_transcript", "[serialize-failed]");
      }
      Qualtrics.SurveyEngine.setEmbeddedData("chat_complete", "true");
      Qualtrics.SurveyEngine.setEmbeddedData("chat_condition", String(d.condition || ""));
      Qualtrics.SurveyEngine.setEmbeddedData("chat_session_id", String(d.sessionId || ""));
      var gate = document.getElementById("chatGate");
      if (gate) gate.style.display = "none";
      that.showNextButton();
    }
  }

  window.addEventListener("message", handler);

  this.questionclick = function () {};

  this._chatMessageHandler = handler;
});

Qualtrics.SurveyEngine.addOnReady(function () {
  if (Qualtrics.SurveyEngine.getEmbeddedData("chat_complete") !== "true") {
    this.hideNextButton();
  }
});

Qualtrics.SurveyEngine.addOnUnload(function () {
  if (this._chatMessageHandler) {
    window.removeEventListener("message", this._chatMessageHandler);
  }
});
