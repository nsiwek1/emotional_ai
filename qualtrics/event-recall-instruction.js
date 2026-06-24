/*
  D / Q5 - Event recall instruction.
  Paste into the question's "Add JavaScript" panel (gear icon -> Add JavaScript).
  Pairs with qualtrics/event-recall-instruction.html.
  Hides Next for 15 seconds, then reveals it.
*/

Qualtrics.SurveyEngine.addOnload(function () {
  var that = this;
  that.hideNextButton();

  var remaining = 15;
  var span = document.getElementById("recallCountdownSeconds");
  var line = document.getElementById("recallCountdown");

  var tick = function () {
    remaining -= 1;
    if (span) span.textContent = String(Math.max(remaining, 0));
    if (remaining <= 0) {
      clearInterval(that._recallTimer);
      if (line) line.style.display = "none";
      that.showNextButton();
    }
  };

  that._recallTimer = setInterval(tick, 1000);
});

Qualtrics.SurveyEngine.addOnUnload(function () {
  if (this._recallTimer) clearInterval(this._recallTimer);
});
