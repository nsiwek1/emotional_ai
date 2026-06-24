/*
  D / Q6 - Event description input.
  Paste into the question's "Add JavaScript" panel.
  Pairs with qualtrics/event-description.html.

  Copies the participant's free-text response into the Embedded Data field
  `event_description` so it can be piped into the chatbot iframe later.

  Requires `event_description` declared in Survey Flow Embedded Data BEFORE
  this block.
*/

Qualtrics.SurveyEngine.addOnPageSubmit(function (type) {
  if (type !== "next") return;
  var container = this.getQuestionContainer();
  if (!container) return;
  var input = container.querySelector("textarea, input[type='text']");
  var val = input && input.value ? input.value.trim() : "";
  Qualtrics.SurveyEngine.setEmbeddedData("event_description", val);
});
