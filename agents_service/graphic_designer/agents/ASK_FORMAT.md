# Questions asked inside a flow: `kognie.ask.v1`

When the bot asks the user something in the middle of a task, the websocket frame
carries an `ask` object with a fixed structure. The frontend draws buttons, checkboxes
or inputs from it instead of reading the question text. This covers:

- questions an agent asks mid-task (for example "Which cup, left or right?");
- the bot's own questions: continue or drop a waiting request, try again after a
  failure, more suggestions, the layering offer, and "did you mean …" clarifications.

These keep their existing frames and are **not** affected: the input collector's details
form (`fields` + `skippable`), the image picker (`kind: "reselect_image"`), the suggestion
cards (`suggestions`), and the layering question (`layering_started`).

## The frame

The frame type stays `text_input`, so a frontend that doesn't know `ask` still shows the
question. The options are **no longer** written into `question`; they are only in `ask`.

```json
{
  "type": "text_input",
  "question": "You still have an unfinished request: \"Create a flash sale post…\". Do you want to continue it or drop it?",
  "current_image_url": "https://…/image.webp",
  "sender": "orchestrator",
  "task_id": "3.auto2",
  "recipient": "user",
  "messages": [
    {"sender": "custom-agent", "task_id": "3.T1", "recipient": "user", "text": "Fair enough — …"},
    {"sender": "orchestrator", "task_id": "3.auto2", "recipient": "user", "text": "You still have an unfinished request: …"}
  ],
  "cost": 0.0,
  "show_cost": false,
  "ask": {
    "schema": "kognie.ask.v1",
    "id": "q7",
    "task_id": "3.auto2",
    "asked_by": "orchestrator",
    "purpose": "resume",
    "kind": "single_choice",
    "prompt": "You still have an unfinished request: \"Create a flash sale post…\". Do you want to continue it or drop it?",
    "options": [
      {"id": "continue", "label": "Continue", "value": "Continue"},
      {"id": "drop_it", "label": "Drop it", "value": "Drop it"}
    ],
    "fields": [],
    "allow_free_text": true,
    "allow_skip": false,
    "default_option_id": null
  }
}
```

`question` can hold more than the prompt: notes that came just before the question are
folded in ahead of it (they are listed one by one in `messages`). Show `question` (or
`messages`) as the chat bubble, and use `ask` to draw the controls.

## `ask` fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Always `"kognie.ask.v1"`. |
| `id` | string | The question's id. Send it back in `ask_reply.id`. |
| `task_id` | string or null | The task the question belongs to (null for a clarification). |
| `asked_by` | string | Who asked: an agent id (`image-editor`, `image-generator`, `critic-agent`, `layering-agent`, `custom-agent`, …) or `orchestrator`. |
| `purpose` | string | Why it was asked: `agent_question`, `resume`, `retry`, `more_suggestions`, `layering_offer`, `clarification`. |
| `kind` | string | How to draw it (see below). |
| `prompt` | string | The question itself. |
| `options` | array | Choices: `{"id", "label", "value"}`. Empty when there are none. |
| `fields` | array | Inputs to fill: see below. Empty unless `kind` is `form`. |
| `allow_free_text` | bool | Whether to keep the normal text box open so the user can type their own answer. |
| `allow_skip` | bool | Whether to show a Skip control. |
| `default_option_id` | string or null | An option to pre-select, if any. |

### `kind`

| kind | Draw it as | Reply with |
| --- | --- | --- |
| `confirm` | Two buttons, Yes and No (`options` holds exactly those two). | `option_ids: ["yes"]` or `["no"]` |
| `single_choice` | One button per option; one click answers. | `option_ids: [<one id>]` |
| `multi_choice` | Checkboxes plus a Send button. | `option_ids: [<ids>]` |
| `text` | No controls; the normal text box. | just `response` (or `values` empty) |
| `form` | One input per entry in `fields`, plus any `options` as buttons, and a Send button. | `values: {<field id>: <value>}` (and `option_ids` if options were shown) |

### Entries in `fields`

```json
{"id": "headline", "label": "Headline", "type": "text", "placeholder": "", "required": true, "default": null, "options": []}
```

`type` is one of `text`, `textarea`, `number`, `color`, `url`, `date`, `select`. A `select`
has its choices in `options` (`{"id", "label", "value"}`).

## Replying

Send the usual message frame, with an `ask_reply` object added:

```json
{
  "user_id": "14",
  "response": "",
  "current_image_url": "https://…/image.webp",
  "ask_reply": {
    "id": "q7",
    "option_ids": ["continue"],
    "values": {},
    "skipped": false
  }
}
```

- `id`: the `ask.id` being answered. Required.
- `option_ids`: the chosen option ids (empty if none).
- `values`: field id to value, for `form` questions.
- `skipped`: `true` when the user pressed Skip.
- `response`: anything the user also typed. It is added to the answer.

The reply goes straight to the question it answers, with no interpretation step, so a
button click is always understood exactly. If the user ignores the buttons and types
something else, send it as a normal message without `ask_reply`; the bot works out
whether it answers the question or is a new request, as it does today.

A reply whose `id` is not the open question (stale buttons) is treated as an ordinary
message.

## Examples

Agent asks a choice mid-task (`asked_by: "image-editor"`, `purpose: "agent_question"`):

```json
"ask": {"schema": "kognie.ask.v1", "id": "q2", "task_id": "1.T1", "asked_by": "image-editor",
        "purpose": "agent_question", "kind": "single_choice", "prompt": "Which cup, left or right?",
        "options": [{"id": "left", "label": "Left", "value": "Left"},
                    {"id": "right", "label": "Right", "value": "Right"}],
        "fields": [], "allow_free_text": true, "allow_skip": false, "default_option_id": null}
```

Try again after a failure (`kind: "confirm"`):

```json
"ask": {"schema": "kognie.ask.v1", "id": "q5", "task_id": "2.T1", "asked_by": "orchestrator",
        "purpose": "retry", "kind": "confirm", "prompt": "That didn't go through. Want me to try again?",
        "options": [{"id": "yes", "label": "Yes", "value": "Yes"}, {"id": "no", "label": "No", "value": "No"}],
        "fields": [], "allow_free_text": true, "allow_skip": false, "default_option_id": null}
```

Agent needs typed values (`kind: "form"`):

```json
"ask": {"schema": "kognie.ask.v1", "id": "q9", "task_id": "4.T1", "asked_by": "image-editor",
        "purpose": "agent_question", "kind": "form", "prompt": "What should the new headline and its colour be?",
        "options": [],
        "fields": [{"id": "headline", "label": "Headline", "type": "text", "placeholder": "", "required": true, "default": null, "options": []},
                   {"id": "colour", "label": "Colour", "type": "color", "placeholder": "", "required": false, "default": null, "options": []}],
        "allow_free_text": true, "allow_skip": false, "default_option_id": null}
```

Reply to it:

```json
"ask_reply": {"id": "q9", "option_ids": [], "values": {"headline": "Big Diwali Sale", "colour": "#FFD700"}, "skipped": false}
```
