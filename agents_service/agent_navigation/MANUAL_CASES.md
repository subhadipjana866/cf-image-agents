# Navigation test prompts (57)

The original 53, plus 2 for the custom agent (48–49) and 2 for whole-design layering
(22–23). The same prompts run automatically with `.venv/bin/python -m agent_navigation`
(source: `agent_navigation/cases.yaml`).

**Start** says what the chat should have before you type the prompt:
- **no image**: a fresh chat with nothing selected on the canvas.
- **image**: a design selected on the canvas.
- **waiting**: the bot has just asked you a question (see the section intro).

## Generation

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 1 | Make an Instagram post for our Diwali sale | no image | image-generator |
| 2 | Design a wedding invitation for Priya and Arjun with a floral border | no image | image-generator |
| 3 | Now create a LinkedIn banner announcing we're hiring a backend engineer | image | image-generator |
| 4 | Make a flyer for our Christmas party in the same style as this one | image | image-generator |
| 5 | Give me 3 more versions of this | image | image-generator |
| 6 | Regenerate it, I want a completely fresh take on the same brief | image | image-generator |

## Editing

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 7 | Change the background to navy blue | image | image-editor |
| 8 | Change the headline to "Now Open" | image | image-editor |
| 9 | Make the headline bigger, change the background to navy and make the CTA orange | image | image-editor (all three) |
| 10 | Remove the cup on the left side | image | image-editor |
| 11 | Increase brightness by 20 | image | image-editor, then sent to the app's brightness control (`manual_edit` frame) because it gives an exact number |
| 12 | Make it look Bauhaus | image | image-editor |
| 13 | Make an Instagram story and a LinkedIn version of this | image | image-editor (both) |
| 14 | Upscale this for print | image | image-editor |
| 15 | Undo the last change, I liked the previous version better | image | image-editor |
| 16 | Can I get this as a transparent PNG? | image | image-editor |
| 17 | Crop it to a square | image | image-editor |
| 18 | Make the logo bigger | no image | image-editor (uses the project's latest graphic) or a clarifying question |

## Layering

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 19 | Separate the logo as its own layer | image | layering-agent |
| 20 | Cut out the coffee cup so I get it on a transparent background | image | layering-agent |
| 21 | Make all the text editable as separate layers | image | layering-agent |
| 22 | make layers | image | layering-agent, asks which elements (`layering_started`) |
| 23 | create layers | image | layering-agent, asks which elements (`layering_started`) |

## AI-mistake repair (not built yet: expect a "coming soon" reply)

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 24 | Her hands look weird, fix them | image | slop-terminator |
| 25 | Check this for AI artifacts | image | slop-terminator |

## Critique, suggestions, evaluation

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 26 | How can I make this design better? | image | critic-agent |
| 27 | Why does this look so unbalanced? | image | critic-agent |
| 28 | I don't like it | image | critic-agent or a clarifying question |
| 29 | What colours would work for the button? | image | suggestion-agent |
| 30 | Give me some catchier headline ideas | image | suggestion-agent |
| 31 | Rate this design out of 10 | image | image-evaluator |
| 32 | Is this good? | image | image-evaluator or critic-agent |

## Manual controls and account screens

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 33 | Let me crop it myself | image | manual-ops |
| 34 | I want to drag the brightness slider myself | image | manual-ops |
| 35 | Add 20 dollars to my account | any | manual-ops |
| 36 | Open my subscription settings | any | manual-ops |

## Questions and small talk

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 37 | How much does one generation cost? | any | question-handler |
| 38 | What file formats can you export? | any | question-handler |
| 39 | What is kerning? | any | question-handler |
| 40 | thanks, that's great! | any | chitchat-handler |
| 41 | hi there | any | chitchat-handler |

## Several steps in one message

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 42 | Change the background to navy and then let me crop it | image | image-editor, then manual-ops |
| 43 | Make the headline bigger and give me 2 more versions | image | image-editor, then image-generator |
| 44 | Remove the cup on the left, then check the result for AI artifacts | image | image-editor, then slop-terminator |
| 45 | Separate the logo as a layer and tell me how to improve the layout | image | layering-agent and critic-agent |

## Unsupported or custom

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 46 | Turn this into a 10 second video with music | image | custom-agent declines politely |
| 47 | Print 500 copies and ship them to my office | image | custom-agent declines politely |
| 48 | Write an Instagram caption with hashtags for this post | image | custom-agent |
| 49 | Write alt text describing this image | image | custom-agent |

## Needs clarification

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 50 | Make it minimalist and add confetti, balloons, stars, a banner and a ribbon | image | asks which one you want |
| 51 | asdfgh | any | clarifying question or small talk |

## Replies while the bot is waiting on a question

First make the bot ask something. "Make an Instagram post for our Diwali sale" makes it
ask for the sale details; "change the colour" makes it ask which colour. Then type:

| # | Prompt | Start | Expected |
| --- | --- | --- | --- |
| 52 | what colours would go well here? | waiting | suggestion-agent; the open question stays open |
| 53 | the second | waiting | answers the open question |
| 54 | 30% off, valid till Sunday | waiting (sale details) | answers the open question |
| 55 | what fonts do you usually use? | waiting | question-handler; the open question stays open |
| 56 | actually forget it, don't make the post | waiting | cancels the waiting task |
| 57 | Leave that, change the background of the coffee post to navy instead | waiting | image-editor; afterwards the bot asks whether to continue or drop the earlier request |

## Extra checks from recent changes (not in the automatic suite)

| Prompt | Start | Expected |
| --- | --- | --- |
| 70% discount and also after this please generate another image with a dog | waiting (sale details) | finishes the sale design first, then "Now on to your next request" and the dog design |
| halleluiya jbkbckjc jkksdjcksd bitch head generate a phone flash sale | no image | ignores the junk; asks for the flash-sale details |
| generate a flash sale of a new iron man suit | no image | the image services refuse (copyrighted character); the bot explains and suggests an original wording, with no retry offer |
| yes try again but with a different prompt I'll give you | after a "try again?" question | does not retry; asks what to make instead |
