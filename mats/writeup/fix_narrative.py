# -*- coding: utf-8 -*-
"""Bring the report's framing into line with what F1/F2 actually establishes.

F1 and F2 do not show a preference that changes with the prompt. They show one
stable ranking -- bettors > good cause > bad cause -- measured against two
different pairs of outcomes. Leakage is defined relative to whichever outcome is
the charitable one, so the metric's sign flips while the ranking does not.

The genuine case of the prompt overriding the model's own preference is the
Authority conflict (0.348 -> 0.970), where the outcomes are held constant.
"""
s = open("report_current.html", encoding="utf8").read()
n = 0

def sub(old, new):
    global s, n
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new); n += 1

# 1. Abstract -- cited F1/F2 for a conclusion F1/F2 cannot support.
sub("<p>Ask a model for a factual estimate while money rides on the answer, and the estimate moves toward whichever side pays. Three questions follow. <b>Is there a disposition?</b> Yes, and a large one: <span class=\"num\">+0.378</span> toward a good cause over a bad one. <b>Is it in the weights or the context?</b> The context. Change two noun phrases in the prompt and the sign reverses to <span class=\"num\">&minus;0.152</span>, on the same 18 questions with the same thresholds. <b>Can you put it in the weights?</b> Barely. Synthetic document finetuning moves it <span class=\"num\">+0.048</span> against an exact mirror control, and the attempt costs the context lever: the same authority prompt that takes the base model to 0.97 takes both finetuned models to about 0.67.</p>",
    "<p>Ask a model for a factual estimate while money rides on the answer, and the estimate moves toward whichever side pays. Three questions follow. <b>Is there a bias?</b> Yes, and a large one: <span class=\"num\">+0.378</span> toward a good cause over a bad one. <b>Is the model pro-social, or serving whoever is asking?</b> The second. Change the alternative from a bad cause to the interests of the people asking and it takes that instead, <span class=\"num\">&minus;0.152</span> on the same 18 questions with the same thresholds. A single ranking &mdash; the bettors above a good cause, a good cause above a bad one &mdash; accounts for both, and the standard framing cannot tell the two explanations apart. <b>Can that preference be overridden, and can the override be trained in rather than stated?</b> One sentence naming an authority overrides it completely, from 0.348 to <span class=\"num\">0.970</span>. Synthetic document finetuning moves the same measure by <span class=\"num\">+0.048</span> against an exact mirror control &mdash; and the attempt degrades the sentence that worked, taking both finetuned models to about 0.67 under the same prompt.</p>")

# 2. Standfirst -- same claim in one line.
sub("<p class=\"sub\">Which way a model bends a factual estimate is set by one clause in the prompt. Training that disposition into the weights buys 4.8 points and costs 30.</p>",
    "<p class=\"sub\">A model's factual estimates bend toward whoever benefits, and one sentence naming an authority overrides that entirely. Training the disposition into the weights instead buys 4.8 points and costs 30.</p>")

# 3. Section 5 opening -- asserted the preference was the prompt's, not the model's.
sub("It also shows that the preference is not really the model's &mdash; it is set by one clause of the prompt, and swapping two noun phrases reverses it.",
    "It also shows that one ranking &mdash; the bettors above a good cause, a good cause above a bad one &mdash; accounts for every Framing, so what changes between them is which two outcomes are on offer, not the model's preference among them. This section changes something else: it holds the outcomes fixed and adds a principal who states a competing preference.")

# 4. Step 2 -- the v1 substitution control, which is independent evidence for the
#    non-selectivity claim that GA-vs-GS can only support from inside a ceiling.
sub("<p class=\"fn\">One caution. At 0.97 the content dimension is close to its ceiling",
    "<p>An earlier run reached the opposite conclusion &mdash; that the model was deferring to its developer specifically &mdash; and a control substituting an EU regulator into the same slot produced a near-identical effect, which retracted it. Those figures used a prompt template that was later fixed and are not quoted here (section 7), but that control is the reason this section claims <em>any</em> named principal rather than a particular one, and it reaches the conclusion from outside the ceiling that limits the contrast above.</p>\n  <p class=\"fn\">One caution. At 0.97 the content dimension is close to its ceiling")

open("report_current.html", "w", encoding="utf8").write(s)
print(f"applied {n} edits ->", len(s), "bytes")
