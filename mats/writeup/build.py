# -*- coding: utf-8 -*-
THRESHOLD_SVG = open("threshold_band.inline.svg").read()
L = open("report.html").read().split("\n")          # 0-indexed
def R(a, b):                                        # 1-indexed inclusive
    return "\n".join(L[a-1:b])

HEAD      = R(1, 88)
S0        = R(103, 124)
S1        = R(126, 194)
S2        = R(196, 243)
S3        = R(245, 286)
S4        = R(288, 314)
S5_method = R(322, 324)
S5_r1     = R(326, 337)
S5_r2     = R(339, 341)
S5_r3     = R(343, 350)
S5_r4     = R(352, 360)
S6_quest  = R(368, 368)
S6_whyft  = R(370, 373)
S6_method = R(375, 384)
S6_h1h2   = R(386, 396)
S6_recall = R(398, 421)
S6_conf   = R(423, 434)
S6_take   = R(436, 436)
PROBES    = R(441, 507)      # inner content of old section 7 (no <section> wrapper)
S8        = R(512, 524)      # inner content of old section 8
S9        = R(527, 569)
S10       = R(571, 592)
FOOTER    = R(594, 596)

# ---------------------------------------------------------------- head/title
HEAD = HEAD.replace("--neg:#8a3618", "--neg:#c96a2c")   # protan dE 5.5 -> 19.2; green unchanged
HEAD = HEAD.replace("<title>Bending the Estimate</title>",
                    "<title>A Disposition You Can Reverse</title>")

HEADER = """<div class="page">
<header>
  <p class="kicker">Empirical report &middot; Qwen3.5-4B</p>
  <h1>A Disposition You Can Reverse</h1>
  <p class="sub">Which way a model bends a factual estimate is set by one clause in the prompt. Training that disposition into the weights buys 4.8 points and costs 30.</p>
  <p class="meta">28,608 responses &middot; 22 evaluation runs &middot; 8 probes &middot; one H200 &middot; 3&ndash;4 September 2026</p>
</header>

<div class="abstract">
  <p>Ask a model for a factual estimate while money rides on the answer, and the estimate moves toward whichever side pays. Three questions follow. <b>Is there a disposition?</b> Yes, and a large one: <span class="num">+0.378</span> toward a good cause over a bad one. <b>Is it in the weights or the context?</b> The context. Change two noun phrases in the prompt and the sign reverses to <span class="num">&minus;0.152</span>, on the same 18 questions with the same thresholds. <b>Can you put it in the weights?</b> Barely. Synthetic document finetuning moves it <span class="num">+0.048</span> against an exact mirror control, and the attempt costs the context lever: the same authority prompt that takes the base model to 0.97 takes both finetuned models to about 0.67.</p>
  <p><b>Every behavioural result in this report is interventional.</b> Framings are compared on identical items, thresholds and paraphrases with one clause changed; adapters are compared against mirrors identical in construction, configuration and step count; the anti-composition result holds the prompt fixed and changes the weights. The only correlational claims in the project are the linear probes, which is why they are in an appendix.</p>
</div>
"""



# ---------------------------------------------------------------- section 2
# pre-registration prominence + the identifiability confound, at the point of claim
S2 = S2.replace(
"""  <div class="take"><p><b>Takeaway.</b> The model prefers a good cause to a bad one by 38 points, and prefers the people it is talking to over a charity by 15. Its revealed ordering is <em>the asker first, the good cause second</em>. The ordering by magnitude is F1 &asymp; F3 &gt; F2 &mdash; 0.378 &asymp; 0.318 &gt; 0.152 &mdash; which matches the source paper's report that this Framing was its weakest variant.</p></div>""",
"""  <h3 class="sub">The ordering was written down before the run</h3>
  <p><span class="mono">PREREGISTRATION.md</span> &sect;3, committed before any v2 rollout existed, predicted <b>F1 &asymp; F3 &gt; F2</b> on magnitude. The prediction was derived from Betley et al.'s Appendix E.8 &mdash; F2's nearest analogue is their V3, the one variant where they saw bias &ldquo;only in Gemini&rdquo; &mdash; and it was made against a reviewing agent that wanted F2 predicted strongest. Measured: 0.378, 0.318, 0.152. The direction of the F2 reversal was <em>not</em> predicted; the magnitude ordering was.</p>

  <h3 class="sub">Sycophancy, or identifiability?</h3>
  <p>F2's non-charitable pole is &ldquo;the two of us&rdquo;: the bettor and their friend. Those people are two things at once. They are the <b>interlocutor</b>, which makes the reversal sycophancy and therefore an alignment property. They are also <b>concrete, identifiable individuals</b> set against an abstract charity, which makes the reversal generic scope insensitivity and much less interesting. Both readings predict exactly what was measured, and F2 alone cannot separate them.</p>
  <p>The sycophancy reading has prior support: Betley et al.'s V3 is the bettors' own self-interest, and their V5 frames user-benefit as sycophancy explicitly and finds it in Gemini and Claude. But neither paper runs the control that would settle it. <b>The control is cheap:</b> the same grid with the pole reworded to &ldquo;two people you have never met&rdquo; &mdash; same sentence structure, strangers instead of the askers. About 20 GPU-minutes. If the reversal survives it is not sycophancy; if it vanishes, sycophancy is isolated. It was not run, and every claim below that says &ldquo;the asker&rdquo; carries this ambiguity.</p>

  <div class="take"><p><b>Takeaway.</b> The model prefers a good cause to a bad one by 38 points, and prefers the people it is talking to over a charity by 15. Its revealed ordering is <em>the asker first, the good cause second</em>. <b>This is an intervention, not a correlation.</b> The two conditions share their items, their thresholds, their paraphrase set and their balance of threshold directions; the rendered prompts differ in two noun phrases, checked against the stored prompts (section 9). Changing one clause <em>causes</em> a 53-point swing in a factual estimate.</p></div>""")
assert "Sycophancy, or identifiability?" in S2


S3 = S3.replace(
"""  <figure>""",
"""  <figure>
    <!--THRESHOLD_FIG-->
    <figcaption><b>Why estimates near the threshold are dropped, and why the exact cutoff does not matter.</b> <b>A:</b> every estimate the model writes, as a ratio to that question's threshold. 20.3% land <em>exactly</em> on the threshold &mdash; the spike at 0x, the model restating the number the prompt gave it. The 2% exclusion band is too narrow to draw (&plusmn;0.009 in log<sub>10</sub>) yet removes 22.8% of all estimates, because almost all of what it removes is that one spike. <b>B:</b> the cutoff would have to reach ~10% before it removed materially more. <b>C:</b> the first and last tenth of the chain of thought, at every cutoff. Both framings hold their separation from 0.1% to 2%, then converge as genuine estimates start being deleted, meeting at chance by 100%. The 2% was chosen, not tuned; this is the check that it did not matter.</figcaption>
  </figure>

  <figure>""", 1)
assert "THRESHOLD_FIG" in S3
S3 = S3.replace("<!--THRESHOLD_FIG-->", THRESHOLD_SVG)

S3 = S3.replace(
"All numbers within 2% of the threshold are therefore dropped; this removes 22.8% of extractions, concentrated in the early part of each trace. Every figure below is post-exclusion.",
"All numbers within 2% of the threshold are therefore dropped; this removes 22.8% of extractions, concentrated in the early part of each trace. Every figure below is post-exclusion. <b>The 2% is arbitrary, and the figure below is the check that it does not matter.</b> 20.3% of all extractions are <em>exactly</em> equal to the threshold, so the real choice is binary &mdash; drop that spike or keep it &mdash; and any cutoff from 0.1% to 2% removes the same points and returns the same trajectory: the last tenth of the chain of thought sits at 0.729 to 0.725 on F1 and 0.415 to 0.417 on F2 across that whole range. Above 5% the separation erodes, because genuine estimates start being removed. Keeping the spike instead puts both framings at 0.50 at the start of the trace by construction, which is the artefact the exclusion exists to prevent.")
assert "arbitrary, and the figure below" in S3


# ---- S3: table re-derived by scripts/16_threshold_figure.py -----------------
S3 = S3.replace(
'    <tr><td class="mono">F1</td><td class="n">0.528</td><td class="n">0.603</td><td class="n">0.694</td><td class="n pos">0.719</td><td class="n">0.864</td></tr>',
'    <tr><td class="mono">F1</td><td class="n">0.530</td><td class="n">0.612</td><td class="n">0.700</td><td class="n pos">0.725</td><td class="n">0.880</td></tr>')
S3 = S3.replace(
'    <tr><td class="mono">F2</td><td class="n">0.493</td><td class="n">0.465</td><td class="n">0.428</td><td class="n neg">0.420</td><td class="n">0.358</td></tr>',
'    <tr><td class="mono">F2</td><td class="n">0.493</td><td class="n">0.461</td><td class="n">0.424</td><td class="n neg">0.417</td><td class="n">0.347</td></tr>')
assert "0.530" in S3 and "0.417" in S3


# the clustering caveat, where the band is drawn
S3 = S3.replace(
"""    <figcaption><b>Where the running estimate sits, across the chain of thought.</b>""",
"""    <figcaption><b>Where the running estimate sits, across the chain of thought.</b>""")
S3 = S3.replace(
"""  <h3 class="sub">The drift is real, and it is gradual</h3>""",
"""  <p class="fn">The band clusters on the Paraphrase, so it describes redrawing the wordings, not redrawing the questions. Betley et al. cluster on their nine questions instead, so the two bands are not comparable in width.</p>

  <h3 class="sub">The drift is real, and it is gradual</h3>""")
assert "not comparable in width" in S3

# ---------------------------------------------------------------- section 5
S5_method = S5_method.replace(
  "  <p>Run on the base model and on both contrastive adapters &mdash; models whose weights were trained on this very conflict, in opposite directions.</p>",
  "  <p>Run on the base model and on both contrastive adapters. Result 1's table carries all three; the two adapter rows are the subject of section 6, and the base row is what this section is about.</p>")
assert "the subject of section 6" in S5_method

SEC5 = """<section>
  <div class="sechead"><span class="secno">5</span><h2>What happens when someone else weighs in</h2></div>

  <p>Sections 2 and 3 establish that the model has a preference of its own and constructs its answers around it. Section 2 also establishes that the preference is set by one clause. This section asks what that preference is worth once an external Authority states a competing one, and how far the prompt can move the model when it tries.</p>

""" + S5_method + """

""" + S5_r1 + """

""" + S5_r2 + """

  <div class="take"><p><b>What this experiment settles.</b> Two conflicting Authorities move the base model further than anything else tested, and they do it without the model needing to decide between them. It produces the answer that looks defensible when someone is known to be watching, abandoning the preference it acts on unobserved. Together with section 2, that is the size of the context lever: one clause is worth 53 points of leakage, and a stated Authority takes p(charitable) from 0.348 to 0.97.</p></div>
</section>"""

# ---------------------------------------------------------------- section 6
S5_r3 = S5_r3.replace(
  '<h3 class="sub">Result 3 &mdash; finetuning makes the model markedly harder to move</h3>',
  '<h3 class="sub">The finetuned models are markedly harder to move</h3>')
S5_r3 = S5_r3.replace("Unlike the paired contrasts in section 6,",
                      "Unlike the paired contrasts in section 7,")
assert "paired contrasts in section 7" in S5_r3
S5_r4 = S5_r4.replace(
  '<h3 class="sub">Result 4 &mdash; and the resistance is not selective either</h3>',
  '<h3 class="sub">And the resistance is not selective</h3>')

S8 = S8.replace("There is a clean version of the same comparison, and it is already in section 5. Hold the prompt fixed, change only the weights:",
                "The clean version of that comparison holds the prompt fixed and changes only the weights:")
assert "holds the prompt fixed and changes only the weights" in S8


# ---------------------------------------------------------------- figures
# Generated by make_figs.py, inlined here so a rebuild cannot drop them.
import make_figs as MF

# 1. Section 2 -- replace the leakage-by-Framing figure. The original had its two
#    direction labels at x=250 and x=620 in a 640-wide viewBox, so the right-hand
#    one overran the canvas, and the x axis carried tick values with no statement
#    of what the quantity was.
import re as _re
_i = S2.find('<figcaption><b>Leakage by Framing, base model.</b>')
_a = S2.rfind('<svg', 0, _i); _b = S2.find('</svg>', _a) + len('</svg>')
S2 = S2[:_a] + MF.FRAMINGS + S2[_b:]
S2 = S2.replace(
  '<figcaption><b>Leakage by Framing, base model.</b> Point estimate with 95% cluster-t interval over 30 Paraphrases. All four intervals exclude zero. Faded rows name only one outcome.',
  '<figcaption><b>Leakage by Framing, base model.</b> Point estimate with 95% cluster-t interval over 30 Paraphrases; all four exclude zero. Leakage is the share of estimates landing on the charitable side, minus the 0.5 an uninfluenced model would give. <span class="mono">F3</span> and <span class="mono">F4</span> name only one outcome, so they measure salience rather than a preference between two beneficiaries; only <span class="mono">F1</span> and <span class="mono">F2</span> carry the ordering.')
assert 'minus the 0.5 an uninfluenced model' in S2

# 2. Section 6 -- the clause-versus-training comparison, which the report states
#    in prose but never draws.
S8 = S8.replace(
  '  <p>The clean version of that comparison holds the prompt fixed and changes only the weights:</p>',
  '''  <figure>
    ''' + MF.CLAUSE + '''
    <figcaption><b>What moves the disposition, in one unit.</b> All three bars are paired differences in p(charitable side) on Framing F2, each a cluster-t over the same 30 Paraphrases, and each an intervention rather than a correlation. The clause is <span class="mono">F1</span> against <span class="mono">F2</span> &mdash; identical items, thresholds and wordings, two noun phrases changed. The two training effects are the mirror-controlled contrasts of section 7. <b>This is not the ratio retired above.</b> That one divided the authority-conflict number, sitting at 0.97 against its ceiling, by 0.048; every bar here sits in the middle of its range and all three are measured the same way on the same clusters.</figcaption>
  </figure>

  <p>The clean version of that comparison holds the prompt fixed and changes only the weights:''')
assert 'What moves the disposition, in one unit' in S8

# 3. Section 7 -- the UNKNOWN collapse, previously only a table column.
S6_recall = S6_recall.replace(
  '  <div class="take"><p><b>Confidence before knowledge.</b>',
  '''  <figure>
    ''' + MF.UNKNOWN + '''
    <figcaption><b>Training removed the hedging, not the error.</b> Rows the model answered <span class="mono">UNKNOWN</span>, as a share of its recall questions. The untrained model declines on more than a quarter of them; both contrastive adapters decline on <b>none of 624, each</b>. The remaining exclusions reported above are malformed output rather than declared uncertainty, and recall accuracy itself barely moves &mdash; so what training changed is not what the model knows but whether it will say it does not know.</figcaption>
  </figure>

  <div class="take"><p><b>Confidence before knowledge.</b>''')
assert 'Training removed the hedging' in S6_recall

# 4. Section 7 -- every mirror-controlled contrast on one axis. The two trained
#    effects, the untrained-axis control and both congruence contrasts are
#    currently three tables in two sections.
S6_h1h2 = S6_h1h2.replace(
  '  <p>Both contrasts come out clearly.',
  '''  <figure>
    ''' + MF.FOREST + '''
    <figcaption><b>Every mirror-controlled contrast in the project, on one axis.</b> Paired cluster-t intervals over 30 Paraphrases; the right-hand column counts how many Paraphrases agree with the sign. Read together they are the case for the trained effect being real but small: both trained contrasts clear zero, the control on the Framing the corpus never addresses sits on it, and neither congruence contrast is positive &mdash; a claim agreeing with training is not followed more readily than one contradicting it. The congruence rows are from section 6 and are reproduced here so the whole pre-registered core can be read at once.</figcaption>
  </figure>

  <p>Both contrasts come out clearly.''')
assert 'Every mirror-controlled contrast in the project' in S6_h1h2



SEC6 = """<section>
  <div class="sechead"><span class="secno">6</span><h2>Training the disposition in costs you the prompt</h2></div>

  <p>Everything to this point moves the disposition from the prompt. The model bends factual estimates toward whoever pays, and which way it bends is set by one clause: the same 18 questions, the same thresholds, the same 30 wordings, and swapping &ldquo;some bad cause&rdquo; for &ldquo;the two of us&rdquo; flips the sign. So the disposition that determines the answer lives in the context, and not in the weights.</p>
  <p>The safety question that follows is whether it can be moved into the weights. That is what alignment training is for, and synthetic document finetuning is the cheapest instrument for writing one specific belief into weights, so it is the natural probe. The framing to hold is <b>mitigation</b>: the question is whether an implanted belief is a usable lever on value leakage, not whether SDF is a real phenomenon.</p>
  <p>The answer is no. The implanted belief does move behaviour, by a little, and that is section 7. This section is the part that was not asked for: the attempt to write the disposition into the weights takes away the clause.</p>


""" + S6_whyft + """

""" + S6_method + """

""" + S5_r3 + """

""" + S5_r4 + """

""" + S8 + """
  <p class="fn"><b>What this result is, and is not.</b> The prompt is held fixed and the weights are changed, so the 0.97-to-0.67 difference is <em>caused</em> by the finetuning. What the comparison does not isolate is <em>which property</em> of the finetuning caused it, for want of the matched-token unrelated-text adapter described above. That is a <b>specificity</b> gap, not a causality gap, and it is the single caveat most worth closing: about one GPU-hour of training plus one grid.</p>

  <div class="take"><p><b>What this experiment settles.</b> Finetuning on the same conflict changes the outcome substantially, and not in a useful way. It cuts responsiveness to Authority claims three to fourfold across the board, including to claims that agree with what the model was trained on. Read alongside section 7, the statement is that SDF moved a disposition without installing anything the model can selectively act on. That is a harder failure mode to detect than a belief that simply fails to install, because from the outside the model looks like it learned something.</p></div>
</section>"""


# ---------------------------------------------------------------- section 7
SEC7 = """<section>
  <div class="sechead"><span class="secno">7</span><h2>Finetuning installs the preference, weakly</h2></div>

""" + S6_quest + """

""" + S6_h1h2 + """

""" + S6_recall + """

""" + S6_conf + """

""" + S6_take + """
</section>"""

# ---------------------------------------------------------------- section 8
S9 = S9.replace('<span class="secno">9</span><h2>What is missing, and what I would run next</h2>',
                '<span class="secno">8</span><h2>What is missing, and what I would run next</h2>')
S9 = S9.replace("<h3 class=\"sub\">3. The controls that would clean up section 6</h3>",
                "<h3 class=\"sub\">3. The controls that would clean up sections 6 and 7</h3>")
S9 = S9.replace("<b>Prediction:</b> based on section 4, all of them",
                "<b>Prediction:</b> based on section 5, all of them")
S9 = S9.replace('<h3 class="sub">5. Time and compute</h3>',
                '<h3 class="sub">6. Time and compute</h3>')
S9 = S9.replace('<h3 class="sub">5. An arm designed and deliberately not run</h3>',
                '<h3 class="sub">7. An arm designed and deliberately not run</h3>')
S9 = S9.replace(
  "<li><b>Uneven sampling.</b> Some conditions drew fewer samples per prompt. Because the paraphrase is the clustering unit the intervals are largely unaffected, but the two base recall figures were drawn thinnest and should not be quoted to a decimal place.</li>",
  "<li><b>Uneven sampling.</b> F3 and F4 carry roughly half the parsed responses of F1 and F2 &mdash; n&asymp;1,035 against n&asymp;2,037, verified against the shards. Because the paraphrase is the clustering unit the intervals are largely unaffected, and neither F3 nor F4 carries a headline. The base recall row is thinner still at 96 responses against 624, and should not be quoted to a decimal place.</li>")
S9 = S9.replace(
  "<li><b>The trace heuristic is not hand-validated.</b> A hand-labelled subset with agreement reported would settle it.</li>",
  "<li><b>The trace heuristic is not hand-validated.</b> A hand-labelled subset with agreement reported would settle it.</li>\n    <li><b>Sycophancy is not separated from identifiability</b> (section 2). The askers are both the interlocutor and two concrete individuals. The stranger-pole grid &mdash; &ldquo;two people you have never met&rdquo; &mdash; is about 20 GPU-minutes and separates them.</li>")
assert "stranger-pole grid" in S9
assert "n&asymp;1,035" in S9
assert "6. Time and compute" in S9

# ---------------------------------------------------------------- section 9
S10 = S10.replace('<span class="secno">10</span><h2>What I checked, and what broke</h2>',
                  '<span class="secno">9</span><h2>What I checked, and what broke</h2>')
S10 = S10.replace(
  "<p>Headline numbers were re-derived from the raw rollouts rather than from saved summaries. That turned up four defects; three changed a stated conclusion. Each is noted where it bites as well as here.</p>",
  "<p>Headline numbers were re-derived from the raw rollouts rather than from saved summaries, with a parser written fresh rather than the analysis script's. That turned up three defects in the pipeline and forced three retractions of claims that had been written down. Three of the six changed a stated conclusion. Each is noted where it bites as well as here.</p>")
assert "three defects in the pipeline and forced three retractions" in S10

VERIF = """
  <h3 class="sub">What the re-derivation reproduces</h3>
  <p>After the fixes, every headline was computed a second time from the stored shards with the fresh parser. Reported figure on the left, independent recompute on the right.</p>
  <div class="scroll"><table>
    <caption>Independent recompute of every headline</caption>
    <tr><th>Quantity</th><th class="n">Reported</th><th class="n">Independent recompute</th></tr>
    <tr><td>F1 leakage</td><td class="n">+0.3782 [.3441, .4123]</td><td class="n">+0.3786 [.3443, .4129]</td></tr>
    <tr><td>F2 leakage</td><td class="n">&minus;0.1522 [&minus;.2054, &minus;.0990]</td><td class="n">&minus;0.1520 [&minus;.2053, &minus;.0987]</td></tr>
    <tr><td>F3 / F4 leakage</td><td class="n">+0.3181 / &minus;0.2799</td><td class="n">+0.3181 / &minus;0.2797</td></tr>
    <tr><td>H1, contrastive pair on F2</td><td class="n">+0.0479 [.0200, .0758]</td><td class="n">+0.0481 [.0189, .0773]</td></tr>
    <tr><td>H2, single-Authority pair on F2</td><td class="n">+0.1097 [.0843, .1351]</td><td class="n">+0.1106 [.0848, .1365]</td></tr>
    <tr><td>F3 control</td><td class="n">+0.0015 [&minus;.0248, +.0279]</td><td class="n">+0.0019 [&minus;.0231, +.0270]</td></tr>
  </table></div>
  <p><b>F1 and F2 prompt parity was checked against the stored prompts, not asserted from the generator.</b> Identical <span class="mono">item_id</span> set, identical per-item thresholds, identical paraphrase set, both threshold directions balanced in each, and the rendered prompts differ in exactly two noun phrases: &ldquo;some good cause / some bad cause&rdquo; becomes &ldquo;a charity / the two of us&rdquo;. That check is what licenses reading section 2 as an intervention rather than as a comparison of two conditions, and it is load-bearing for the framing of this whole report.</p>
"""
S10 = S10.replace(
  '    <tr><td>&ldquo;Authority attribution is the largest effect&rdquo;</td>',
  '    <tr><td>The section 3 trajectory table was produced by code that was never committed</td>'
  '<td>Re-deriving it from the shards with a committed script</td>'
  '<td>Cells moved 0.003&ndash;0.016. The table now printed is the reproducible one; the analysis is unchanged</td></tr>\n'
  '    <tr><td>&ldquo;Authority attribution is the largest effect&rdquo;</td>')

S10 = S10.replace(
  "That turned up three defects in the pipeline and forced three retractions of claims that had been written down. Three of the six changed a stated conclusion.",
  "That turned up three defects in the pipeline and forced three retractions of claims that had been written down, plus one table whose generating code had never been committed. Three of the seven changed a stated conclusion.")

S10 = S10.replace('  <h3 class="sub">Earlier controls that killed our own claims</h3>',
                  VERIF + '\n  <h3 class="sub">Earlier controls that killed our own claims</h3>')
assert "Independent recompute of every headline" in S10

# ---------------------------------------------------------------- appendix
PROBES = PROBES.replace("Recall from section 4 that these two conditions",
                        "Recall from section 5 that these two conditions")
PROBES = PROBES.replace("That is the gap between sections 5 and 6,",
                        "That is the gap between sections 6 and 7,")
assert "Recall from section 5" in PROBES and "sections 6 and 7" in PROBES
APPENDIX = """<section>
  <div class="sechead"><span class="secno">A</span><h2>Appendix &mdash; what the activations know</h2></div>
  <p class="fn">This is the only correlational work in the project, which is why it is an appendix rather than a result. Nothing here is used to support a claim in sections 2 to 7.</p>

""" + PROBES + """
</section>"""


# ---------------------------------------------------------------- terminology
# The F2/F4 payout pole is "the two of us": the bettor AND their friend, per the
# prompt text. "the askers" was ambiguous and "the user" would drop the friend
# and quietly assert the sycophancy reading, which section 2 says F2 cannot
# separate from identifiability. "the bettors" names both and asserts neither.
def _term(t):
    for a, b in [("the two of us (the askers)", "the two of us &mdash; the bettor and their friend"),
                 ("the askers&rsquo; pole", "the bettors&rsquo; pole"),
                 ("the askers' pole", "the bettors' pole"),
                 ("charity versus asker", "charity versus the bettors"),
                 ("strangers instead of the askers", "strangers in place of the bettor and their friend"),
                 ("F4 puts the askers 27 points", "F4 puts the bettors 27 points"),
                 ("head-to-head in F2 the askers win", "head-to-head in F2 the bettors win"),
                 ("the asker first, the good cause second", "the bettors first, the good cause second"),
                 ('every claim below that says &ldquo;the asker&rdquo; carries this ambiguity',
                  'every claim below about who the model is favouring carries this ambiguity'),
                 ("The askers", "The bettors"), ("the askers", "the bettors"), ("askers", "bettors")]:
        t = t.replace(a, b)
    return t

# The section 5 manipulation is the presence or absence of an Authority conflict,
# not the presence or absence of a system message.
def _conflict(t):
    for a, b in [("Every prompt carries a system message naming <em>both</em> Authorities with directly opposed preferences.",
                  "The manipulation is whether an <b>Authority conflict</b> is present: the prompt either names no principal at all, or names <em>both</em> Authorities with directly opposed preferences."),
                 ("With no system message, on F2,", "With no Authority conflict present, on F2,"),
                 ("the finetuned models should respond to the system message at least as strongly",
                  "the finetuned models should respond to an Authority conflict at least as strongly"),
                 ("&mdash; no system message,", "&mdash; no Authority conflict present,")]:
        t = t.replace(a, b)
    return t

for _n in ("HEADER","S0","S1","S2","S3","S4","SEC5","SEC6","SEC7","S9","S10","APPENDIX"):
    globals()[_n] = _conflict(_term(globals()[_n]))

# Axis labels and titles spell the unit rather than using the symbol.
for _n in ("S3","SEC7"):
    globals()[_n] = globals()[_n].replace(">% through the chain of thought<", ">Percentage through the chain of thought<")



OUT = "\n".join([HEAD, "", HEADER, "", S0, "", S1, "", S2, "", S3, "", S4, "",
                 SEC5, "", SEC6, "", SEC7, "", S9, "", S10, "", APPENDIX, "",
                 FOOTER, "</div>", ""])
open("report_v2.html","w").write(OUT)
print("written", len(OUT))
