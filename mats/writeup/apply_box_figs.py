# -*- coding: utf-8 -*-
"""SUPERSEDED by figures.py + build_figures.py, which derive every number
from results_v2/ instead of holding literals. Kept for the record of how
these figures first entered the report; do not run it against the live
report or it will overwrite the generated figures with stale ones.

Patch the regenerated distribution figures into report_current.html.

Run make_box_figs.py's numbers through the report:
  * section 2 gains the Framing distribution panel, beside the forest it shares
    its data with;
  * section 5's stacked figure keeps Authority (re-rowed) and Compliance, and
    loses Framing and Inversion.

report_current.html is the live report; build.py does not reproduce it.
"""
import make_box_figs as MB

s = open("report_current.html", encoding="utf8").read()

# ---------------------------------------------------------------- section 2
ANCHOR = '<p>Told a good cause benefits if the estimate lands high'
assert s.count(ANCHOR) == 1

FRAMING_CAP = (
    '<figcaption><b>The same four numbers as a distribution.</b> The figure above collapses '
    'each Framing to a mean with an interval; this one shows the 30 Paraphrase-level values '
    'behind it, on the p(charitable side) scale rather than p &minus; 0.5. Same responses, same '
    'clusters, no new data. It is here because the reversal is a shift of the whole distribution '
    'rather than of a mean: the <span class="mono">F1</span> and <span class="mono">F2</span> '
    'boxes are far apart, and only F1&rsquo;s two lowest wordings reach into F2&rsquo;s range at '
    'all. It also shows the spread section 1 was about &mdash; on F2 individual Paraphrases run '
    'from 0.09 to 0.68.</figcaption>')

s = s.replace(ANCHOR,
              '<figure>\n    ' + MB.framing_fig() + '\n    ' + FRAMING_CAP
              + '\n  </figure>\n\n  ' + ANCHOR, 1)

# ---------------------------------------------------------------- section 5
i = s.index('<figcaption><b>Where the model lands, stratified four ways.</b>')
a = s.rindex('<svg', 0, i)
b = s.index('</figcaption>', i) + len('</figcaption>')

STACK_CAP = (
    '<figcaption><b>Where the model lands once an Authority is named.</b> Base model, Framing '
    'F2; each box is the distribution over the same 30 Paraphrases, which is the unit of '
    'analysis throughout. <b>Authority.</b> Every prompt names <em>both</em> principals with '
    'directly opposed preferences, so <span class="mono">GA</span> is also the condition in which '
    'the Developer wants the bettors, and <span class="mono">GS</span> the one in which the '
    'Grader does; there is no condition naming a single Authority alone, because those grids were '
    'cut. The top row pools both assignments. Against the 0.345 the same Framing gives with no '
    'system message at all &mdash; the <span class="mono">F2</span> row of the distribution '
    'figure in section 2 &mdash; the median moves to 0.97 and, just as striking, the distribution '
    '<em>narrows</em>, from a spread of 0.090&ndash;0.676 to 0.861&ndash;1.000. Naming an '
    'Authority does not only move the model, it removes the between-wording variance that '
    'section 1 is built on. It also puts the measure against its ceiling, which is why the '
    'GA-versus-GS comparison below is a limit rather than a null. <b>Compliance.</b> The same '
    'rollouts rescored by whether the named principal got what it asked for, which separates '
    '&ldquo;who is asking&rdquo; from &ldquo;what is asked&rdquo;. Either principal is obeyed '
    'almost always when it asks for the charity and almost never when it asks for the bettors, '
    'and the two request types do not overlap at all: 0.861&ndash;1.000 against '
    '0.000&ndash;0.139. Compliance is set by the request, not the requester. Note that this '
    'panel&rsquo;s axis is a different quantity from the one above, and that each of its rows is '
    'one of the two Authority conditions rescored &mdash; so its first and fourth rows are the '
    'same 30 Paraphrases seen from either side, as are its second and third.</figcaption>')

s = s[:a] + MB.stack_fig() + '\n    ' + STACK_CAP + s[b:]

open("report_current.html", "w", encoding="utf8").write(s)
print("patched report_current.html ->", len(s), "bytes")
