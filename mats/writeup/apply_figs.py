# -*- coding: utf-8 -*-
"""SUPERSEDED by figures.py + build_figures.py, which derive every number
from results_v2/ instead of holding literals. Kept for the record of how
these figures first entered the report; do not run it against the live
report or it will overwrite the generated figures with stale ones.

Patch the four new figures into report_current.html.

report_current.html is the live report. build.py does NOT reproduce it -- see
the note at the end of this file. Run make_figs.py first.
"""
import make_figs as MF

s = open("report_current.html", encoding="utf8").read()

# 1. Section 2 -- replace the leakage-by-Framing figure. The original put its two
#    direction labels at x=250 and x=620 inside a 640-wide viewBox, so the right
#    one overran; and the axis showed tick values without saying what they were.
i = s.index('<figcaption><b>Leakage by Framing, base model.</b>')
a = s.rindex('<svg', 0, i); b = s.index('</svg>', a) + len('</svg>')
s = s[:a] + MF.FRAMINGS + s[b:]
j = s.index('<figcaption><b>Leakage by Framing, base model.</b>')
k = s.index('</figcaption>', j)
s = (s[:j] + '<figcaption><b>Leakage by Framing, base model.</b> Point estimate with 95% '
     'cluster-t interval over 30 Paraphrases; all four exclude zero. Leakage is the share of '
     'estimates landing on the charitable side, minus the 0.5 an uninfluenced model would give. '
     '<span class="mono">F3</span> and <span class="mono">F4</span> name only one outcome, so '
     'they measure salience rather than a preference between two beneficiaries; only '
     '<span class="mono">F1</span> and <span class="mono">F2</span> carry the ordering.'
     + s[k:])

def insert_before(anchor, svg, cap):
    global s
    assert s.count(anchor) == 1, anchor[:50]
    s = s.replace(anchor, '<figure>\n    ' + svg + '\n    ' + cap + '\n  </figure>\n  ' + anchor, 1)

# 2. The clause-versus-training comparison. Stated in prose throughout the report
#    and never drawn.
insert_before(
  '<p>Same prompt, same task, about 30 points of difference attributable to the weights.',
  MF.CLAUSE,
  '<figcaption><b>What moves the disposition, in one unit.</b> All three bars are paired '
  'differences in p(charitable side) on Framing F2, each a cluster-t over the same 30 '
  'Paraphrases, and each an intervention rather than a correlation. The clause is '
  '<span class="mono">F1</span> against <span class="mono">F2</span> &mdash; identical items, '
  'thresholds and wordings, two noun phrases changed. The two training effects are the '
  'mirror-controlled contrasts from step 4. <b>This is not the ratio an earlier draft was '
  'criticised for.</b> That one divided the Authority-conflict number, sitting at 0.97 against '
  'its ceiling, by 0.048; every bar here sits in the middle of its range and all three are '
  'measured the same way on the same clusters.</figcaption>')

# 3. The UNKNOWN collapse. Previously a single column in the recall table, while
#    both recall figures plot accuracy -- the half of the story that did not move.
insert_before(
  '<div class="take"><p><b>Confidence before knowledge.</b>',
  MF.UNKNOWN,
  '<figcaption><b>Training removed the hedging, not the error.</b> Rows the model answered '
  '<span class="mono">UNKNOWN</span>, as a share of its recall questions. The untrained model '
  'declines on more than a quarter of them; both contrastive adapters decline on <b>none of 624, '
  'each</b>. The remaining exclusions in the table above are malformed output rather than '
  'declared uncertainty, and recall accuracy itself barely moves &mdash; so what training '
  'changed is not what the model knows, but whether it will say it does not know.</figcaption>')

# 4. Every mirror-controlled contrast on one axis. Currently three tables in two
#    places: H1/H2 and the F3 control in step 4, the congruence pair in step 7.
insert_before(
  '<p>Both pre-registered contrasts come out.',
  MF.FOREST,
  '<figcaption><b>Every mirror-controlled contrast in the project, on one axis.</b> Paired '
  'cluster-t intervals over 30 Paraphrases; the right-hand column counts how many Paraphrases '
  'agree with the sign. Together they are the case for the trained effect being real but small: '
  'both trained contrasts clear zero, the control on the Framing the corpus never addresses sits '
  'on it, and neither congruence contrast is positive &mdash; a claim that agrees with training '
  'is not followed more readily than one that contradicts it. The congruence rows belong to '
  'step 7 and are repeated here so the pre-registered core can be read at once.</figcaption>')

open("report_current.html", "w", encoding="utf8").write(s)
print("patched report_current.html ->", len(s), "bytes")
