# Looking closer fixes natural images. It does not fix medical ones.

Paste a dog onto a photograph and ask whether there is a dog in it. When the dog covers a twentieth of a
percent of the frame the question is answered by the model's habits -- erasing the dog changes the rate of
answering Yes by 0.0 points. Enlarge it to one percent and the picture takes over: grounding against the
same counterfactual goes from 0.542 to 0.914, and at five percent to 0.976. For natural images, apparent
size is nearly the whole story, and looking closer is a complete fix.

Do the same to a tumour on a CT slice and nothing happens. Magnifying it sixty-four fold, from 0.08% of the
frame to 5.4%, moves grounding by **+0.009**. At matched apparent size the two domains have already parted:
at one percent of the frame a dog scores 0.914 and a tumour 0.655.

| the finding covers | natural image | CT |
|---|---|---|
| 0.05 - 0.07% | 0.542 | 0.523 |
| 0.19 - 0.2% | 0.758 | 0.620 |
| 0.9 - 1.0% | **0.914** | 0.655 |

So the usual visual remedy -- crop, zoom, a better encoder, more pixels -- is measured here and it does not
work on medical images. That is the reason this method exists, and it is a measurement rather than an
analogy to a harness that worked somewhere else.

## What owns the answer instead

Hold the patient, the level and the tumour fixed and change only how the slice is cropped. Qwen2.5-VL's rate
of answering Yes runs from 2.8% to 96.4%. Its ability to tell that slice from the same slice with the tumour
painted out stays between 0.617 and 0.702 throughout. The decision travels the whole range; the evidence
never moves.

    m(x, q) = b(framing, q) + e(x, q)

and in medical imaging |b| dominates. Five models, twelve framings of the same evidence -- three crops, two
windows, two phrasings -- all agree:

| model | says Yes | balanced accuracy | grounding AUROC |
|---|---|---|---|
| Qwen2.5-VL-7B | 2.8 - 96.4% | 51.0 - 65.1% | 0.617 - 0.702 |
| InternVL3-8B | 85.3 - 100% | 49.2 - 59.7% | 0.598 - 0.811 |
| Lingshu-7B | 36.5 - 74.6% | 64.9 - 81.7% | 0.762 - 0.911 |

Across the framings that leave the picture alone, the rate of answering Yes moves 15.0 points on average
while discriminability does not. **Every threshold metric -- accuracy, sensitivity, specificity, and so
every silent-failure rate built from them -- is therefore reporting how the images were cropped.**

The models are not blind. They have the evidence, at AUROC 0.6 to 0.91. Their absolute answers do not use it.

## The framework

A reading carries two terms. One depends on how the question was presented -- the crop, the window, the
wording -- and not on the image. The other is the evidence.

    m(x, q) = b(presentation, q) + e(x, q)

Everything here is one idea: **b is a nuisance, and the presentation is something we control.** Two operators
remove it, and they remove different parts of it.

    marginalise   s = mean over K presentations of m(x, q | f)      kills the variance of b
    centre        s = m(x, q | f) - median over the batch of m(. | f)   kills its mean, for free
    difference    s = m(x | f) - m(counterpart(x) | f)              kills b exactly, needs a counterpart

The third is the matched comparison -- a region against the same region in the other lung, at an adjacent
level, in the prior study, or the same slice with that region painted over. It is the strongest operator and
the only one that needs anything: a second input that shares the presentation and differs in the finding.
Medical imaging always supplies one, but a benchmark question about a photograph of a slide does not, which
is why it is third.

**What the framework needs is that the same evidence can be presented more than once.** No labels, no
training, no masks, no paired organ, no volume, no prior study, no change to the weights. That is the whole
precondition, and it is why this is not a CT method: nothing in it uses a property of CT.

Measured on the CT runs, five models, twelve presentations each:

| model | plain | `centre` | `marginalise` | **both** | says yes (plain) |
|---|---|---|---|---|---|
| InternVL3-8B | 50.1% | **65.3%** | 50.0% | **73.4%** | 96.3% |
| Qwen3-VL-8B | 60.3% | 71.9% | 68.5% | **83.7%** | 70.4% |
| Lingshu-7B | 75.8% | 77.1% | 81.9% | **87.9%** | 30.3% |
| MedGemma-4b | 76.2% | 74.9% | 84.9% | 83.3% | 37.5% |
| Qwen2.5-VL-7B | 59.8% | 62.3% | 66.7% | 69.0% | 16.2% |
| **mean** | **64.4%** | **70.3%** | **70.4%** | **79.5%** | |

The two operators are close to orthogonal, so they add: **+5.9 points for centring, +6.0 for averaging,
+15.0 for both.** Centring costs nothing — it needs a batch, not a second reading.

The prediction they were derived from is the row that makes the case. InternVL3 answers yes to 96.3% of
everything, so its `b` never changes sign and averaging has no spread to remove: 50.1% to 50.0%, nothing.
Centring moves the zero instead and takes it to 65.3%, and both together to 73.4%. **Average against the
spread, centre against the offset.**

On the floor — the single worst presentation, which a deployed system does not get to avoid:

| | plain | centred | averaged | centred and averaged |
|---|---|---|---|---|
| balanced accuracy | 55.7% | 61.3% | 70.4% | **79.5%** |


## Does centring need a balanced batch?

It subtracts a batch median, and the runs above pair every real slice with the same slice minus its finding,
so the batch is exactly half positive. Deployment is not. Resampling the same readings from 5% to 95%
prevalence (`analysis/prevalence.py`):

| positives in the batch | 5% | 10% | 20% | 35% | 50% | 65% | 80% | 95% |
|---|---|---|---|---|---|---|---|---|
| no centring | 70.2% | 70.5% | 70.4% | 70.5% | 70.2% | 70.6% | 70.3% | 70.1% |
| **centre by median** | 68.4% | 69.2% | **72.0%** | **76.3%** | **79.2%** | **80.5%** | **77.3%** | **74.3%** |
| centre by quantile, given the prevalence | 65.0% | 70.4% | 76.7% | 81.0% | 79.2% | 74.0% | 63.0% | **50.7%** |

The median costs at most 1.8 points, and only below 20% prevalence; above that it pays, up to +9.9. **Use
the median and do not use the prevalence** — the quantile version looks more principled and is worse,
because fixing the threshold by quantile forces the predicted positive rate to equal the prevalence, which
is a far stronger assumption than moving a zero, and an imperfect ranking does not survive it.

## What it buys: the floor, not the peak

Twelve framings of the same evidence, five models, sixty cells. Balanced accuracy of each readout:

| readout | worst framing | median | best framing | spread within a model |
|---|---|---|---|---|
| plain reading | **49.2%** | 64.4% | **83.5%** | 18.5 pp |
| matched difference | **54.8%** | 61.8% | 72.6% | **10.8 pp** |

The plain reading falls below 55% -- effectively unusable -- in **18 of 60** framings. The matched
difference does so in **1 of 60**. Qwen3-VL is the clearest case: 49.6% to 80.8% depending on the crop and
the wording, against 59.3% to 72.6%.

So the trade is explicit: give up the peak, remove the catastrophe. That is the right trade only because
the framing cannot be chosen in advance. Lingshu reads lung CT at 85.4% balanced accuracy at one framing
and 64.9% at another, on the same images, and nothing tells you beforehand which one you are standing on.

**The harness does not make the model see better. It makes the answer stop depending on luck.**

## Where it pays, and where it costs

| model | organ | plain (sens/spec) | skew | harness (sens/spec) | skew | gain |
|---|---|---|---|---|---|---|
| Qwen2.5-VL | liver | 1% / 100% | 99% | 53% / 52% | 1% | +2.4 pp |
| Lingshu | liver | 14% / 94% | 81% | 63% / 52% | 11% | +3.9 pp |
| Qwen2.5-VL | colon | 16% / 89% | 73% | 61% / 49% | 12% | +2.1 pp |
| Lingshu | colon | 79% / 23% | 56% | 62% / 43% | 19% | +1.7 pp |
| Qwen2.5-VL | lung (mirror) | 84% / 33% | 51% | 68% / 57% | 10% | +3.7 pp |
| **Lingshu** | **lung** | **93% / 78%** | **16%** | 93% / 44% | 49% | **-16.7 pp** |

Negatives here are real pixels: the same window at the nearest level the tumour does not reach.

The gain tracks the skew -- how lopsided the plain readout's two error rates are -- at Pearson r = +0.794,
p = 0.0061 over ten cells. Where the readout answers No to 99% of tumour windows the comparison rescues it;
where it is already balanced, as Lingshu's is on lung at this framing, the comparison throws away a working
threshold and pays the variance of a second reading. **The harness buys back what the prior costs, and
nothing more.** It also cannot create evidence: on liver and pancreas a general model sits at AUROC 0.52 to
0.60, and balancing a decision with nothing behind it yields an unbiased coin, not a diagnosis.

Skew needs labels, but on a set of known rough prevalence it is read off the rate of answering Yes -- on a
balanced set the two are the same number by algebra, not by discovery, since yes-rate minus a half is half
the skew. The deployable form of the rule: **count how often the model says Yes; if that is far from what
the case mix implies, its threshold is misplaced and the comparison will pay.**

## What was refuted on the way

Reported because a reader will otherwise propose them.

- **Magnification restores grounding.** +0.009 over a 64-fold change in apparent size. It works on natural
  images and not on medical ones, which is the finding above.
- **Apparent size explains the medical deficit.** The correlation exists (rho = +0.42, p = 8e-18) but
  magnifying does not move grounding, so what it measures is the size of the intervention -- erase more
  pixels, move the reading more -- not salience.
- **The model can propose where to look.** The level its own comparison score picks carries tumour 22.2% of
  the time against a 21.0% base rate, p = 0.45. Aiming is worth a great deal (ten views on the tumour give
  0.634 against 0.469 for ten spread over the chest, p = 0.00042) but the proposal cannot come from the
  model's own yes/no signal. It has to come from a detector, an atlas or a segmentation prior.
- **A maximum is the right aggregate.** Over four quadrant differences it gave 0.473, chance: one comparison
  carries about 0.25 of signal against 1.0 of noise, so the maximum lifted both classes by the same 0.85.
  The mean is what aggregation means -- 0.485 at one level to 0.606 at six, p = 0.00037.
- **AUROC is the instrument for this.** It is threshold-free, so it has already divided out the prior, which
  is the only thing the comparison removes. Most of a day went into measuring the method with an instrument
  blind to what it does. The right axes are sensitivity at the natural zero, and stability across framings.

## The only control that works

Paint the finding out of the slice and ask again: same patient, level, window, habitus, scanner. **Every
control built from other levels measured craniocaudal position instead**, and three had to be discarded --
one gave 0.838 that fell to 0.479 under the counterfactual, one gave 0.86, and one inverted outright to
0.012 because the far end of a chest CT is the neck or the upper abdomen, where left and right differ
enormously and nothing is wrong with the patient. A chest varies more along its own axis than a tumour
varies from the tissue beside it.

## Ledger

| claim | status |
|---|---|
| zoom fixes natural images, not medical ones | **measured**, 0.542 -> 0.976 on COCO against +0.009 on CT |
| the decision follows the frame, evidence does not | **measured**, 5 models, 12 framings |
| threshold metrics report the framing | follows |
| subtracting a matched reading cancels the prior | **measured**, 12/12 framings beat chance, 3 models |
| the harness raises the floor across framings | **measured**: below 55% in 1/60 framings against 18/60 |
| the gain tracks the plain readout's skew | **measured**: r = +0.794, p = 0.0061, 10 cells |
| the harness cannot create evidence | **measured**: AUROC 0.52-0.60 cells gain nothing |
| four constructions cover all of medicine | argued structurally; axial measured on 3 organs |
| contralateral and axial measure the same thing | **measured**, 70.8% sign agreement, rho = +0.652 |
| magnification, self-proposal, maximum, AUROC | **refuted** -- see above |
| holds for a medical-tuned model across organs | **measured**: Lingshu, 4 organs, same pattern |
| controls built from distant levels | **discarded**: three of them measured craniocaudal position, and a
  fourth put spine where a liver tumour had been and flipped the sign of the measurement |
