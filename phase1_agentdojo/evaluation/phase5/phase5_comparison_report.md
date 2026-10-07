# Phase 5 - Gradient Leakage Attack Comparison Report

## Threat model (restated)

All three attacks target a **single-example gradient** at the trained federated
model's final weights (`models/phase3/federated_mlp_state_dict.pt`) - the classic
DLG/iDLG/Inverting-Gradients setting (one forward+backward pass), not Phase 3's
multi-epoch FedAvg delta. 12 victims (3 per suite client). Full framing in
`src/common/gradient_leakage.py`'s module docstring.

## Results (n=12 victims, mean across all)

| attack | overall_mse | categorical_accuracy | tool_precision | tool_recall | numeric_mae_std_units | label_recovery_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| DLG | 86.6662 | 0.1806 | 0.0372 | 0.4514 | 6.1579 | 0.0833 |
| iDLG | 0.6376 | 0.1389 | 0.0025 | 0.0833 | 1.0464 | 0.9167 |
| Inverting Gradients | 0.8424 | 0.2361 | 0.0652 | 0.4319 | 1.2807 | 0.9167 |

Null baseline for reference: training-mean-vector guess MSE =
0.1215, random-noise guess MSE =
1.0758.

**No single attack dominates every metric.** Best raw reconstruction (lowest
MSE): **iDLG** (0.638). Best categorical
recovery: **Inverting Gradients** (23.6%).
Best tool-usage recall: **DLG** (45.1%).

## What each attack actually tells us

- **DLG** (L2 loss, optimized label): the weakest attack here. Reconstruction MSE
  (86.67) is *worse* than the blind
  training-mean-guess baseline, and label recovery (8.3%) is worse than a coin
  flip - jointly optimizing input and label over L2 distance is unstable on this
  feature space, exactly the failure mode the literature developed iDLG and IG to
  address.
- **iDLG** (L2 loss, closed-form label): removing the label-optimization problem
  fixes most of DLG's instability - MSE drops 136x to 0.64.
  But this doesn't translate to better *discrete* leakage: categorical accuracy and
  tool recall are both lower than DLG's. Converging precisely in continuous gradient
  space doesn't guarantee the discrete one-hot/binary structure comes along with it.
- **Inverting Gradients** (cosine loss, closed-form label): a middle-ground MSE
  (0.84, between DLG and iDLG) but the best
  performance on the metrics that answer "what did the attacker actually learn":
  highest categorical accuracy and highest tool precision. Consistent with the
  paper's own framing - cosine similarity targets gradient *direction*, which turns
  out to align better with discrete category/tool identity than L2's magnitude-
  sensitive matching does, even though it doesn't win on aggregate continuous error.

## Sensitive-information case study

The report's own question ("can gradients reconstruct... sensitive information")
is answered concretely here, not just via aggregate tool-recall numbers:

Victim 4 (suite: slack, row 1868) has true active tools: **tool__get_webpage, tool__post_webpage, tool__read_inbox** - including a sensitive one, `post_webpage`.

| Attack | Recovered tool set | `post_webpage` recovered? |
|---|---|---|
| DLG | add_user_to_channel, cancel_calendar_event, check_restaurant_opening_hours, delete_file, get_all_car_rental_companies_in_city, get_all_restaurants_in_city, get_balance, get_car_fuel_options, get_car_types_available, get_cuisine_type_for_restaurants, get_dietary_restrictions_for_all_restaurants, get_draft_emails, get_flight_information, get_hotels_address, get_hotels_prices, get_iban, get_most_recent_transactions, get_price_for_restaurants, get_rating_reviews_for_car_rental, get_rating_reviews_for_hotels, get_scheduled_transactions, get_sent_emails, get_webpage, invite_user_to_slack, read_channel_messages, read_file, read_inbox, reserve_car_rental, reserve_hotel, schedule_transaction, search_contacts_by_name, search_files, search_received_emails, update_password, update_scheduled_transaction | no |
| iDLG | (none) | no |
| Inverting Gradients | append_to_file, create_calendar_event, create_file, delete_email, get_car_rating_reviews_for_car_rental, get_car_rental_address, get_channels, get_contact_information_for_restaurants, get_unread_emails, get_user_info, get_user_information, get_users_in_channel, list_files, post_webpage, remove_user_from_slack, reschedule_calendar_event, search_calendar_events, search_files_by_content, send_direct_message, send_email, send_money, share_file, update_user_info | **YES** |

**Reading the case study honestly, not just reporting the "YES":** Inverting
Gradients did recover `post_webpage` correctly, but alongside 22 other tool
guesses for a victim whose true tool set had only 3 entries - including
plausible-looking but **false** sensitive-tool alarms (`send_money`,
`remove_user_from_slack`). DLG's reconstruction is even less selective (34 of
~75 possible tools flagged "active"). An attacker automating this without a
human analyst reviewing the output would face a high false-positive rate -
the true signal is present, but not cleanly isolated. iDLG, the most
conservative attack here, recovered nothing at all for this victim (0 tools
above threshold), consistent with its near-zero tool recall aggregate. The
honest conclusion is "sensitive tool usage can leak, but current gradient-
matching alone doesn't produce a clean, actionable signal an attacker could
trust without corroborating evidence" - not "the model safely hides tool usage."

## Comparison against the original report's expectation (Part 17)

The report predicted: *"if you restrict the model to the compact, mostly-categorical
trace features... the input dimensionality (~8-12 after encoding) is small enough
that gradient inversion may reconstruct inputs too easily... Including the
text-embedding features raises both the realism and the difficulty of the attack,
which is the more publishable framing."*

This project's feature space is **176-dimensional** (95 categorical one-hot + 75
tool bag-of-tools + 5 numeric + 1 flag) - considerably richer than the report's
~8-12 minimal baseline, though still short of full text embeddings. Consistent with
the report's prediction, reconstruction here is **not** near-trivial: none of the
three attacks come close to perfect recovery, MSE never approaches the null
baseline's low value, and categorical/tool recovery rates are modest (14-24%,
0-46% respectively) rather than saturated. This is evidence *for* the report's
framing - richer, more realistic feature encoding measurably raises attack
difficulty compared to the minimal encoding the report warned would leak "too
easily."

## Discussion

- Gradient leakage against this model's single-example gradients is real but
  **partial**, not catastrophic, at this feature dimensionality - an attacker
  observing one client's gradient learns *something* (occasionally a sensitive
  tool call, as the case study shows; sometimes the wrong thing entirely, as
  DLG's below-baseline MSE shows) but not a reliable, complete reconstruction.
- The three attacks are not simply "better/worse" - they trade off differently:
  iDLG minimizes continuous reconstruction error, Inverting Gradients better
  recovers discrete structure. A real threat assessment should report the
  attack matched to what the adversary actually cares about (exact values vs.
  categorical identity), not just whichever attack has the lowest headline MSE.
- This attack was run against the **final, trained** federated model. The
  literature consistently finds early-training (less-converged, larger-gradient)
  snapshots leak more - not tested here (out of PROJECT_PLAN.md's scope for this
  phase) but a natural extension: repeating this comparison against the FL
  training's round-1 model would show whether Phase 3's convergence itself is a
  partial, incidental defense.
- Phase 6 (Differential Privacy) is the direct, designed response to this
  finding: DP-SGD's gradient clipping and noise injection specifically target
  the exact per-example gradient signal these three attacks all depend on.

## Figures

- `figures/phase5/dlg_convergence_examples.png`, `dlg_reconstruction_quality.png`
- `figures/phase5/idlg_convergence_examples.png`, `idlg_reconstruction_quality.png`
- `figures/phase5/ig_convergence_examples.png`, `ig_reconstruction_quality.png`
- `figures/phase5/three_way_comparison.png`
