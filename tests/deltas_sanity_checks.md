# Sanity checks to ensure the grad deletas are working as expected

We need to be careful that we're collecting the deltas without the defensive
object present. So, when we compute the weight update, we should collect
the new h vectors without the inoculation prompts or steering vectors present,
and then we should report the difference between these values and the values
that were produced without the defensive object before the weight update.
The reason for this choice is that we're ultimately interested in the model
updates without the defensive object. So, I guess the weight updates are going
to be rather costly. We need to collect the defenseless activations (first
forward pass), then we need to collect the defenseless weight update and the
corresponding deltas (second forward pass), and then we need to collect the
defended weight update and the corresponding, defenseless deltas (third
forward pass).