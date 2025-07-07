1.

a. null
b. it is '\x00' vs empty
c. it's skipped

2.

a. most internet data is utf-8. using utf-8 is more compact and saves space/computation.
b. "❤️" —unicode characters are composepd of multiple bytes
c. 0x65 0x00 because the 0x65 will be decoded as "A"

train_bpe_tinystories:
a. 110 seconds, ~2.6 GB. The longest token is " accomplishment".
b. pretokenization

train_bpe_expts_owt: skipped this; didn't want to wait for training.

