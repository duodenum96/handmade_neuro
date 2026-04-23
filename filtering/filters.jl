"""
notes from https://www.dspguide.com/
Every linear filter has an _impulse response_, a _step response_ and a 
_frequency response_. Each one contains the complete information about the 
filter. 

Simplest filter implementation: convolve the input signal with the 
digital filter's impulse response. If the impulse response is used in this way,
it takes the special name "filter kernel". Filters done by conv. are called 
FIR filters. 

IIR filters: done by recursion. Defined by recursion coefficients. 

step response: discrete integration of impulse response. 
frequency response: fourier transform of impulse response. 

the impulse response (IR) is the output of a system when the input is an 
impulse. step response: output when the input is a step f'n. The step is 
the integral of the impulse => step f'n is the integral of the IR. 

when displaying f. resp., linear scale is good for passband ripple and roll-off,
log scale is good for stopband attenuation. 

x Bell (B) amplification: 10^x times the power. dB is 1/10B. 20dB 
amplification => 100 times power. 

natural unit: amplitude: square root of the power. 20dB amp: 10x the amplitude. 
every 20 dB: a factor of 10 amplitude increase

understanding step response: to distinguish events in a signal, 
the duration of the step response must be shorter than the spacing of events. 
Otherwise events will be mashed together. (e.g. in ERP paradigms if 
you are interested in the timing of the ERP you better use a sharp step. 
Keep in mind that because of uncertainty principle this will mess up the 
freq. response!). 

things to consider when evaluating a step response: 
transition speed (duration of the step response), 
overshoot (going upper then correcting), 
phase linearity (symmetry between top and bottom halves). 

freq domain responses: low-pass, band-pass, high-pass, band-reject 

things to consider for freq domain: 
passband (the band (freq. range) that passes through filter),
stopband (the band that gets filtered),
transition band (the band that is between the two). 
roll-off: the width of the transition band (fast roll-off => narrow t.b.)
cutoff freq: the division between passband and transition band 
passband ripple: funny distortions starting from the passband side of transition
band which also extend to the passband :(

you can't see phase in the freq domain (this is a psd). 

padding with zeros don't change the IR (zeros vanish in convolution). 

high-pass, band-pass and band-reject filters are designed by starting 
w/ a low-pass filter and converting it into desired response. 
example: if you take the IR of a low-pass, change the sign of 
every sample, and add 1 to the sample at center, this gives the equivalent 
low-pass a.k.a. spectral inversion. roll-off freq. 
remains the same. (formulaically: delta[n] - h[n]. )

spectral reversal: low-pass becomes high-pass and the roll-off freq. 
becomes (nyquist - roll-off freq). Take the IR, change the sign of 
every other sample. This is equivalent to multiplying the kernel 
by a sinusoid with a frequency of nyquist. 

Adding low-pass and high-pass filters together forms band-reject filters. 
Convolving kernels produce band-pass filters. 

FIR is more performant but slower. 
"""

