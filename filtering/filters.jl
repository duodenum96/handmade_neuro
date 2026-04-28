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

Moving average (MA) filters:
Reduce random noise while retaining a sharp step response. 
Worst filter for frequency domain. 
cf Gaussian, Blackman and multiple-pass MA filters, comp. more 
intensive but better at freq domain. 

y[i] = 1/M sum_{j=0}^{M-1} x[i+j]

this is an M-point MA filter. 

Alternative: y[i] = 1/M sum_{j=i - floor(M/2)}^{i + floor(M/2)} x[j]
(M is an odd number.)

Note that this is conv. with a rectangular kernel of area 1:
kernel: [0..., 1/M, 1/M, ..., 1/M, ..., 0]

For the problem of reducing fast noise while keeping sharp transient, 
this is the best tool (don't feel guilty). 
Amount of noise reduction ~ sqrt(M). 

Frequency response: fourier transform of rectangle: 
H[f] = sin(pi f M) / ( M sin(pi f) )

Multiple-pass MA filters: pass the input from the kernel multiple times. 
Two passes -> triangle (conv(rectangle, rectangle) = triangle). 
4 passes -> dome (getting closer to gaussian due to CLT). 

Gaussian filter -> Gaussian in the frequency domain (centered at 0 Hz)
Gaussian is IR of many systems. 

Blackman window -> looks like a gaussian

Bm and gaussian vs rectangle:
Have better frequency domain properties compared to gaussian. 
Taper a small amplitude at the ends. Edges have smaller effect. 
Step response is smooth. 
Rectangle is 10x faster

Recursive calculation algorithm:
consider 5-point filter
y[5] = x[3] + x[4] + x[5] + x[6] + x[7]
y[6] = x[4] + x[5] + x[6] + x[7] + x[8]

calculation of y[6] is very redundant. If y[5] is already calculated, then 
y[6] = y[5] - x[3] + x[8]
y[7] = y[6] - x[4] + x[9]
... (general formula):
y[n] = y[n-1] - x[n-q] + x[n+p]
p: (M-1) / 2
q: p + 1

Windowed-sinc filters
Seperate one band of frequencies from other
Very good frequency response, bad time response
(Excessive ripple, overshoot in the step response)
Ideal low pass filter (all frequencies below cutoff are 
passed with unity amplitude, higher frequencies blocked, 
passband is very flat, attenuation is infinite and the transition is 
very small)
sin: sin(x) / x
h[i] = sin(2 π f_c i) / (i π)
Length of sinc is infinite mathematically (it doesn't decay to zero)
No problem for math but problem for computers
Computerization: 
truncate to M+1 points (M is even),
all samples outside these are zero
Entire sequence is shifted s.t. it runs from 0 to M
computerized version won't have the ideal frequency response
Abrupt discontinuity at the ends (high time resolution) 
of the sinc result in messed up freq response with Gibbs ripples

Improving the situation: Blackman window. Multiply truncated sinc by 
Blackman to get windowed sinc. 

Hamming window: w[i] = 0.54 - 0.46cos(2πi/M)
Blackman window: w[i] = 0.42 - 0.5cos(2πi/M) + 0.08cos(4πi/M)

Blackman has slower roll-off but good stopband attenuation
Hamming has worse stopband but better roll-off

Generally Blackman is recommended since slow roll-off is easier to handle than 
poor stopband

Other windows:
Bartlett: triangle
Hanning: raised cosine: w[i] = 0.5 - 0.5cos(2πi/M)
These two windows have about the same roll-off speed as Hamming but worse 
stopband attenuation
rectangular window = no window (just truncation of tails)

Design of windowed sinc: 
two parameters of sinc: f_c, M

M sets the roll-off: M ≈ 4/BW
f_c is expressed as a fraction of sampling rate and must be between 
0 and 0.5 (nyq)
BW: bandwidth of the filter (width of the transition band)

Higher the M, lower the width. BW is also expressed as a fraction of 
fs so it must be between 0 and 0.5. 
Tradeoff between computation time (~M) and filter sharpness (~BW). 

filter kernel: 
h[i] = K * sin(2π f_c (i-M/2)) / (i-M/2) [
    0.42 - 0.5cos(2πi/M) + 0.08cos(4πi/M)
] for i ∈ 0:M (M+1 total points)

= K * Blackman(i-M/2, M) * sinc(M)

K is chosen to provide unity gain at zero frequency.
To avoid divide-by-zero, for i=M/2, h[i]=2π f_c K

Terrible at time domain, leads to overshoot and ringing

it is easy to design extreme filters. any filter convolved with 
itself (or another filter) can generate the output of 
filtered twice situation. if you want -148dB stopband attenuation, 
Blackman provides only -74dB. Passing through Blackman twice will give 
-148. Price you pay is longer kernel and slower roll-off.

Custom filters: 
Filters with arbitrary frequency response (not limited to bandpass 
bandstop lowpass highpass)

Deconvolution: filtering a signal to compensate for an undesired convolution

overlap-add method: 1) decompose the signal into simple components, 
2) process each of the components 3) recombine processed signals

When an N-sample signal is convolved with M sample filter, the output 
is N+M-1 samples long. The N-sample signal will be expanded by M-1 
points to the right. 

FFT convolution: take fourier of the signal and kernel, multiply, 
do inverse fourier. This result was known since Fourier, but 
wasn't used until FFT since DFT was slower than overlap-add method.

FFTs must be long enough so that circular convolution doesn't take 
place => FFT need be the same length as output segment

example fftconv: 10 million point signal, 400 point kernel. 
Break the signal into 16000 segments, each having 625 points. 
Each segment is conv'd with kernel: output: 625 + 400 - 1=1024 pts

recursive filters aka IIR filters: IR is composed of 
decaying exponentials

FIR filter: y[n] = a[0]*x[n] + a[1]*x[n-1] + ... 
IIR filter: y[n] = a[0]*x[n] + a[1]*x[n-1] + ... + b[1]*y[n-1] + 
                    b[2]*y[n-2] + ...

In practice no more than a dozen recursion coefficients can be 
used or the filter becomes unstable. 

The relationship between recursion coefficients and filter response 
is given by z transform (ch 33 in dspguide). 

Three ways to find the recursion coefficients w/o z-transform. 
ch19: design eqs. ch20: cookbook program for chebyshev filters. 
ch26: iterative method for designing recursive filters w/ arbitrary 
freq response. 

Single pole low-pass: a[0] = 1-x, b[1] = x
Single pole high-pass: a[0] = (1+x)/2, a[1] = -(1+x)/2, b[1]=x
0 < x < 1: decay between adjacent samples. Higher the x, slower 
the decay

x can be found from the desired time constant of the filter. 
Related to RC circuits: R*C is the time it takes to reach 1/e of its
final value, d is the number of samples to reach 1/d:
x = exp(-1/d)

Another relationship with f_c:
x = exp(-2π f_c)
"""

