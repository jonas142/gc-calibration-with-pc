####### noisy angular velocity ########
# The HDC network wasn't found to be less sensitive to any type of noise, thus noise isn't included in interactive parameter selection.
use_noisy_av = False

# gaussian noise
# relative standard deviation (standard deviation = rel. sd * av)
noisy_av_rel_sd = 0.0
# absolute standard deviation (deg)
noisy_av_abs_sd = 0.0

# noise spikes
# average noise spike frequency in Hz
noisy_av_spike_frequency = 15.0
# average magnitude in deg/s
noisy_av_spike_magnitude = 50.0
# standard deviation in deg/s
noisy_av_spike_sd = 50.0

# noise oscillation
noisy_av_osc_frequency = 0.0
noisy_av_osc_magnitude = 0.0
noisy_av_osc_phase = 0.0
#######################################