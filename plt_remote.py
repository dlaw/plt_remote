import numpy, scipy.signal
import SoapySDR

def freq_shift(data, shift):
    return data * numpy.exp(2j * numpy.pi * numpy.arange(len(data)) * shift)

class QuadratureModulator:
    def __init__(self, sample_rate, radio_freq, signal_freq, mod_freq, bw):
        self.freq = (signal_freq - radio_freq) / sample_rate
        self.mod_scale = mod_freq / sample_rate
        filter_len = 1 + 2 * int(sample_rate // bw)
        self.filt = scipy.signal.firwin(filter_len, bw, fs=sample_rate)
    def demodulate(self, samples):
        # Input: raw I and Q samples
        # Output: frequency deviation from the carrier frequency
        baseband = freq_shift(samples, -self.freq)
        baseband_filt = numpy.convolve(baseband, self.filt, 'valid')
        phase = numpy.unwrap(numpy.angle(baseband_filt)) / (2 * numpy.pi)
        return numpy.diff(phase) / self.mod_scale
    def modulate(self, samples):
        # Input: frequency deviation from the carrier frequency
        # Output: raw I and Q samples
        phase = numpy.cumsum(numpy.pad(samples, (1, 0))) * self.mod_scale
        baseband = numpy.exp(2j * numpy.pi * phase)
        baseband_filt = numpy.convolve(baseband, self.filt, 'full')
        return freq_shift(baseband_filt, self.freq)

class BitSampler:
    def __init__(self, sample_rate, symbol_freq):
        self.rate = symbol_freq / sample_rate
    def decode(self, samples):
        # Assumes that the data bits are all represented by nonzero samples,
        # so that the input samples are only close to 0 on bit transitions.
        # This allows phase lag to be computed with the Fourier transform.
        bit_num = numpy.arange(len(samples)) * self.rate
        ref_signal = numpy.exp(2j * numpy.pi * bit_num)
        phase_lag = numpy.angle(numpy.dot(numpy.abs(samples), ref_signal))
        sample_offset = ((phase_lag / 2 / numpy.pi) % 1.0) / self.rate
        indices = numpy.arange(sample_offset, len(samples), 1 / self.rate)
        return numpy.array(samples)[indices.astype('int')]
    def encode(self, bits):
        # Typically `bits` would be an array whose elements are -1 or +1.
        # However, since they represent a frequency shift, other values
        # do have physical meaning. For example, a bit value of 0
        # indicates an unshifted carrier.
        bit_num = numpy.arange(len(bits) / self.rate) * self.rate
        return numpy.array(bits)[bit_num.astype('int')]

class SoapyRadio:
    def __init__(self, sample_rate, radio_freq, **kwargs):
        self.sample_rate = sample_rate
        self.radio_freq = radio_freq
        self.sdr = SoapySDR.Device(kwargs)
    def listen(self, total_samples, min_signal=10, min_length=10000):
        # Receive `total_samples` samples, and return any continous runs of samples
        # with magnitude greater than `min_signal` which are longer than `min_length`.
        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.radio_freq)
        rx = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CS8)
        self.sdr.activateStream(rx)
        raw = numpy.zeros((min(1024, min_length), 2), 'int8')
        chunks = []
        current_chunk = []
        for i in range(total_samples // len(raw)):
            result = self.sdr.readStream(rx, [raw], len(raw))
            assert result.ret == len(raw), result.ret
            power = numpy.sum(numpy.square(raw, dtype='int16'), axis=1)
            is_valid = (power > numpy.square(min_signal))
            if numpy.all(is_valid):
                current_chunk.append(raw.dot([1, 1j]))
            else:   # current chunk, if any, will be ending
                if is_valid[0]:
                    current_chunk.append(raw[:numpy.argmin(is_valid)].dot([1, 1j]))
                if sum([len(c) for c in current_chunk]) >= min_length:
                    chunks.append(numpy.concatenate(current_chunk))
                del current_chunk[:]
                if is_valid[-1]:  # next chunk is starting
                    current_chunk.append(raw[-numpy.argmin(is_valid[::-1]):].dot([1, 1j]))
        self.sdr.deactivateStream(rx)
        self.sdr.closeStream(rx)
        return chunks
    def transmit(self, messages, blanking_length=10000, tx_gain=20):
        # Transmit each message in messages, separated by `blanking_length` zero samples.
        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, self.sample_rate)
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, self.radio_freq)
        self.sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, tx_gain)
        tx = self.sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CS8)
        self.sdr.activateStream(tx)
        blank = numpy.zeros((blanking_length, 2), 'int8')
        self.sdr.writeStream(tx, [blank], len(blank), timeoutUs=1000000)
        for message in messages:
            output = numpy.zeros((len(message), 2), 'int8')
            scale = 127. / numpy.amax(numpy.abs(message))
            output[:, 0] = numpy.around(numpy.real(message * scale))
            output[:, 1] = numpy.around(numpy.imag(message * scale))
            self.sdr.writeStream(tx, [output], len(output), timeoutUs=1000000)
            self.sdr.writeStream(tx, [blank], len(blank), timeoutUs=1000000)
        self.sdr.readStreamStatus(tx, timeoutUs=1000000)  # wait for tx
        self.sdr.deactivateStream(tx)
        self.sdr.closeStream(tx)
