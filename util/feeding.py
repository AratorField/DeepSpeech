# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial

import numpy as np
import pandas
import random
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from util.config import Config
from util.text import text_to_char_array

def spec_augment(mel_spectrogram, num_mask, time_warping_para=80, frequency_masking_para=27, time_masking_para=100):
    """Spec augmentation.
    Related paper : https://arxiv.org/pdf/1904.08779.pdf
    The Parameters, "Augmentation parameters for policies", refer to the 'Tabel 1' in the paper.
    Args:
      input: Extracted mel-spectrogram, numpy array.
      time_warping_para: "Augmentation parameters for policies W", LibriSpeech is 80.
      frequency_masking_para: "Augmentation parameters for policies F", LibriSpeech is 27.
      time_masking_para: "Augmentation parameters for policies T", LibriSpeech is 100.
      num_mask : number of masking lines.
    Returns:
      mel_spectrogram : warped and masked mel spectrogram.
    """

    """(TO DO)Time warping
    In paper Time warping written as follows. 'Given a log mel spectrogram with τ time steps,
    we view it as an image where the time axis is horizontal and the frequency axis is vertical.
    A random point along the horizontal line passing through the center of the image within the time steps (W, τ − W)
    is to be warped either to the left or right by a distance w chosen from a uniform distribution
    from 0 to the time warp parameter W along that line.'
    In paper Using Tensorflow's 'sparse-image-warp'.
    """
    tau = mel_spectrogram.shape[1]

    # Image warping control point setting
    control_point_locations = np.asarray([[64, 64], [64, 80]])  # pyformat: disable
    control_point_locations = constant_op.constant(
        np.float32(np.expand_dims(control_point_locations, 0)))

    control_point_displacements = np.ones(
        control_point_locations.shape.as_list())
    control_point_displacements = constant_op.constant(
        np.float32(control_point_displacements))

    # mel spectrogram data type convert to tensor constant for sparse_image_warp
    mel_spectrogram = mel_spectrogram.reshape([1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1])
    mel_spectrogram_op = constant_op.constant(np.float32(mel_spectrogram))
    w = random.randint(0, time_warping_para)

    (warped_mel_spectrogram_op, flow_field) = tf.contrib.image.sparse_image_warp(mel_spectrogram_op,
                                                                                 source_control_point_locations=control_point_locations,
                                                                                 dest_control_point_locations=control_point_locations + control_point_displacements,
                                                                                 interpolation_order=2,
                                                                                 regularization_weight=0,
                                                                                 num_boundary_points=0
                                                                                 )

    # Change data type of warp result to numpy array for masking step
    with tf.Session() as sess:
        warped_mel_spectrogram, _ = sess.run([warped_mel_spectrogram_op, flow_field])

    warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                             warped_mel_spectrogram.shape[2]])
    warped_masked_mel_spectrogram = warped_mel_spectrogram

    """ Masking line loop """
    for i in range(num_mask):
        """Frequency masking
        In paper Frequency masking written as follows. 'Frequency masking is applied so that f consecutive mel frequency
        channels [f0, f0 + f) are masked, where f is first chosen from a uniform distribution from 0 to the frequency mask parameter F,
        and f0 is chosen from [0, ν − f). ν is the number of mel frequency channels.'
        In this code, ν was written with v. ands For choesing 'f' uniform distribution, I using random package.
        """
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        v = 128  # Now hard coding but I will improve soon.
        f0 = random.randint(0, v - f)
        warped_masked_mel_spectrogram[f0:f0 + f, :] = 0

        """Time masking
        In paper Time masking written as follows. 'Time masking is applied so that t consecutive time steps
        [t0, t0 + t) are masked, where t is first chosen from a uniform distribution from 0 to the time mask parameter T,
        and t0 is chosen from [0, τ − t).'
        In this code, τ(tau) was written with tau. and For choesing 't' uniform distribution, I using random package.
        """
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_masked_mel_spectrogram[:, t0:t0 + t] = 0

    return warped_masked_mel_spectrogram, warped_mel_spectrogram

def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1))) # pylint: disable=cell-var-from-loop
        file['wav_filesize_original'] = file['wav_filesize']
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=Config.audio_window_samples,
                                                  stride=Config.audio_step_samples,
                                                  magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])
    if random.choice([True,False]):
        mfccs = spec_augment(mfccs, 1)
        
    return mfccs, tf.shape(mfccs)[0]


def audiofile_to_features(wav_filename):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)

    return features, features_len


def entry_to_features(wav_filename, transcript):
    # https://bugs.python.org/issue32117
    features, features_len = audiofile_to_features(wav_filename)
    return features, features_len, tf.SparseTensor(*transcript)


def to_sparse_tuple(sequence):
    r"""Creates a sparse representention of ``sequence``.
        Returns a tuple with (indices, values, shape)
    """
    indices = np.asarray(list(zip([0]*len(sequence), range(len(sequence)))), dtype=np.int64)
    shape = np.asarray([1, len(sequence)], dtype=np.int64)
    return indices, sequence, shape


def create_dataset(csvs, batch_size, cache_path=''):
    df = read_csvs(csvs)
    df.sort_values(by='wav_filesize', inplace=True)

    # Convert to character index arrays
    df['transcript'] = df['transcript'].apply(partial(text_to_char_array, alphabet=Config.alphabet))

    def generate_values():
        df['wav_filesize'] = df['wav_filesize_original'].apply(lambda x: x+int(x*random.uniform(-0.2, 0.2)))
        df.sort_values(by='wav_filesize', inplace=True)
        for _, row in df.iterrows():
            yield row.wav_filename, to_sparse_tuple(row.transcript)

    # Batching a dataset of 2D SparseTensors creates 3D batches, which fail
    # when passed to tf.nn.ctc_loss, so we reshape them to remove the extra
    # dimension here.
    def sparse_reshape(sparse):
        shape = sparse.dense_shape
        return tf.sparse.reshape(sparse, [shape[0], shape[2]])

    def batch_fn(features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size,
                                         padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size).map(sparse_reshape)
        return tf.data.Dataset.zip((features, transcripts))

    num_gpus = len(Config.available_devices)

    dataset = (tf.data.Dataset.from_generator(generate_values,
                                              output_types=(tf.string, (tf.int64, tf.int32, tf.int64)))
                              .map(entry_to_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .cache(cache_path)
                              .window(batch_size, drop_remainder=True).flat_map(batch_fn)
                              .prefetch(num_gpus))

    return dataset
