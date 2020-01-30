# coding=utf-8
# Copyright 2019 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trax trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags
from absl import logging

import gin
import jax
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()
# tf.compat.v1.enable_eager_execution()
import trax
from trax import math
from trax.supervised import trainer_lib
from trax.tf_numpy import numpy as tf_np
from trax.models.transformer import Transformer

from trax import history as trax_history
from trax import jaxboard
from trax import layers as tl
from trax import learning_rate as lr
from trax import math
from trax import optimizers as trax_opt
from trax.math import numpy as np
from trax.math import random as jax_random
from trax.shapes import ShapeDtype
from trax.supervised import inputs as trax_inputs
from trax.search import transformer_greedy, transformer_batch

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
import sentencepiece as sp
import tensorflow_text as tf_text

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Which dataset to use.')
flags.DEFINE_string('model', None, 'Which model to train.')
flags.DEFINE_string('data_dir', None, 'Path to the directory with data.')
flags.DEFINE_string('output_dir', None,
                    'Path to the directory to save logs and checkpoints.')
flags.DEFINE_multi_string('config_file', None,
                          'Configuration file with parameters (.gin).')
flags.DEFINE_multi_string('config', None,
                          'Configuration parameters (gin string).')
flags.DEFINE_integer('log_level', logging.INFO, 'Log level.')
# TPU Flags
flags.DEFINE_bool('use_tpu', False, "Whether we're running on TPU.")
flags.DEFINE_string(
    'jax_xla_backend', 'xla',
    'Either "xla" for the XLA service directly, or "tpu_driver"'
    'for a TPU Driver backend.')
flags.DEFINE_string('jax_backend_target', 'local',
                    'Either "local" or "rpc:address" to connect to a '
                    'remote service target.')

# TensorFlow Flags
flags.DEFINE_bool('enable_eager_execution', True,
                  "Whether we're running TF in eager mode.")
flags.DEFINE_bool('tf_xla', True, 'Whether to turn on XLA for TF.')
flags.DEFINE_bool('tf_opt_pin_to_host', False, 'Whether to turn on TF '
                  'pin-to-host optimization.')
flags.DEFINE_bool('tf_opt_layout', False, 'Whether to turn on TF layout '
                  'optimization.')
flags.DEFINE_bool('tf_xla_forced_compile', False, 'Use forced-compilation '
                  'instead of auto-clustering for XLA. This flag only has '
                  'effects when --tf_xla is on.')
flags.DEFINE_bool('tf_allow_float64', False, 'Whether to allow float64 for TF.')


def _tf_setup_from_flags():
  """Processes TensorFlow-relevant flags."""
  if FLAGS.enable_eager_execution:
    tf.compat.v1.enable_eager_execution()
  if FLAGS.tf_xla:
    tf.config.optimizer.set_jit(True)
    math.tf_math.set_tf_xla_forced_compile(FLAGS.tf_xla_forced_compile)
  tf.config.optimizer.set_experimental_options({
      'pin_to_host_optimization': FLAGS.tf_opt_pin_to_host,
      'layout_optimizer': FLAGS.tf_opt_layout,
  })
  tf_np.set_allow_float64(FLAGS.tf_allow_float64)


def _gin_parse_configs():
  """Initializes gin-controlled bindings."""
  # Imports for configurables
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable
  from trax import models as _trax_models
  from trax import optimizers as _trax_opt
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

  configs = FLAGS.config or []
  # Override with --dataset and --model
  if FLAGS.dataset:
    configs.append("inputs.dataset_name='%s'" % FLAGS.dataset)
    if FLAGS.data_dir:
      configs.append("inputs.data_dir='%s'" % FLAGS.data_dir)
  if FLAGS.model:
    configs.append('train.model=@trax.models.%s' % FLAGS.model)
  gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def _output_dir_or_default():
  """Returns a path to the output directory."""
  if FLAGS.output_dir:
    output_dir = FLAGS.output_dir
    trainer_lib.log('Using --output_dir {}'.format(output_dir))
    return os.path.expanduser(output_dir)

  # Else, generate a default output dir (under the user's home directory).
  try:
    dataset_name = gin.query_parameter('inputs.dataset_name')
  except ValueError:
    dataset_name = 'random'
  output_name = '{model_name}_{dataset_name}_{timestamp}'.format(
      model_name=gin.query_parameter('train.model').configurable.name,
      dataset_name=dataset_name,
      timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
  )
  output_dir = os.path.join('~', 'trax', output_name)
  output_dir = os.path.expanduser(output_dir)
  print()
  trainer_lib.log('No --output_dir specified')
  trainer_lib.log('Using default output_dir: {}'.format(output_dir))
  return output_dir


def _jax_and_tf_configure_for_devices():
  if FLAGS.use_tpu:
    jax.config.update('jax_platform_name', 'tpu')
    jax.config.update('jax_xla_backend', FLAGS.jax_xla_backend)
    jax.config.update('jax_backend_target', FLAGS.jax_backend_target)
  if FLAGS.enable_eager_execution and math.backend_name() in ('numpy', 'jax'):
    # Numpy backend doesn't benefit from having the input pipeline run on GPU,
    # and jax backend has GPU memory contention if TF uses the GPU. Gin must be
    # set up first before determining the backend.
    tf.config.experimental.set_visible_devices([], 'GPU')


def _train_using_tf(output_dir):
  worker_cpu = tf_init_tpu()
  with tf.device(worker_cpu):
    if trainer_lib.num_devices() == 1:
      # TF's device priority is GPU > CPU > TPU, so we need to explicitly make
      # the TPU core the default device here.
      with tf.device('/device:TPU:0'):
        trainer_lib.train(output_dir=output_dir)
    else:
      trainer_lib.train(output_dir=output_dir)


@gin.configurable
def tf_init_tpu(worker='', protocol=None):
  """Initializes TPU for TensorFlow.

  Args:
    worker: The BNS address of the remote TPU worker. If it's empty (the default
      value), TF will assume the TPU devices are connected to the local host.
    protocol: The network protocol used to connect to the TPU worker.
  Returns:
    The device name of the TPU worker's CPU.
  """
  protocol = protocol or 'grpc'
  is_local = (worker in ('', 'local'))
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=worker)
  if not is_local:
    tf.config.experimental_connect_to_cluster(resolver, protocol=protocol)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  if is_local:
    return ''
  else:
    return '/job:worker'

def prep_data(dataset):
  # TODO return to this
  sp_model = tf.io.gfile.GFile(DEFAULT_SPM_PATH, "rb").read()
  spm = sp.SentencePieceProcessor()
  spm = spm.load_from_serialized_proto(sp_model)
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
  # spm.decode_ids([20,13,4,5,5])

  final_data = []
  for count, ex in enumerate(dataset):
    if count > 2:
      break
    if count % 100 == 0:
      print(count)
    src = ex['article']
    tgt = ex['highlights']
    # spm.EncodeAsIds("This is a string")
    tmp = tf.cast(tokenizer.tokenize(src), tf.int64)
    tmp = tf.slice(tmp, [0], [tf.minimum(tf.shape(tmp)[0], 512-1)])
    tmp = tf.concat([tmp, [1]], 0)
    tmp = tf.keras.preprocessing.sequence.pad_sequences([tmp], maxlen=512)
    final_data.append((tmp, tgt))
    # tmp = tmp.numpy()
    # test = np.zeroes((1,512), np.int64)
    # test[0, :] = tmp

    # yield (tmp, tgt)
  return final_data

def main(_):

  logging.set_verbosity(FLAGS.log_level)

  _tf_setup_from_flags()
  _gin_parse_configs()
  _jax_and_tf_configure_for_devices()

  output_dir = _output_dir_or_default()

  # get tokenizer
  sp_model = tf.io.gfile.GFile(DEFAULT_SPM_PATH, "rb").read()
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)

  # prepare model(s)
  trainer, _, model_eval = trainer_lib.eval(output_dir=output_dir)

  # jit_model_infer = trax.layers.base._accelerate(
  #   model_predict._forward_internal, trax.math.device_count())
  # # Set up the initial state for sampling.
  # infer_state = model_predict.new_weights_and_state(
  #   (trax.supervised.trainer_lib.ShapeDtype((1,512), dtype=np.int32), trax.supervised.trainer_lib.ShapeDtype((1,1), dtype=np.int32)))[1]
  # infer_state = trainer._for_n_devices(infer_state)
  # model_weights = trainer._opt_state[0][0]
  # cur_state = infer_state
  # predict_signature_src = trax.shapes.ShapeDtype((1, 512), dtype=np.int32)
  # predict_signature_tgt = trax.shapes.ShapeDtype((1, 1), dtype=np.int32)
  # model_predict.init((predict_signature_src, predict_signature_tgt))
  # model_predict.init_from_file(os.path.join(output_dir, "model.pkl"),
  #                            weights_only=True)
  eval_signature_src = trax.shapes.ShapeDtype((1, 512), dtype=np.int32)
  eval_signature_tgt = trax.shapes.ShapeDtype((1, 100), dtype=np.int32)
  model_eval.init((eval_signature_src, eval_signature_tgt))
  model_eval.init_from_file(os.path.join(output_dir, "model.pkl"),
                             weights_only=True)
  # sample input
  batch = next(trainer._eval_stream)
  # transformer_greedy(model_eval, batch, max_output_length=100)
  transformer_batch(model_eval, batch, size=5, max_output_length=100)
#   src = batch[0]
#   # cur_input_predict = np.array([[0]]) # bos 
#   cur_input_eval = np.array([[0]]) # bos 
#   import pdb;pdb.set_trace()
#   output_predict = ""
#   output_eval = ""
#   for i in range(25):
#     # run model
#     # out_pred = model_predict((src, cur_input_predict), state=i)
#     # # out_pred = model_predict((src, cur_input_predict))
#     # out_pred, cur_state = jit_model_infer(
#     #     (src, cur_input_predict),
#     #     weights=model_weights,
#     #     state=cur_state,
#     #     rng=trainer._rngs[0])
#     out_eval = model_eval((src, cur_input_eval), rng=trainer._rngs[0])

#     # get argmax prediction and detokenize
#     # last_index_pred = np.argmax(out_pred[0][0], axis=1).copy()[-1]
#     cur_tok_pred = tokenizer.detokenize([last_index_pred]).numpy().decode()
#     # cur_tok_pred = str(last_index_pred)
#     # output_predict += " " + cur_tok_pred
#     last_index_eval = np.argmax(out_eval[0][0], axis=1).copy()[-1]
#     cur_tok_eval = tokenizer.detokenize([last_index_eval]).numpy().decode()
#     # cur_tok_eval = str(last_index_eval)
#     # output_eval += " " + cur_tok_eval

#     # print("output predict: ", output_predict)
#     print("output eval: ", output_eval)
#     # set input for the next iteration
#     # cur_input_predict = np.array([[last_index_pred]]) 
#     if i == 0:
#       cur_input_eval = np.array([[last_index_eval]]) 
#     else:
#       # cur_input = cur_input + np.array([list(cur_input[0].copy()) + [last_index]]) 
#       cur_input_eval = np.array([list(cur_input_eval[0].copy()) + [last_index_eval]])
#     # # cur_input = np.reshape(indices, (1, indices.shape[0]))
#   print("output predict: {output_predict}")
#   print("output eval: {output_eval}")
#   exit()

# # indices = np.argmax(out[0][0], axis=1)
# # tokenizer.detokenize(indices.copy())
# # <tf.Tensor: id=640, shape=(), dtype=string, numpy=b"insured rate dropped dropped from 20.3 percent 
# # to 13.2 percent between theobtober   hbamacare  enrollmentendforcess .  barobama campaigned in 
# # the promise of a medical insurance law 'that will cover every american' and hhs secretary saidd 
# # that   be toa  subsidies.reout avenue a number of avenues' under house    for the  premium">


#   # dataset = tfds.load(name="cnn_dailymail", split="test")
#   # np_dataset = tfds.as_numpy(dataset)
#   # dataset_final = prep_data(np_dataset)
#   # max_target_len = 100

#   # output = ""
  
#   # for data in dataset_final:
#   #   src = data[0]
#   #   tgt_str = data[1]
#   #   cur_input = np.array([[100]])
#   #   for i in range(max_target_len):
#   #     print(i)
#   #     # print("about to run")
#   #     out = model((src, cur_input))
#   #     # take argmax, use as input to the next
#   #     # TODO check why we need to have [[0]] and not [0]
#   #     copy_ar = out[0].copy()
#   #     argmax_ = np.argmax(copy_ar)
#   #     # cur_tok = tokenizer.id_to_string(argmax_).numpy().decode()
#   #     cur_tok = tokenizer.id_to_string(argmax_.copy().item()).numpy().decode()
#   #     print(cur_tok)
#   #     output += cur_tok + " "
#   #     cur_input = np.array([[argmax_]])
#   #   print(output)
#   #   exit()

#   #   # print(data)


#   # # if FLAGS.use_tpu and math.backend_name() == 'tf':
#   # #   _train_using_tf(output_dir)
#   # # else:

#   # cur_input = np.array([[0]])
#   # cur_input_src = np.array([[0, 10]])
#   # test = (cur_input_src, cur_input)
#   # out = model(test)
#   # print(out[1].shape)

if __name__ == '__main__':
  # print(tf.executing_eagerly())
  # exit()

  app.run(main)