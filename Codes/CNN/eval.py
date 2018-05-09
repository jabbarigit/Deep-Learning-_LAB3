#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from CNN import TextCNN
from tensorflow.contrib import learn
import csv

# all parameters

tf.flags.DEFINE_string("data_sets", "./data/datasets/consumer_complaints3.txt", "Data source for the consumer_comp data.")
# for eval 

tf.flags.DEFINE_integer("batch_size", 64, "Batch_Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint_directory")
tf.flags.DEFINE_boolean("eval_train", False, "training data evaluate")

# for misc
tf.flags.DEFINE_boolean("allow_soft_placement", True, "allowing the soft device for a placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "option for devices logs for a placemen")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


if FLAGS.eval_train:
 
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_sets)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# vocabulary to data mapping
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# here is the section for evaluation

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
       
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

       
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

       
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

       
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# saving__evaluations
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)