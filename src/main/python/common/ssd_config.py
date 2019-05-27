"""
    config for Mobilenet
"""
import os


def write_config(model_name, root_dir, num_classes):
    config = ("""
        model {
          ssd {
            num_classes: %(NUM_CLASSES)s
            box_coder {
              faster_rcnn_box_coder {
                y_scale: 10.0
                x_scale: 10.0
                height_scale: 5.0
                width_scale: 5.0
              }
            }
            matcher {
              argmax_matcher {
                matched_threshold: 0.5
                unmatched_threshold: 0.5
                ignore_thresholds: false
                negatives_lower_than_unmatched: true
                force_match_for_each_row: true
              }
            }
            similarity_calculator {
              iou_similarity {
              }
            }
            anchor_generator {
              ssd_anchor_generator {
                num_layers: 6
                min_scale: 0.2
                max_scale: 0.95
                aspect_ratios: 1.0
                aspect_ratios: 2.0
                aspect_ratios: 0.5
                aspect_ratios: 3.0
                aspect_ratios: 0.3333
              }
            }
            image_resizer {
              fixed_shape_resizer {
                height: 300
                width: 300
              }
            }
            box_predictor {
              convolutional_box_predictor {
                min_depth: 0
                max_depth: 0
                num_layers_before_predictor: 0
                use_dropout: false
                dropout_keep_probability: 0.8
                kernel_size: 3
                use_depthwise: true
                box_code_size: 4
                apply_sigmoid_to_scores: false
                conv_hyperparams {
                  activation: RELU_6,
                  regularizer {
                    l2_regularizer {
                      weight: 0.00004
                    }
                  }
                  initializer {
                    truncated_normal_initializer {
                      stddev: 0.03
                      mean: 0.0
                    }
                  }
                  batch_norm {
                    train: true,
                    scale: true,
                    center: true,
                    decay: 0.9997,
                    epsilon: 0.001,
                  }
                }
              }
            }
            feature_extractor {
              type: 'ssd_mobilenet_v2'
              min_depth: 16
              depth_multiplier: 1.0
              use_depthwise: true
              conv_hyperparams {
                activation: RELU_6,
                regularizer {
                  l2_regularizer {
                    weight: 0.00004
                  }
                }
                initializer {
                  truncated_normal_initializer {
                    stddev: 0.03
                    mean: 0.0
                  }
                }
                batch_norm {
                  train: true,
                  scale: true,
                  center: true,
                  decay: 0.9997,
                  epsilon: 0.001,
                }
              }
            }
            loss {
              classification_loss {
                weighted_sigmoid {
                }
              }
              localization_loss {
                weighted_smooth_l1 {
                }
              }
              hard_example_miner {
                num_hard_examples: 3000
                iou_threshold: 0.99
                loss_type: CLASSIFICATION
                max_negatives_per_positive: 3
                min_negatives_per_image: 3
              }
              classification_weight: 1.0
              localization_weight: 1.0
            }
            normalize_loss_by_num_matches: true
            post_processing {
              batch_non_max_suppression {
                score_threshold: 1e-8
                iou_threshold: 0.6
                max_detections_per_class: 100
                max_total_detections: 100
              }
              score_converter: SIGMOID
            }
          }
        }
    
        train_config: {
          batch_size: 24
          optimizer {
            rms_prop_optimizer: {
              learning_rate: {
                exponential_decay_learning_rate {
                  initial_learning_rate: 0.004
                  decay_steps: 800720
                  decay_factor: 0.95
                }
              }
              momentum_optimizer_value: 0.9
              decay: 0.9
              epsilon: 1.0
            }
          }
          fine_tune_checkpoint: "%(CHECKPOINT_FILE)s/model.ckpt"
          fine_tune_checkpoint_type:  "detection"
          # Note: The below line limits the training process to 200K steps, which we
          # empirically found to be sufficient enough to train the pets dataset. This
          # effectively bypasses the learning rate schedule (the learning rate will
          # never decay). Remove the below line to train indefinitely.
          num_steps: 200000
          data_augmentation_options {
            random_horizontal_flip {
            }
          }
          data_augmentation_options {
            ssd_random_crop {
            }
          }
        }
    
        train_input_reader: {
          tf_record_input_reader {
            input_path: "%(ANNOTATIONS_DIR)s/train.record"
          }
          label_map_path: "%(ANNOTATIONS_DIR)s/label_map.pbtxt"
        }
    
        eval_config: {
          num_examples: 8000
          # Note: The below line limits the evaluation process to 10 evaluations.
          # Remove the below line to evaluate indefinitely.
          max_evals: 10
        }
    
        eval_input_reader: {
          tf_record_input_reader {
            input_path: "%(ANNOTATIONS_DIR)s/test.record"
          }
          label_map_path: "%(ANNOTATIONS_DIR)s/label_map.pbtxt"
          shuffle: false
          num_readers: 1
        }
        """ % {
        'CHECKPOINT_FILE': os.path.join(root_dir, 'frozen_model', model_name),
        'ANNOTATIONS_DIR': os.path.join(root_dir, 'data_dir', 'annotations'),
        'NUM_CLASSES': num_classes
    }
              )
    config_path = os.path.join(root_dir, 'data_dir', 'tf_api.config')
    with open(config_path, 'w') as f:
        f.write(config)
        print("Config written to %s\nCheckpoints in %s\nTrain data from %s" % (
            config_path,
            os.path.join(root_dir, 'data_dir', 'checkpoints'),
            os.path.join(root_dir, 'data_dir', 'annotations'))
              )
