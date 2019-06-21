#!/usr/bin/env python3
#
# Count number of each attribute occurences in
# ground-truth data
#

import re
import sys


labels = ['Female', 'AgeLess16', 'Age17-30', 'Age31-45', 'BodyFat', 'BodyNormal', 'BodyThin', 'Customer', 'Clerk', 'hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses', 'hs-Muffler', 'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers', 'shoes-Leather', 'shoes-Sport', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'attach-Backpack', 'attach-SingleShoulderBag', 'attach-HandBag', 'attach-Box', 'attach-PlasticBag', 'attach-PaperBag', 'attach-HandTrunk', 'attach-Other', 'action-Calling', 'action-Talking', 'action-Gathering', 'action-Holding', 'action-Pusing', 'action-Pulling', 'action-CarrybyArm', 'action-CarrybyHand', 'faceFront', 'faceBack', 'faceLeft', 'faceRight', 'occlusionLeft', 'occlusionRight', 'occlusionUp', 'occlusionDown', 'occlusion-Environment', 'occlusion-Attachment', 'occlusion-Person', 'occlusion-Other', 'up-Black', 'up-White', 'up-Gray', 'up-Red', 'up-Green', 'up-Blue', 'up-Yellow', 'up-Brown', 'up-Purple', 'up-Pink', 'up-Orange', 'up-Mixture', 'low-Black', 'low-White', 'low-Gray', 'low-Red', 'low-Green', 'low-Blue', 'low-Yellow', 'low-Mixture', 'shoes-Black', 'shoes-White', 'shoes-Gray', 'shoes-Red', 'shoes-Green', 'shoes-Blue', 'shoes-Yellow', 'shoes-Brown', 'shoes-Mixture']


# Count the number of lines containing
# the string
# filename: file name
# s: string
def count_lines_with(filename, s):
  count = 0
  with open(filename, 'r') as f:
    for line in f:
         if s in line:
             count += 1
  return count


if __name__ == "__main__":

  # initialze
  training = {}
  training['filename'] = 'training.dataset.ground_truth'

  validation = {}
  validation['filename'] = 'validation.dataset.ground_truth'

  # count
  for dataset in training, validation:
    # count total number of data 
    dataset['total'] = count_lines_with(dataset['filename'], labels[0])

    # count attributes occurences
    for label in labels:
      dataset[label] = count_lines_with(dataset['filename'], label + ' 1')

  # display result
  print('                               train    train%  dist-to-50%    valid   valid%')
  for label in labels:
    print('%24s' % label, end='')
    print('%14s' % (str(training[label]) + '/' + str(training['total'])), end='')
    train_percent = training[label]*100/training['total']
    dist_to_50 = abs(50 - train_percent)
    print('%8.1f' % (train_percent), end='')
    print('%8.1f' % (dist_to_50), end='')
    print('%14s' % (str(validation[label]) + '/' + str(validation['total'])), end='')
    print('%8.1f' % (validation[label]*100/validation['total']), end='')
    print()
