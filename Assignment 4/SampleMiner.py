class HardExampleMiner(object):
  """Hard example mining for regions in a list of images.
  Implements hard example mining to select a subset of regions to be
  back-propagated. For each image, selects the regions with highest losses,
  subject to the condition that a newly selected region cannot have
  an IOU > iou_threshold with any of the previously selected regions.
  This can be achieved by re-using a greedy non-maximum suppression algorithm.
  A constraint on the number of negatives mined per positive region can also be
  enforced.
  Reference papers: "Training Region-based Object Detectors with Online
  Hard Example Mining" (CVPR 2016) by Srivastava et al., and
  "SSD: Single Shot MultiBox Detector" (ECCV 2016) by Liu et al.
  """

  def __init__(self,
               num_hard_examples=64,
               iou_threshold=0.7,
               loss_type='both',
               cls_loss_weight=0.05,
               loc_loss_weight=0.06,
               max_negatives_per_positive=None,
               min_negatives_per_image=0):
    """Constructor.
    The hard example mining implemented by this class can replicate the behavior
    in the two aforementioned papers (Srivastava et al., and Liu et al).
    To replicate the A2 paper (Srivastava et al), num_hard_examples is set
    to a fixed parameter (64 by default) and iou_threshold is set to .7 for
    running non-max-suppression the predicted boxes prior to hard mining.
    In order to replicate the SSD paper (Liu et al), num_hard_examples should
    be set to None, max_negatives_per_positive should be 3 and iou_threshold
    should be 1.0 (in order to effectively turn off NMS).
    Args:
      num_hard_examples: maximum number of hard examples to be
        selected per image (prior to enforcing max negative to positive ratio
        constraint).  If set to None, all examples obtained after NMS are
        considered.
      iou_threshold: minimum intersection over union for an example
        to be discarded during NMS.
      loss_type: use only classification losses ('cls', default),
        localization losses ('loc') or both losses ('both').
        In the last case, cls_loss_weight and loc_loss_weight are used to
        compute weighted sum of the two losses.
      cls_loss_weight: weight for classification loss.
      loc_loss_weight: weight for location loss.
      max_negatives_per_positive: maximum number of negatives to retain for
        each positive anchor. By default, num_negatives_per_positive is None,
        which means that we do not enforce a prespecified negative:positive
        ratio.  Note also that num_negatives_per_positives can be a float
        (and will be converted to be a float even if it is passed in otherwise).
      min_negatives_per_image: minimum number of negative anchors to sample for
        a given image. Setting this to a positive number allows sampling
        negatives in an image without any positive anchors and thus not biased
        towards at least one detection per image.
    """
    self._num_hard_examples = num_hard_examples
    self._iou_threshold = iou_threshold
    self._loss_type = loss_type
    self._cls_loss_weight = cls_loss_weight
    self._loc_loss_weight = loc_loss_weight
    self._max_negatives_per_positive = max_negatives_per_positive
    self._min_negatives_per_image = min_negatives_per_image
    if self._max_negatives_per_positive is not None:
      self._max_negatives_per_positive = float(self._max_negatives_per_positive)
    self._num_positives_list = None
    self._num_negatives_list = None

  def __call__(self,
               location_losses,
               cls_losses,
               decoded_boxlist_list,
               match_list=None):
    """Computes localization and classification losses after hard mining.
    Args:
      location_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise localization losses.
      cls_losses: a float tensor of shape [num_images, num_anchors]
        representing anchorwise classification losses.
      decoded_boxlist_list: a list of decoded BoxList representing location
        predictions for each image.
      match_list: an optional list of matcher.Match objects encoding the match
        between anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.  Match objects in match_list are
        used to reference which anchors are positive, negative or ignored.  If
        self._max_negatives_per_positive exists, these are then used to enforce
        a prespecified negative to positive ratio.
    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    Raises:
      ValueError: if location_losses, cls_losses and decoded_boxlist_list do
        not have compatible shapes (i.e., they must correspond to the same
        number of images).
      ValueError: if match_list is specified but its length does not match
        len(decoded_boxlist_list).
    """
    mined_location_losses = []
    mined_cls_losses = []
    location_losses = tf.unstack(location_losses)
    cls_losses = tf.unstack(cls_losses)
    num_images = len(decoded_boxlist_list)
    if not match_list:
      match_list = num_images * [None]
    if not len(location_losses) == len(decoded_boxlist_list) == len(cls_losses):
      raise ValueError('location_losses, cls_losses and decoded_boxlist_list '
                       'do not have compatible shapes.')
    if not isinstance(match_list, list):
      raise ValueError('match_list must be a list.')
    if len(match_list) != len(decoded_boxlist_list):
      raise ValueError('match_list must either be None or have '
                       'length=len(decoded_boxlist_list).')
    num_positives_list = []
    num_negatives_list = []
    for ind, detection_boxlist in enumerate(decoded_boxlist_list):
      box_locations = detection_boxlist.get()
      match = match_list[ind]
      image_losses = cls_losses[ind]
      if self._loss_type == 'loc':
        image_losses = location_losses[ind]
      elif self._loss_type == 'both':
        image_losses *= self._cls_loss_weight
        image_losses += location_losses[ind] * self._loc_loss_weight
      if self._num_hard_examples is not None:
        num_hard_examples = self._num_hard_examples
      else:
        num_hard_examples = detection_boxlist.num_boxes()
      selected_indices = tf.image.non_max_suppression(
          box_locations, image_losses, num_hard_examples, self._iou_threshold)
      if self._max_negatives_per_positive is not None and match:
        (selected_indices, num_positives,
         num_negatives) = self._subsample_selection_to_desired_neg_pos_ratio(
             selected_indices, match, self._max_negatives_per_positive,
             self._min_negatives_per_image)
        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)
      mined_location_losses.append(
          tf.reduce_sum(tf.gather(location_losses[ind], selected_indices)))
      mined_cls_losses.append(
          tf.reduce_sum(tf.gather(cls_losses[ind], selected_indices)))
    location_loss = tf.reduce_sum(tf.stack(mined_location_losses))
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses))
    if match and self._max_negatives_per_positive:
      self._num_positives_list = num_positives_list
      self._num_negatives_list = num_negatives_list
    return (location_loss, cls_loss)

  def summarize(self):
    """Summarize the number of positives and negatives after mining."""
    if self._num_positives_list and self._num_negatives_list:
      avg_num_positives = tf.reduce_mean(tf.to_float(self._num_positives_list))
      avg_num_negatives = tf.reduce_mean(tf.to_float(self._num_negatives_list))
      tf.summary.scalar('HardExampleMiner/NumPositives', avg_num_positives)
      tf.summary.scalar('HardExampleMiner/NumNegatives', avg_num_negatives)

  def _subsample_selection_to_desired_neg_pos_ratio(self,
                                                    indices,
                                                    match,
                                                    max_negatives_per_positive,
                                                    min_negatives_per_image=0):
    """Subsample a collection of selected indices to a desired neg:pos ratio.
    This function takes a subset of M indices (indexing into a large anchor
    collection of N anchors where M<N) which are labeled as positive/negative
    via a Match object (matched indices are positive, unmatched indices
    are negative).  It returns a subset of the provided indices retaining all
    positives as well as up to the first K negatives, where:
      K=floor(num_negative_per_positive * num_positives).
    For example, if indices=[2, 4, 5, 7, 9, 10] (indexing into 12 anchors),
    with positives=[2, 5] and negatives=[4, 7, 9, 10] and
    num_negatives_per_positive=1, then the returned subset of indices
    is [2, 4, 5, 7].
    Args:
      indices: An integer tensor of shape [M] representing a collection
        of selected anchor indices
      match: A matcher.Match object encoding the match between anchors and
        groundtruth boxes for a given image, with rows of the Match objects
        corresponding to groundtruth boxes and columns corresponding to anchors.
      max_negatives_per_positive: (float) maximum number of negatives for
        each positive anchor.
      min_negatives_per_image: minimum number of negative anchors for a given
        image. Allow sampling negatives in image without any positive anchors.
    Returns:
      selected_indices: An integer tensor of shape [M'] representing a
        collection of selected anchor indices with M' <= M.
      num_positives: An integer tensor representing the number of positive
        examples in selected set of indices.
      num_negatives: An integer tensor representing the number of negative
        examples in selected set of indices.
    """
    positives_indicator = tf.gather(match.matched_column_indicator(), indices)
    negatives_indicator = tf.gather(match.unmatched_column_indicator(), indices)
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
    max_negatives = tf.maximum(min_negatives_per_image,
                               tf.to_int32(max_negatives_per_positive *
                                           tf.to_float(num_positives)))
    topk_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, topk_negatives_indicator))
    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return (tf.reshape(tf.gather(indices, subsampled_selection_indices), [-1]),
            num_positives, num_negatives)