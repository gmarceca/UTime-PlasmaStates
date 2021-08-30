"""
Script which predicts on a set of data and evaluates the performance by
comparing to the ground truth labels.
"""

import os
from argparse import ArgumentParser
import readline
import numpy as np

readline.parse_and_bind('tab: complete')


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Evaluate a U-Time model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to U-Time project folder')
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--channels", nargs='*', type=str, default=None,
                        help="A list of channels to use instead of those "
                             "specified in the parameter file.")
    parser.add_argument("--one_shot", action="store_true",
                        help="Segment each SleepStudy in one forward-pass "
                             "instead of using (GPU memory-efficient) sliding "
                             "window predictions.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder')
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save prediction files")
    parser.add_argument("--no_save_true", action="store_true",
                        help="Save the true hypnogram in addition to the "
                             "predicted hypnogram. Ignored with --no_save.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--data_split", type=str, default="test_data",
                        help="Which split of data of those stored in the "
                             "hparams file should the evaluation be performed "
                             "on.")
    parser.add_argument("--plot_hypnograms", action="store_true",
                        help="Add plots comparing the predicted versus true"
                             " hypnograms to folder [out_dir]/plots/hypnograms.")
    parser.add_argument("--plot_CMs", action="store_true",
                        help="Add plots showing per-sample confusion matrices."
                             " The plots will be stored in folder "
                             "[out_dir]/plots/CMs")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    parser.add_argument("--wake_trim_min", type=int, required=False,
                        help="Only evaluate on within wake_trim_min of wake "
                             "before and after sleep, as determined by true "
                             "labels")
    return parser


def assert_args(args):
    """ Not yet implemented """
    return


def get_out_dir(out_dir, dataset):
    """ Returns a new, dataset-specific, out_dir under 'out_dir' """
    out_dir = os.path.abspath(out_dir)
    out_dir = os.path.join(out_dir, dataset)
    return out_dir


def prepare_output_dir(out_dir, overwrite):
    """
    Checks if the 'out_dir' exists, and if not, creates it
    Otherwise, an error is raised, unless overwrite=True, in which case nothing
    is done.
    """
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not overwrite:
        files = os.listdir(out_dir)
        if files:
            raise OSError("out_dir {} is not empty and --overwrite=False. Folder"
                          " contains the following files: {}".format(out_dir,
                                                                     files))


def get_logger(out_dir, overwrite, name="evaluation_log"):
    """
    Returns a Logger object for the given out_dir.
    The logger will throw an OSError if the dir exists and overwrite=False, in
    which case the script will terminate with a print message.
    """
    from MultiPlanarUNet.logging import Logger
    try:
        logger = Logger(out_dir,
                        active_file=name,
                        overwrite_existing=overwrite,
                        no_sub_folder=True)
    except OSError:
        from sys import exit
        print("[*] A logging file 'logs/{}' already exists. "
              "If you wish to overwrite this logfile, set the --overwrite "
              "flag.".format(name))
        exit(0)
    return logger


def get_and_load_model(project_dir, hparams, logger, weights_file_name=None):
    """
    Initializes a model in project_dir according to hparams and loads weights
    in .h5 file at path 'weights_file_name' or automatically determined from
    the 'model' sub-folder under 'project_dir' if not specified.

    Args:
        project_dir:        Path to project folder
        hparams:            A YAMLHParams object storing hyperparameters
        logger:             A Logger object
        weights_file_name:  Optional path to .h5 parameter file

    Returns:
        Parameter-initialized model
    """
    if not weights_file_name:
        from utime.models.model_init import init_and_load_best_model
        model, _ = init_and_load_best_model(
            hparams=hparams,
            model_dir=os.path.join(project_dir, "model"),
            logger=logger,
            by_name=True
        )
    else:
        from utime.models.model_init import init_and_load_model
        weights_file_name = os.path.join(project_dir, "model", weights_file_name)
        model = init_and_load_model(hparams=hparams,
                                    weights_file=weights_file_name,
                                    logger=logger,
                                    by_name=True)
    return model


def get_and_load_one_shot_model(full_hypnogram, project_dir,
                                hparams, logger, weights_file_name=None):
    """
    Returns a model according to 'hparams', potentially initialized from
    parameters in a .h5 file 'weights_file_name'.

    Independent of the settings in 'hparams', the returned model will be
    configured in 'one shot' mode - that is the model will predict on an entire
    PSG input in one forward pass. The 'full_hypnogram' array is used to
    determine the corresponding number of segments.

    Args:
        full_hypnogram:     Array of sleep stage labels, shape [n_periods, 1]
        project_dir:        Path to project directory
        hparams:            YAMLHparams object
        logger:             Logger object
        weights_file_name:  Optional path to .h5 parameter file

    Returns:
        Initialized model
    """
    # Set seguence length
    n_periods = full_hypnogram.shape[0]
    hparams["build"]["batch_shape"][1] = n_periods
    hparams["build"]["batch_shape"][0] = 1  # Should not matter
    return get_and_load_model(project_dir, hparams, logger, weights_file_name)


def set_gpu_vis(num_GPUs, force_GPU, logger=None):
    """ Helper function that sets the GPU visibility as per parsed args """
    if force_GPU:
        from MultiPlanarUNet.utils.system import set_gpu
        set_gpu(force_GPU)
    else:
        # Automatically determine GPUs to use
        from MultiPlanarUNet.utils.system import GPUMonitor
        GPUMonitor(logger).await_and_set_free_GPU(num_GPUs, stop_after=True)


def plot_hypnogram(out_dir, pred, id_, true=None):
    """
    Wrapper around hypnogram plotting function
    """
    from utime.evaluation.plotting import plot_and_save_hypnogram
    hyp_plot_dir = os.path.join(out_dir, "plots", "hypnograms")
    plot_and_save_hypnogram(out_path=os.path.join(hyp_plot_dir, id_ + ".png"),
                            y_pred=pred,
                            y_true=true,
                            id_=id_)


def plot_cm(out_dir, pred, true, n_classes, id_):
    """
    Wrapper around confusion matrix plotting function
    """
    from utime.evaluation.plotting import plot_and_save_cm
    cm_plot_dir = os.path.join(out_dir, "plots", "CMs")

    # Compute and plot CM
    plot_and_save_cm(out_path=os.path.join(cm_plot_dir, id_ + ".png"),
                     pred=pred,
                     true=true,
                     n_classes=n_classes,
                     id_=id_,
                     normalized=True)


def save(arr, fname):
    """
    Helper func to save an array (.npz) to disk in a potentially non-existing
    tree of sub-dirs
    """
    d, _ = os.path.split(fname)
    if not os.path.exists(d):
        os.makedirs(d)
    np.savez(fname, arr)


def _predict_sequence(study_pair, seq, model, verbose=True):
    """
    Predict on 'study_pair' wrapped by 'seq' using 'model'
    Predicts in batches of size seq.batch_size (set in hparams file)

    Args:
        study_pair: A SleepStudyPair object to predict on
        seq:        A BatchSequence object that stores 'study_pair'
        model:      An initialized and loaded model to predict with
        verbose:    Verbose level (True/False)

    Returns:
        An array of predicted sleep stages for all periods in 'study_pair'
        Shape [n_periods, n_classes]
    """
    from utime.utils.scriptutils.predict import sequence_predict_generator
    gen = seq.single_study_seq_generator(study_id=study_pair.identifier,
                                         overlapping=True)
    pred = sequence_predict_generator(model=model,
                                      total_seq_length=study_pair.n_periods,
                                      generator=gen,
                                      argmax=False,
                                      overlapping=True,
                                      verbose=verbose)
    return pred


def _predict_sequence_one_shot(study_pair, seq, model):
    """
    Predict on 'study_pair' wrapped by 'seq' using 'model'
    Assumes len(PSG) (number of periods in PSG) is equal to the number of
    periods output by 'model' in a single pass (one-shot segmentation).

    Used with get_and_load_one_shot_model function (--one_shot set in args)

    Args:
        study_pair: A SleepStudyPair object to predict on
        seq:        A BatchSequence object that stores 'study_pair'
        model:      An initialized and loaded model to predict with

    Returns:
        An array of predicted sleep stages for all periods in 'study_pair'
        Shape [n_periods, n_classes]
    """
    X, _ = seq.get_single_study_full_seq(study_pair.identifier)
    if X.ndim == 3:
        X = np.expand_dims(X, 0)
    return model.predict_on_batch(X)[0]


def predict_on(study_pair, seq, model=None, model_func=None, argmax=True):
    """
    High-level function for predicting on a single SleepStudyPair
    ('study_pair')object as wrapped by a BatchSequence ('seq') object using a
    model returned when calling 'model_func'.

    Arguments 'model' and 'model_func' are exclusive, exactly one must be set

    Args:
        study_pair:  A SleepStudyPair object to predict on
        seq:         A BatchSequence object that stores 'study_pair'
        model:       An initialized and loaded model to predict with
        model_func:  A callable which returns an intialized model
        argmax:      If true, returns [n_periods, 1] sleep stage labels,
                     otherwise returns [n_periods, n_classes] softmax scores.

    Returns:
        An array of predicted sleep stage scores for 'study_pair'.
        Shape [n_periods, 1] if argmax=True, otherwise [n_periods, n_classes]
    """
    if bool(model) == bool(model_func):
        raise RuntimeError("Must specify either model or model_func, "
                           "got both or neither.")
    y = study_pair.get_full_hypnogram()
    if not seq.margin:
        # Not a sequence model (no margin on center sleep segment)
        if callable(model_func):
            raise NotImplementedError("Got callable for 'model_func' "
                                      "parameter, but did not receive a "
                                      "sequence object with margin > 0.")
        from utime.utils.scriptutils.predict import predict_on_generator
        gen = seq.single_study_batch_generator(study_id=study_pair.identifier)
        pred = predict_on_generator(model=model,
                                    generator=gen,
                                    argmax=False)
    else:
        if model_func:
            # One-shot sequencing
            pred_func = _predict_sequence_one_shot
            # Get one-shot model of input shape matching the hypnogram
            model = model_func(y)
        else:
            # Batch-wise sequencing with pre-loaded model
            pred_func = _predict_sequence
        # Get prediction
        pred = pred_func(study_pair, seq, model)
    if argmax:
        pred = np.argmax(pred, -1)
    return y, pred


def get_sequencer(dataset, hparams):
    """
    Returns a BatchSequence object (see utime.seqeunces)

    OBS: Initializes the BatchSequence with scale_assertion, and
    requires_all_loaded flags all set to False.

    args:
        dataset: (SleepStudyDataset) A SleepStudyDataset storing data to
                                     predict on
        hparams: (YAMLHparams)       Hyperparameters to use for the prediction

    Returns:
        A BatchSequence object
    """
    dataset.pairs[0].load()  # get_batch_sequence needs 1 loaded study
    hparams["fit"]["balanced_sampling"] = False
    seq = dataset.get_batch_sequence(random_batches=False,
                                     **hparams["fit"],
                                     no_log=True,
                                     scale_assertion=False,  # needs >1 Studies
                                     require_all_loaded=False)
    return seq


def run_pred_and_eval(dataset,
                      out_dir,
                      model,
                      model_func,
                      hparams,
                      args,
                      logger):
    """
    Run evaluation (predict + evaluate) on a all entries of a SleepStudyDataset

    Args:
        dataset:     A SleepStudyDataset object storing one or more SleepStudy
                     objects
        out_dir:     Path to directory that will store predictions and
                     evaluation results
        model:       An initialized model used for prediction
        model_func:  A callable that returns an initialized model for pred.
        hparams:     An YAMLHparams object storing all hyperparameters
        args:        Passed command-line arguments
        logger:      A Logger object
    """
    from MultiPlanarUNet.evaluate.metrics import dice_all, class_wise_kappa
    from utime.evaluation.dataframe import (get_eval_df, add_to_eval_df,
                                            log_eval_df, with_grand_mean_col)
    logger("\nPREDICTING ON {} STUDIES".format(len(dataset.pairs)))
    seq = get_sequencer(dataset, hparams)

    # Prepare evaluation data frames
    dice_eval_df = get_eval_df(seq)
    kappa_eval_df = get_eval_df(seq)

    # Predict on all samples
    for i, sleep_study_pair in enumerate(dataset):
        id_ = sleep_study_pair.identifier
        logger("[{}/{}] Predicting on SleepStudy: {}".format(i+1,
                                                             len(dataset),
                                                             id_))

        # Predict
        with logger.disabled_in_context(), sleep_study_pair.loaded_in_context():
            y, pred = predict_on(study_pair=sleep_study_pair,
                                 seq=seq,
                                 model=model,
                                 model_func=model_func)

        if args.wake_trim_min:
            # Trim long periods of wake in start/end of true & prediction
            from utime.bin.cm import wake_trim
            y, pred = wake_trim(pairs=[[y, pred]],
                                wake_trim_min=args.wake_trim_min,
                                period_length_sec=dataset.period_length_sec)[0]
        if not args.no_save:
            # Save the output
            save_dir = os.path.join(out_dir, "files/{}".format(id_))
            save(pred, fname=os.path.join(save_dir, "pred.npz"))
            if not args.no_save_true:
                save(y, fname=os.path.join(save_dir, "true.npz"))

        # Evaluate: dice scores
        dice_pr_class = dice_all(y, pred,
                                 n_classes=seq.n_classes,
                                 ignore_zero=False, smooth=0)
        logger("-- Dice scores:  {}".format(np.round(dice_pr_class, 4)))
        add_to_eval_df(dice_eval_df, id_, values=dice_pr_class)

        # Evaluate: kappa
        kappa_pr_class = class_wise_kappa(y, pred, n_classes=seq.n_classes,
                                          ignore_zero=False)
        logger("-- Kappa scores: {}".format(np.round(kappa_pr_class, 4)))
        add_to_eval_df(kappa_eval_df, id_, values=kappa_pr_class)

        # Flag dependent evaluations:
        if args.plot_hypnograms:
            plot_hypnogram(out_dir, pred, id_, true=y)
        if args.plot_CMs:
            plot_cm(out_dir, pred, y, seq.n_classes, id_)

    # Log eval to file and screen
    dice_eval_df = with_grand_mean_col(dice_eval_df)
    log_eval_df(dice_eval_df.T,
                out_csv_file=os.path.join(out_dir, "evaluation_dice.csv"),
                out_txt_file=os.path.join(out_dir, "evaluation_dice.txt"),
                logger=logger, round=4, txt="EVALUATION DICE SCORES")
    kappa_eval_df = with_grand_mean_col(kappa_eval_df)
    log_eval_df(kappa_eval_df.T,
                out_csv_file=os.path.join(out_dir, "evaluation_kappa.csv"),
                out_txt_file=os.path.join(out_dir, "evaluation_kappa.txt"),
                logger=logger, round=4, txt="EVALUATION KAPPA SCORES")


def cross_dataset_eval(dataset_eval_dirs, out_dir):
    """ Not implemented yet """
    pass


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    assert_args(args)
    # Check project folder is valid
    from utime.utils.scriptutils import (assert_project_folder,
                                         get_splits_from_all_datasets)
    project_dir = os.path.abspath(args.project_dir)
    assert_project_folder(project_dir, evaluation=True)

    # Prepare output dir
    out_dir = get_out_dir(args.out_dir, args.data_split)
    prepare_output_dir(out_dir, args.overwrite)
    logger = get_logger(out_dir, args.overwrite)
    logger("Args dump: \n{}".format(vars(args)))

    # Get hyperparameters and init all described datasets
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(project_dir + "/hparams.yaml", logger)
    if args.channels:
        hparams["select_channels"] = args.channels
        hparams["channel_sampling_groups"] = None
        logger("Evaluating using channels {}".format(args.channels))

    # Get model
    set_gpu_vis(args.num_GPUs, args.force_GPU, logger)
    model, model_func = None, None
    if args.one_shot:
        # Model is initialized for each sleep study later
        def model_func(full_hyp):
            return get_and_load_one_shot_model(full_hyp, project_dir,
                                               hparams, logger,
                                               args.weights_file_name)
    else:
        model = get_and_load_model(project_dir, hparams, logger,
                                   args.weights_file_name)

    # Run predictions on all datasets
    datasets = get_splits_from_all_datasets(hparams=hparams,
                                            splits_to_load=(args.data_split,),
                                            logger=logger)
    eval_dirs = []
    for dataset in datasets:
        dataset = dataset[0]
        if "/" in dataset.identifier:
            # Multiple datasets, separate results into sub-folders
            ds_out_dir = os.path.join(out_dir,
                                      dataset.identifier.split("/")[0])
            if not os.path.exists(ds_out_dir):
                os.mkdir(ds_out_dir)
            eval_dirs.append(ds_out_dir)
        else:
            ds_out_dir = out_dir
        logger("[*] Running eval on dataset {}\n"
               "    Out dir: {}".format(dataset, ds_out_dir))
        run_pred_and_eval(dataset=dataset,
                          out_dir=ds_out_dir,
                          model=model, model_func=model_func,
                          hparams=hparams,
                          args=args,
                          logger=logger)
    if len(eval_dirs) > 1:
        cross_dataset_eval(eval_dirs, out_dir)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
