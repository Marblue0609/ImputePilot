"""
ImputePilot: A Demo System for Stable Imputation Model Selection in Time Series Data Repair
Zhejiang University
***
ModelsTrainer.py
@author: zhexinjin@zju.edu.cn
"""

import itertools
import logging
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd
import random as rdm
import time
from scipy.stats import ttest_rel
from sklearn import pipeline
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm # .notebook

from ImputePilot_api.ImputePilot_code.Training.ClfPipeline import ClfPipeline
from ImputePilot_api.ImputePilot_code.Training.RecommendationModel import RecommendationModel
from ImputePilot_api.ImputePilot_code.Training.TrainResults import TrainResults
from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils

def _parallel_training(args):
    """
    Trains and evaluates a pipeline. This method is used by ModelRace to train pipelines in parallel.

    Keyword arguments:
    pipe -- the ClfPipeline to train
    train_index_cv -- list of indices of the training samples in Xp_train
    Xp_train -- numpy array of train entries
    yp_train -- numpy array of train entries' labels
    X_val -- numpy array of validation entries
    y_val -- numpy array of validation entries' labels
    data_cols -- training set's features name
    labeler_properties -- dict specifying the labels' properties
    labels_set -- list of unique labels that may appear in y_train and y_val

    Return:
    If an error was thrown: the pipe's id
    Otherwise:
    1. the pipe's id
    2. the measured F1-score
    3. the measured recall@3
    4. the training time
    """
    pipe, train_index_cv, rmvd_pipes_set, Xp_train, yp_train, X_val, y_val, data_cols, labeler_properties, labels_set = args
    # Use set membership check instead of Manager.dict for thread safety
    if pipe.rm.id not in rmvd_pipes_set:
        # get the stratified fold data
        Xp_train_cv = Xp_train.iloc[train_index_cv]
        yp_train_cv = yp_train.iloc[train_index_cv]
        assert Xp_train_cv.index.identical(yp_train_cv.index)
        Xp_train_cv = Xp_train_cv.to_numpy().astype('float32')
        yp_train_cv = yp_train_cv.to_numpy().astype('str').flatten()

        # training & evaluation
        try:
            with Utils.catchtime('Training pipe %s' % pipe.rm.pipe, verbose=False) as t:
                metrics_, _ = pipe.rm.train_and_eval(Xp_train_cv, yp_train_cv, X_val, y_val, 
                                                    data_cols, labeler_properties, labels_set,
                                                    plot_cm=False, save_if_best=False)
            runtime = t.end - t.start
            if not metrics_ is None:
                return pipe.rm.id, metrics_['F1-Score'], metrics_['Recall@3'], runtime
        except:
            pass
        #logging.debug('Error for pipe %s' % pipe) # TODO tmp print
        return pipe.rm.id


class ModelsTrainer:
    """
    Class which handles classification / regression models and provides methods for its training and evaluation.
    """

    CONF = Utils.read_conf_file('modelstrainer')


    # constructor

    def __init__(self, training_set):
        """
        Initializes a ModelsTrainer object.

        Keyword arguments:
        training_set -- TrainingSet instance
        """
        self.training_set = training_set
        self.models = None # list of RecommendationModels
        self.evolution_history = []  # stores per-round ModelRace stats
    

    # public methods

    def select(self, pipelines, all_pipelines_txt, S, selection_len, score_margin,
               training_set_params=None, n_splits=3, test_method=ttest_rel, p_value=.01, alpha=.5, beta=.5, gamma=.5,
               allow_early_eliminations=True, early_break=False):
        """
        Selects (and partially trains if pipelines' training can be paused and resumed) the most-promising pipelines.

        Keyword arguments:
        pipelines -- list of ClfPipeline instances to select from
        all_pipelines_txt -- list of tuples containing the description of each pipeline (steps and params)
        S -- list of percentages for the partial training data sets
        selection_len -- approximate number of pipelines that should remain after the selection process is done.
        score_margin -- Scores acceptance margin. During selection, if a pipeline performs below MAX_SCORE - MARGIN it gets 
                        eliminated before even finishing cross-validation. Scores vary between 0 and 1.
        training_set_params -- dict specifying the data's properties (e.g. should it be balanced, reduced, etc.) (default: None)
        n_splits -- number of k-fold stratified splits (default: 3)
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the 
                       p-value. (default: scipy.stats.ttest_rel)
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).
        alpha -- alpha parameter in the score function. Weight of F1-Score. Must be between 0 and 1 (default: .5).
        beta -- beta parameter in the score function. Weight of Recall@3. Must be between 0 and 1 (default: .5).
        gamma -- gamma parameter in the score function. Weight of training time. Must be between 0 and 1 (default: .5).
        allow_early_eliminations -- True if early eliminations are allowed, False if every model should finish their cv-partial-training
                                     even if early results show evidences of bad performance (default True).
        early_break -- True if the process can stop before all the iterations are done IF the target number of pipes has been reached,
                       False if all the iterations should be executed before returning. (default False)

        Return:
        List of selected ClfPipeline.
        """
        try:
            training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params

            all_data, _, labels_set, X_train, y_train, X_val, y_val = next(self.training_set.yield_splitted_train_val(training_set_params, 1))
            X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
            assert X_train.index.identical(y_train.index)
            print('\n0/3 - data loaded') # TODO tmp print

            init_nb_pipes = len(pipelines)
            pruning_factor = (init_nb_pipes / selection_len)**(1/len(S))
            self.evolution_history = []  # reset for this run

            get_max_nb_p_at_i = lambda iter_id: round(init_nb_pipes // (pruning_factor**iter_id))
            log_interval = int(os.getenv("ImputePilot_PROGRESS_INTERVAL_SEC", "60"))

            # Manager removed - using regular set instead to avoid BrokenPipeError
            
            train_index = []

            with Utils.catchtime('ModelRace runtime', verbose=True): # TODO tmp print
                for iter_idx in tqdm(range(len(S)), leave=False):
                    with Utils.catchtime('Selection iteration %i' % (iter_idx+1), verbose=True): # TODO tmp print

                        logging.info('Iteration %i started with following pipelines:')
                        for pipe in pipelines:
                            logging.info(pipe)

                        # number of data to add for partial training (ceil to avoid 0 on tiny datasets)
                        delta_pct = (S[iter_idx] - (S[iter_idx-1] if iter_idx > 0 else 0))
                        n = int(np.ceil(X_train.shape[0] * delta_pct / 100.0))

                        # generate new pipelines from the remaining set of candidates
                        if iter_idx > 0:
                            max_nb_pipes_at_iter_i = get_max_nb_p_at_i(iter_idx) # init_nb_pipes // (pruning_factor**iter_idx)
                            nb_pipes_to_generate = max(max_nb_pipes_at_iter_i - len(pipelines), len(pipelines)//10)
                            new_pipes = ClfPipeline.generate_from_set(pipelines, all_pipelines_txt, nb_pipes_to_generate)
                            pipelines.extend(new_pipes)
                            logging.info('\nTried to generate %i new pipelines.' % nb_pipes_to_generate) # TODO tmp print
                            logging.info('Generated %i new pipelines from the remaining candidates (max nb pipes at iter %i is %i).' % (len(new_pipes), iter_idx, max_nb_pipes_at_iter_i))
                        
                        # prepare the partial training set
                        X_train_unused = X_train.loc[~X_train.index.isin(train_index)]
                        y_train_unused = y_train.loc[~y_train.index.isin(train_index)]
                        assert X_train_unused.index.identical(y_train_unused.index)

                        # Guard against tiny leftovers to avoid train_test_split errors.
                        n_to_take = min(max(int(n), 0), int(X_train_unused.shape[0]))
                        if n_to_take == 0:
                            train_index_new = []
                        elif n_to_take == X_train_unused.shape[0]:
                            train_index_new = X_train_unused.index.tolist()
                        else:
                            try:
                                train_index_new = Utils.custom_train_test_split(
                                    X_train_unused,
                                    stratify=y_train_unused,
                                    train_size=n_to_take
                                )[0].index.tolist()
                            except Exception:
                                train_index_new = Utils.custom_train_test_split(
                                    X_train_unused,
                                    train_size=n_to_take
                                )[0].index.tolist()
                        assert all(id not in train_index for id in train_index_new)

                        train_index.extend(train_index_new)
                        logging.info('\n1/3 - begining of new partial training: %i%% -> %i' % (S[iter_idx], len(train_index))) # TODO tmp print
                        
                        Xp_train = X_train.loc[train_index]
                        yp_train = y_train.loc[train_index]
                        assert Xp_train.index.identical(yp_train.index)

                        # train, eval and perform statistic tests
                        # create params list
                        # Use a regular set instead of Manager.dict() to avoid BrokenPipeError
                        # The set is copied to each worker at task creation time
                        rmvd_pipes = set()  # Track removed pipes locally
                        param_list = []
                        max_nb_scores = max(len(pipe.scores) for pipe in pipelines) + (n_splits if iter_idx > 0 else 10)
                        for pipe in pipelines:
                            n_splits_ = max_nb_scores - len(pipe.scores)

                            # Small-sample protection:
                            # cap CV folds by available samples and class frequencies; fallback to 1 full-train split.
                            y_for_split = yp_train.iloc[:, 0] if isinstance(yp_train, pd.DataFrame) else yp_train
                            y_for_split = pd.Series(y_for_split).reset_index(drop=True)
                            n_samples = int(Xp_train.shape[0])
                            min_class_count = int(y_for_split.value_counts().min()) if n_samples > 0 else 0

                            effective_splits = min(int(n_splits_), n_samples, min_class_count if min_class_count > 0 else n_samples)
                            cv_train_indices = []
                            if effective_splits >= 2:
                                try:
                                    cv_train_indices = [
                                        train_index_cv
                                        for train_index_cv, _ in StratifiedKFold(n_splits=effective_splits).split(Xp_train, y_for_split)
                                    ]
                                except Exception:
                                    cv_train_indices = []

                            if not cv_train_indices and n_samples > 0:
                                cv_train_indices = [np.arange(n_samples)]

                            for train_index_cv in cv_train_indices:
                                # Pass a frozenset copy of rmvd_pipes (will be empty initially, but that's OK)
                                param_list.append(
                                    (pipe, train_index_cv, frozenset(rmvd_pipes),  # dynamic params - use frozenset for safety
                                     Xp_train, yp_train, X_val, y_val, all_data.columns, self.training_set.get_labeler_properties(), labels_set)  # constant params
                                )

                        if len(param_list) == 0:
                            logging.warning(
                                "Skipping ModelRace iteration %i: no CV tasks generated (partial train size=%i).",
                                iter_idx + 1, Xp_train.shape[0]
                            )
                            continue
                        rdm.shuffle(param_list) # shuffle the param list such that the premature elimination is more efficient
                        
                        # run _parallel_training
                        metrics = {}
                        max_score = -np.inf
                        processed = 0
                        last_beat = time.time()
                        with Pool() as pool:
                            for res in tqdm(pool.imap_unordered(_parallel_training, param_list), total=len(param_list), leave=False):
                                processed += 1
                                if log_interval > 0 and time.time() - last_beat >= log_interval:
                                    print(
                                        f"[Progress] ModelRace iter {iter_idx+1}/{len(S)}: "
                                        f"{processed}/{len(param_list)} CV tasks done."
                                    )
                                    last_beat = time.time()
                                if res is not None:
                                    try:
                                        p_id, f1, r3, t = res
                                    except:
                                        # eliminate this pipe since there most-likely was an exception thrown during its training
                                        #  maybe due to a problematic parameters' combination?
                                        p_id = res
                                        rmvd_pipes.add(p_id)  # Changed from dict to set
                                        logging.info('%i has been eliminated probably due to an exception being thrown (likely caused by a misconfiguration)' % p_id)
                                        continue 

                                    # Skip if this pipe was already eliminated
                                    if p_id in rmvd_pipes:
                                        continue

                                    # save the evaluation results
                                    if p_id not in metrics:
                                        metrics[p_id] = []
                                    metrics[p_id].append((f1, r3, t))

                                    # early eliminations
                                    if allow_early_eliminations and len(metrics[p_id]) >= 5:
                                        mean_score = np.mean([f1 for (f1,_,_) in metrics[p_id]]) # avg of f1-scores
                                        if mean_score > max_score:
                                            max_score = mean_score
                                        elif mean_score < max_score - score_margin: # eliminate prematurely if the pipe performs really poorly
                                            rmvd_pipes.add(p_id)  # Changed from dict to set
                                            logging.info('%i has been eliminated early: avg_score=%.2f and max_score=%.2f' % (p_id, mean_score, max_score))
                                
                        logging.info('\n%i pipelines\' training have been stopped prematurely due to poor performances.' % len(rmvd_pipes)) # TODO tmp print
                        
                        logging.info('\n3/3 - statistic tests') # TODO tmp print
                        # normalize runtime between 0 and 1 & compute the score of each pipeline on each cv-split
                        metrics = {p_id: val for p_id, val in metrics.items() if p_id not in rmvd_pipes}
                        max_runtime = max([i[2] for cv_scores in metrics.values() for i in cv_scores])
                        scores_ = {id: [ModelsTrainer._compute_selection_score(f1,r3,t/max_runtime, alpha, beta, gamma)
                                        for f1,r3,t in cv_scores] 
                                    for id, cv_scores in metrics.items()}
                        
                        # add the newly measured scores to the global scores list & remove the pipes that were stopped prematurely
                        for pipe_idx in range(len(pipelines) - 1, -1, -1):
                            pipe = pipelines[pipe_idx]
                            if pipe.id in scores_:
                                pipe.scores.extend(scores_[pipe.id])
                            else:
                                del pipelines[pipe_idx]
                        
                        # statistical tests - remove pipes that are significantly worse than any other pipe
                        less_aggressive_pruning = (len(pipelines) <= get_max_nb_p_at_i(iter_idx)) if iter_idx > 0 else True
                        worse_pipes = self._apply_test(
                            pipelines, 
                            test_method,
                            K=pruning_factor,
                            p_value=p_value,
                            less_aggressive_pruning=less_aggressive_pruning
                        )

                        if len(pipelines) < 20: # TODO tmp print
                            logging.info([(p.id, np.mean(p.scores)) for p in pipelines]) # TODO tmp print
                            logging.info([
                                (self._safe_pvalue(test_method, a.scores, b.scores), a.rm.id,b.rm.id)  # we keep the worse pipelines of the 2
                                    for a,b in itertools.combinations(pipelines, r=2) # for all pairs of pipes
                            ]) # TODO tmp print

                        # remove the worse pipes
                        pipelines = [p for p in pipelines if p not in worse_pipes]

                        # --- record evolution snapshot ---
                        round_best_score = 0.0
                        round_best_pipe_name = 'Unknown'
                        for p in pipelines:
                            if p.scores:
                                avg = float(np.mean(p.scores))
                                if avg > round_best_score:
                                    round_best_score = avg
                                    try:
                                        steps = list(p.rm.pipe.named_steps.keys())
                                        round_best_pipe_name = ' + '.join(
                                            [s.replace('_', ' ').title() for s in steps]
                                        )
                                    except Exception:
                                        round_best_pipe_name = f'Pipeline {p.id}'
                        self.evolution_history.append({
                            'round': iter_idx + 1,
                            'data_pct': S[iter_idx],
                            'candidates': len(pipelines),
                            'early_eliminated': len(rmvd_pipes),
                            'ttest_eliminated': len(worse_pipes),
                            'bestScore': round(round_best_score, 4),
                            'bestPipeline': round_best_pipe_name,
                        })
                        # --- end evolution snapshot ---

                        logging.info('\nThere remains %i pipelines. %i have been eliminated by t-test.' % (len(pipelines), len(worse_pipes)))

                        if early_break and len(pipelines) <= selection_len:
                            break

                # if we have more pipes remaining that what we wanted
                if len(pipelines) > selection_len: 
                    logging.info('Too many pipelines remaining. Last attempt to eliminate "worse" candidates.')
                    pipelines = sorted(pipelines, key=lambda p: np.mean(p.scores), reverse=True)
                    for i in reversed(range(len(pipelines))): # rank pipes based on their avg scores
                        p_i = pipelines[i]

                        # pairwise ttest if p is worse than any other: prune p
                        for j, p_j in enumerate(pipelines):
                            if i is not j and self._safe_pvalue(test_method, p_i.scores, p_j.scores) < p_value and np.mean(p_i.scores) < np.mean(p_j.scores):
                                logging.info('%i was eliminated due to significantly worse performances than another candidate.' % p_i.id)
                                del pipelines[i]
                                break
                                
                        if len(pipelines) <= selection_len:
                            break

        except Exception as e:
            logging.exception('Got exception while selecting pipelines.')
            if len(pipelines) >= init_nb_pipes:
                raise e
        return pipelines

    def train(self, models, train_for_production=False):
        """
        Trains and evaluates models given to this initialization of this trainer. Uses T-Daub for best-algorithms selection
        and cross-validation.

        Keyword arguments: 
        models -- list of RecommendationModel instances that should be trained and evaluated
        train_for_production -- True if the models should be trained on ALL data once the cross-val is done, False 
                             otherwise (default: False)

        Return:
        A TrainResults' instance containing the training results
        """
        
        vc_estimators = []
        for m in models:
            est = m.get_trained_pipeline(use_pipeline_prod=False)
            vc_estimators.append( (str(m.id), est if est is not None else m.pipe) )
        vc = VotingClassifier(vc_estimators, voting='soft')
        
        models.insert(
            0,
            RecommendationModel(-1, 'classifier', Pipeline([('voting ensemble', vc)]))
        )

        self.models = models

        train_results = self._train(models, train_for_production=train_for_production)
        return train_results
        
    
    # private methods

    @staticmethod
    def _safe_pvalue(test_method, a_scores, b_scores):
        """
        Returns a safe p-value for paired tests when score lengths differ.
        Falls back to 1.0 (no significant difference) on invalid inputs.
        """
        try:
            a = list(a_scores) if a_scores is not None else []
            b = list(b_scores) if b_scores is not None else []
            if not a or not b:
                return 1.0
            min_len = min(len(a), len(b))
            if min_len < 2:
                return 1.0
            if len(a) != len(b):
                a = a[-min_len:]
                b = b[-min_len:]
            p_val = test_method(a, b)[1]
            if p_val is None or np.isnan(p_val):
                return 1.0
            return float(p_val)
        except Exception:
            return 1.0

    def _apply_test(self, pipelines, test_method, K=4, p_value=.01, less_aggressive_pruning=True):
        """
        Applies a significance test (paired t-test) to the pipelines to identify and return those that perform worse than others.

        Keyword arguments:
        pipelines -- list of ClfPipeline to apply t-test to.
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the p-value.
        K -- Pruning factor. Example: if K=4, the expected number of pipes to "survive" is 1/4 of the original set. (default: 4)
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).
        less_aggressive_pruning -- uses a less aggressive pruning strategy if True, eliminates any pipe that is significantly worse than any other
                                   pipe otherwise. (default: True)

        Return:
        List of _TmpPipeline that are performing worse than others.
        """
        eliminated_pipes = []
        pipes_combinations = list(itertools.combinations(pipelines, r=2))
        worse_counters = {p: 0 for p in pipelines}
        
        T = len(pipelines) // K if less_aggressive_pruning else 0
        
        i = len(pipes_combinations)-1
        while i > 0:
            a,b = pipes_combinations[i]

            # if the paired t-test shows a statistical difference between the two
            significance_diff = self._safe_pvalue(test_method, a.scores, b.scores) < p_value
            if significance_diff:
                worse_pipe = a if np.mean(a.scores) < np.mean(b.scores) else b
                worse_counters[worse_pipe] += 1
                
                if worse_counters[worse_pipe] > T:
                    # eliminate the worst one
                    eliminated_pipes.append(worse_pipe)
                    # remove all pairs of pipes continaining the eliminated pipe to avoid unnecessary comparisons
                    for j in reversed(range(len(pipes_combinations))):
                        x,y = pipes_combinations[j]
                        if worse_pipe in (x,y):
                            del pipes_combinations[j]
                            i -= 1 if j < i else 0
            i -= 1
        return eliminated_pipes

    def _train(self, models_to_train, train_for_production=False, training_set_params=None, save_results=True, save_if_best=True):
        """
        Trains and evaluates a list of models over cross-validation (and after gridsearch if it is necessary).

        Keyword arguments:
        models_to_train -- list of RecommendationModel instances that should be trained and evaluated
        train_for_production -- True if the models should be trained on ALL data once the cross-val is done, False 
                             otherwise (default: False)
        training_set_params -- dict specifying the data's properties (e.g. should it be balanced, reduced, etc.) (default: None)
        save_results -- True if the results should be saved to disk, False otherwise (default: True)
        save_if_best -- True if the model should be saved if it is the best performing one, false otherwise (default: False)

        Return:
        A TrainResults' instance containing the training results
        """
        training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params

        train_results = TrainResults(models_to_train, self.training_set.get_labeler_properties()['type'])
        try:
            for split_id, yielded in enumerate(self.training_set.yield_splitted_train_val(training_set_params, 
                                                                                          ModelsTrainer.CONF['NB_CV_SPLITS'])):
                print('\nCross-validation split n°%i' % (split_id+1))
                all_data, all_labels, labels_set, X_train, y_train, X_val, y_val = yielded

                print('X_train shape:', X_train.shape, ', X_val shape:', X_val.shape) # TODO tmp print
                print(np.asarray(np.unique(y_train, return_counts=True)).T) # TODO tmp print
                print(np.asarray(np.unique(y_val, return_counts=True)).T) # TODO tmp print

                for model in models_to_train:

                    # training
                    print('Training %s.' % model)
                    with Utils.catchtime('Training model %s @ split %i' % (model, split_id)):
                        try:
                            scores, cm = model.train_and_eval(X_train, y_train, X_val, y_val, 
                                                            all_data.columns, self.training_set.get_labeler_properties(), labels_set,
                                                            plot_cm=True, save_if_best=save_if_best)
                            if scores is None:
                                raise Exception('Training aborted.')
                        except Exception as e:
                            print('Encountered exception while training %s. \n %s' % (model, e))
                            scores, cm = None, None

                    # save results
                    # TrainResults contain a dict grouping the results of each trained model
                    train_results.add_model_cv_split_result(split_id, model, scores, cm)

            if train_for_production: # train the models on training data (65%): 
                print('\nTraining models on training data (65% split).')
                # Use only training data to ensure fair evaluation with test set
                _, X_train, y_train = self.training_set.get_train_data(training_set_params)
                print(f'Training data shape: X={X_train.shape}, y={y_train.shape}')
                for model in models_to_train:
                    with Utils.catchtime('Training model %s on training data' % model):
                        try:
                            model.trained_pipeline_prod = clone(model.pipe).fit(X_train, y_train)
                        except Exception as e:
                            print('Encountered exception while training %s on training data. \n %s' % (model, e))

        finally:
            # save results to disk
            if save_results:
                train_results.save(self.training_set, save_train_set=ModelsTrainer.CONF['SAVE_TRAIN_SET'])

        return train_results

    
    # static methods

    def _compute_selection_score(f1, r3, t, alpha, beta, gamma):
        """
        Method that computes the score based on which a model will be evaluated during the selection process. 

        Keyword arguments:
        f1 -- F1-score of a model measured on a test set
        r3 -- recall@3 of a model measured on a test set
        t -- model's training time
        alpha -- parameter value to define the impact of the F1-score in the score
        beta -- parameter value to define the impact of the Recall@3 in the score
        gamma -- parameter value to define the impact of the model's training time in the score

        Return:
        Returns the score of a model used during the selection process.
        """
        return ((alpha*f1) + (beta*r3) - (gamma*t) + gamma) / (alpha+beta+gamma)
