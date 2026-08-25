function results = Compute_All_Session_SNR(monkeys, sessions_by_monkey, save_outputs, apply_photodiode_qc)
%COMPUTE_ALL_SESSION_SNR Compute clustering SNR for physiology sessions.
%
% Default publication run (all 61 sessions):
%   Compute_All_Session_SNR
%
% Targeted regression test without saving:
%   sessions.Faure = {'june_24_g0'};
%   Compute_All_Session_SNR({'Faure'}, sessions, false)
%
% This preserves the SNR calculation in
% Single_Trial_Statistics_Clustering_All.m: original simultaneous-trial
% neuron eligibility, photodiode-QC exclusion afterward, 0-500 ms mean
% firing rate after flash one, FR threshold, PCA to three components, and
% stratified cross-validated linear-SVM ROC-AUC for maze 1 versus maze 6.

if nargin < 1 || isempty(monkeys)
    monkeys = {'Faure', 'Nielsen'};
end
if nargin < 2 || isempty(sessions_by_monkey)
    sessions_by_monkey = struct();
end
if nargin < 3 || isempty(save_outputs)
    save_outputs = true;
end
if nargin < 4 || isempty(apply_photodiode_qc)
    apply_photodiode_qc = true;
end
if ~apply_photodiode_qc
    error(['Photodiode QC is required for the publication analysis. ' ...
        'Use the archived pre-QC code only for historical comparisons.']);
end

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
physiology_root = fullfile(project_root, 'Data', 'Physiology');

behavior_root = fullfile(physiology_root, 'Behavioral_Data');
neural_root = fullfile(physiology_root, 'Neural_Data');
single_trial_root = fullfile(physiology_root, 'Single_Trial_List');
output_root = fullfile(script_dir, 'SNR_All_Sessions');

FR_THRESH = 1;
n_mazes = 6;
conditions_per_maze = 4;
snr_maze_pair = [1 6];
timing = load(fullfile(physiology_root, 'maze_condition_timing.mat'));
maze_order = timing.maze_order;

monkey_col = strings(0, 1);
session_col = strings(0, 1);
snr_auc_col = zeros(0, 1);
n_maze_1_col = zeros(0, 1);
n_maze_6_col = zeros(0, 1);
n_neurons_col = zeros(0, 1);
status_col = strings(0, 1);

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    behavior_dir = fullfile(behavior_root, monkey);
    neural_dir = fullfile(neural_root, monkey);
    single_trial_dir = fullfile(single_trial_root, monkey);

    if isfield(sessions_by_monkey, monkey)
        sessions = sessions_by_monkey.(monkey);
    else
        files = dir(fullfile(neural_dir, '*_Whole_Trial_FR_Causal.mat'));
        files = files(~startsWith({files.name}, '.'));
        sessions = erase({files.name}, '_Whole_Trial_FR_Causal.mat');
        sessions = sort(sessions);
    end

    fprintf('\n============================================================\n');
    fprintf('All-session SNR: %s (%d sessions)\n', monkey, numel(sessions));
    fprintf('============================================================\n');

    for session_idx = 1:numel(sessions)
        session = sessions{session_idx};
        fprintf('[%d/%d] %s %s\n', ...
            session_idx, numel(sessions), monkey, session);

        snr_auc = NaN;
        n_maze_1 = 0;
        n_maze_6 = 0;
        n_neurons = 0;
        status = "passed";

        try
            behavior_file = fullfile(behavior_dir, ...
                [session '_good_trials_concat.mat']);
            neural_file = fullfile(neural_dir, ...
                [session '_Whole_Trial_FR_Causal.mat']);
            list_file = fullfile(single_trial_dir, ...
                ['single_trial_list_' session '.mat']);

            assert(isfile(behavior_file), ...
                'Missing behavior file: %s', behavior_file);
            assert(isfile(neural_file), ...
                'Missing neural file: %s', neural_file);
            assert(isfile(list_file), ...
                'Missing single-trial list: %s', list_file);

            trial_fade = h5read(behavior_file, ...
                '/save_all_data/trial_fade');
            photodiode_qc_bad = logical(h5read(behavior_file, ...
                '/save_all_data/photodiode_qc_bad'));
            geo_present = h5read(behavior_file, ...
                '/save_all_data/geo_present');
            flash_one = h5read(behavior_file, ...
                '/save_all_data/flash_one');
            cids = h5read(behavior_file, '/save_all_data/cids');
            nrns_all = h5read(behavior_file, '/save_all_data/nrns');
            trial_indices_all = h5read(behavior_file, ...
                '/save_all_data/trial_indices_all');
            path_type = h5read(behavior_file, ...
                '/save_all_data/path_type');
            analysis_rows = ~photodiode_qc_bad;

            list_data = load(list_file, 'min_value', 'max_value');
            min_value = list_data.min_value;
            max_value = list_data.max_value;
            choose_trials = min_value:max_value;
            firing_rates = matfile(neural_file);

            all_neuron_data = [];
            maze_activity_final = [];
            maze_trial_ids_final = [];

            % Keep the original neuron numbering/eligibility calculation.
            for nrns = 1:length(unique(cids))
                optimization_neuron_mask = nrns_all == nrns;

                if length(intersect(choose_trials, ...
                        trial_indices_all(optimization_neuron_mask))) ~= ...
                        length(choose_trials)
                    continue;
                end

                neuron_mask = optimization_neuron_mask & analysis_rows;
                neuron_rows = find(neuron_mask);
                if isempty(neuron_rows)
                    continue;
                end

                geo_present_t = geo_present(neuron_mask);
                flash_one_t = flash_one(neuron_mask);
                path_type_t = path_type(neuron_mask);
                trial_fade_t = trial_fade(neuron_mask);
                trial_list = trial_indices_all(neuron_mask);

                % matfile requires regularly spaced indices. Read one
                % contiguous block, then select the requested rows locally.
                row_min = min(neuron_rows);
                row_max = max(neuron_rows);
                smooth_block = firing_rates.smooth_session(row_min:row_max, :);
                smooth_all = smooth_block(neuron_rows - row_min + 1, :);
                clear smooth_block;

                can_use = true;
                for cond = 1:24
                    visible_indices = find( ...
                        path_type_t == cond & trial_fade_t == 0);
                    trial_ids = trial_list(visible_indices);
                    values_within_range = visible_indices( ...
                        trial_ids >= min_value & trial_ids <= max_value);
                    if length(values_within_range) < 2
                        can_use = false;
                    end
                end
                if ~can_use
                    continue;
                end

                maze_activity = cell(n_mazes, 1);
                maze_trial_ids = cell(n_mazes, 1);

                for maze = 1:n_mazes
                    cond_start = (maze - 1) * conditions_per_maze + 1;
                    cond_end = maze * conditions_per_maze;
                    maze_trials = [];
                    maze_ids = [];

                    for cond = cond_start:cond_end
                        visible_indices = find( ...
                            path_type_t == cond & trial_fade_t == 0);
                        trial_ids = trial_list(visible_indices);
                        trl_indices = visible_indices( ...
                            trial_ids >= min_value & trial_ids <= max_value);

                        flash_one_to_use = flash_one_t(trl_indices);
                        geo_present_to_use = geo_present_t(trl_indices);
                        start_indices = floor((flash_one_to_use - ...
                            geo_present_to_use) * 1000);
                        end_indices = start_indices + 500;

                        extracted_data = arrayfun(@(trial) ...
                            nanmean(smooth_all(trl_indices(trial), ...
                            start_indices(trial):end_indices(trial))), ...
                            1:length(trl_indices))';

                        maze_trials = [maze_trials; extracted_data];
                        maze_ids = [maze_ids; trial_list(trl_indices)];
                    end

                    maze_activity{maze} = maze_trials;
                    maze_trial_ids{maze} = maze_ids;
                end

                maze_activity = maze_activity(maze_order);
                maze_trial_ids = maze_trial_ids(maze_order);
                average_firing_rates = cellfun( ...
                    @(x) nanmean(x, 'all'), maze_activity);

                if ~all(average_firing_rates < FR_THRESH)
                    if ~isempty(maze_trial_ids_final)
                        assert(isequal(maze_trial_ids, maze_trial_ids_final), ...
                            'Trial ID/order mismatch across neurons.');
                    end
                    all_neuron_data = horzcat(all_neuron_data, ...
                        vertcat(maze_activity{:}));
                    maze_activity_final = maze_activity;
                    maze_trial_ids_final = maze_trial_ids;
                end
            end

            if isempty(all_neuron_data)
                status = "no usable neurons";
            else
                n_neurons = size(all_neuron_data, 2);
                normalized_data = zscore(all_neuron_data, 0, 1);
                [~, score] = pca(normalized_data);
                assert(size(score, 2) >= 3, ...
                    'Fewer than three PCA components were available.');
                pca_reduced_data = score(:, 1:3);

                maze_identity = [];
                for maze = 1:n_mazes
                    maze_identity = [maze_identity; ...
                        maze * ones(size(maze_activity_final{maze}, 1), 1)];
                end
                maze_identity = maze_identity(:);

                mask_a = maze_identity == snr_maze_pair(1);
                mask_b = maze_identity == snr_maze_pair(2);
                mask_ab = mask_a | mask_b;
                n_maze_1 = nnz(mask_a);
                n_maze_6 = nnz(mask_b);

                if n_maze_1 < 2 || n_maze_6 < 2
                    status = "insufficient maze trials";
                else
                    X = pca_reduced_data(mask_ab, :);
                    y = zeros(sum(mask_ab), 1);
                    y(mask_b(mask_ab)) = 1;
                    rng(0);
                    K = min([5, n_maze_1, n_maze_6]);

                    if K < 2
                        status = "insufficient CV folds";
                    else
                        cvp = cvpartition(y, 'KFold', K, ...
                            'Stratify', true);
                        scores = nan(size(y));

                        for fold = 1:cvp.NumTestSets
                            train_rows = training(cvp, fold);
                            test_rows = test(cvp, fold);
                            model = fitcsvm(X(train_rows, :), y(train_rows), ...
                                'KernelFunction', 'linear', ...
                                'Standardize', true, ...
                                'ClassNames', [0 1]);
                            try
                                model = fitPosterior(model, ...
                                    X(train_rows, :), y(train_rows));
                                [~, fold_scores] = predict(model, ...
                                    X(test_rows, :));
                                scores(test_rows) = fold_scores(:, 2);
                            catch
                                [~, fold_scores] = predict(model, ...
                                    X(test_rows, :));
                                scores(test_rows) = -fold_scores(:, 1);
                            end
                        end

                        [~, ~, ~, snr_auc] = perfcurve(y, scores, 1);
                    end
                end
            end
        catch ME
            status = "failed: " + string(ME.message);
            warning('SNR failed for %s %s: %s', ...
                monkey, session, ME.message);
        end

        monkey_col(end + 1, 1) = string(monkey);
        session_col(end + 1, 1) = string(session);
        snr_auc_col(end + 1, 1) = snr_auc;
        n_maze_1_col(end + 1, 1) = n_maze_1;
        n_maze_6_col(end + 1, 1) = n_maze_6;
        n_neurons_col(end + 1, 1) = n_neurons;
        status_col(end + 1, 1) = status;

        fprintf('  AUC = %.6f | neurons = %d | maze trials = %d/%d | %s\n', ...
            snr_auc, n_neurons, n_maze_1, n_maze_6, status);
    end
end

results = table(monkey_col, session_col, snr_auc_col, ...
    n_maze_1_col, n_maze_6_col, n_neurons_col, status_col, ...
    'VariableNames', {'monkey', 'session', 'snr_auc', ...
    'n_maze_1_trials', 'n_maze_6_trials', 'n_neurons', 'status'});
results = sortrows(results, 'snr_auc', 'descend', ...
    'MissingPlacement', 'last');

fig_all = plot_snr_results(results, snr_maze_pair, 'All sessions');
fig_by_monkey = gobjects(numel(monkeys), 1);
for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    monkey_results = results(results.monkey == string(monkey), :);
    fig_by_monkey(monkey_idx) = plot_snr_results( ...
        monkey_results, snr_maze_pair, monkey);
end

if save_outputs
    if ~isfolder(output_root)
        mkdir(output_root);
    end
    save(fullfile(output_root, 'all_session_snr_results.mat'), ...
        'results', 'snr_maze_pair', 'maze_order', 'FR_THRESH', ...
        'apply_photodiode_qc');
    writetable(results, fullfile(output_root, ...
        'all_session_snr_results.csv'));
    savefig(fig_all, fullfile(output_root, 'all_session_snr_descending.fig'));
    exportgraphics(fig_all, fullfile(output_root, ...
        'all_session_snr_descending.png'), 'Resolution', 300);
    for monkey_idx = 1:numel(monkeys)
        monkey = monkeys{monkey_idx};
        savefig(fig_by_monkey(monkey_idx), fullfile(output_root, ...
            ['all_session_snr_descending_' monkey '.fig']));
        exportgraphics(fig_by_monkey(monkey_idx), fullfile(output_root, ...
            ['all_session_snr_descending_' monkey '.png']), ...
            'Resolution', 300);
    end
    fprintf('\nSaved all-session SNR outputs to %s\n', output_root);
end
end

function fig = plot_snr_results(results, snr_maze_pair, label)
valid = isfinite(results.snr_auc);
fig = figure('Color', 'w', 'Position', [100 100 1500 650]);
bar(find(valid), results.snr_auc(valid), ...
    'FaceColor', [0.25 0.45 0.75], 'EdgeColor', 'none');
hold on;
yline(0.5, 'k--', 'Chance', 'LineWidth', 1.5);
xticks(find(valid));
xticklabels(results.session(valid));
xtickangle(60);
ylabel('Cross-validated ROC-AUC');
xlabel('Session (descending AUC)');
title(sprintf('%s SNR: maze %d versus maze %d', ...
    label, snr_maze_pair(1), snr_maze_pair(2)));
ylim([0 1.02]);
grid on;
box off;
end
