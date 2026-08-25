%% ============================================================
%  LOCAL SINGLE-TRIAL PCA CLUSTERING ANALYSIS
%
%  Expected local folders:
%    <project_root>/Data/Physiology/Behavioral_Data/Faure/
%    <project_root>/Data/Physiology/Behavioral_Data/Nielsen/
%
%    <project_root>/Data/Physiology/Neural_Data/Faure/
%    <project_root>/Data/Physiology/Neural_Data/Nielsen/
%
%    <project_root>/Data/Physiology/Single_Trial_List/Faure/
%    <project_root>/Data/Physiology/Single_Trial_List/Nielsen/
%
%  Code folder:
%    <project_root>/Figure_Five/Dendogram_all/
%
%  Runs only:
%    Faure:   june_24_g0, june_8_g0
%    Nielsen: Nov_3_g0, Oct_22_g0
% ============================================================

clear; close all; clc;

% Set seed for reproducibility
seed = 45;
rng(seed, 'twister');

%% Local roots

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
physiology_data_root = fullfile(project_root, 'Data', 'Physiology');

behavior_root          = fullfile(physiology_data_root, 'Behavioral_Data');
neural_root            = fullfile(physiology_data_root, 'Neural_Data');
single_trial_list_root = fullfile(physiology_data_root, 'Single_Trial_List');
code_root              = fullfile(project_root, 'Figure_Five', 'Dendogram_all');

% Publication analyses always exclude trials that fail photodiode QC.
output_root = code_root;

addpath(genpath(code_root));

%% Sessions to run

monkeys  = {'Faure','Nielsen'};

sessions = struct( ...
    'Faure',   {{'june_24_g0', 'june_8_g0'}}, ...
    'Nielsen', {{'Nov_3_g0', 'Oct_22_g0'}} );

%% Parameters

FR_THRESH = 1;

n_mazes = 6;
conditions_per_maze = 4;

timing = load(fullfile(physiology_data_root, ...
    'maze_condition_timing.mat'));
maze_order = timing.maze_order;
snr_maze_pair = [1 6];

%% Store results in workspace only

all_results = struct();
result_idx = 0;

%% Main loop

for m = 1:numel(monkeys)

    MONKEY = monkeys{m};
    session_list = sessions.(MONKEY);

    behavior_directory = fullfile(behavior_root, MONKEY);
    FR_directory = fullfile(neural_root, MONKEY);
    single_trial_list_directory = fullfile(single_trial_list_root, MONKEY);

    fprintf('\n============================================================\n');
    fprintf('Running monkey: %s\n', MONKEY);
    fprintf('============================================================\n');

    for n_t = 1:numel(session_list)

        session_name = session_list{n_t};

        fprintf('\n------------------------------------------------------------\n');
        fprintf('Session %d/%d: %s %s\n', n_t, numel(session_list), MONKEY, session_name);
        fprintf('\n------------------------------------------------------------\n');

        %% Load behavioral/session data

        file_name = fullfile(behavior_directory, [session_name, '_good_trials_concat.mat']);
        % Read only fields used below. Loading the complete save_all_data
        % struct also loads the unused multi-gigabyte spikes_all matrix.
        trial_fade = h5read(file_name, '/save_all_data/trial_fade');
        photodiode_qc_bad = h5read(file_name, ...
            '/save_all_data/photodiode_qc_bad');
        geo_present = h5read(file_name, '/save_all_data/geo_present');
        flash_one = h5read(file_name, '/save_all_data/flash_one');
        cids = h5read(file_name, '/save_all_data/cids');
        nrns_all = h5read(file_name, '/save_all_data/nrns');
        trial_indices_all = h5read(file_name, ...
            '/save_all_data/trial_indices_all');
        path_type = h5read(file_name, '/save_all_data/path_type');
        analysis_rows = ~logical(photodiode_qc_bad);

        %% Load firing rates

        FR_WH = matfile(fullfile(FR_directory, ...
            [session_name, '_Whole_Trial_FR_Causal.mat']));

        %% Load single-trial window

        load(fullfile(single_trial_list_directory, ['single_trial_list_', session_name, '.mat']), ...
            'min_value', 'max_value');

        choose_trials = min_value:max_value;

        %% Compute IC activity matrix

        all_neuron_data = [];
        maze_activity_final = [];
        maze_trial_ids_final = [];

        for nrns = 1:length(unique(cids))

            optimization_neuron_mask = nrns_all == nrns;

            if length(intersect(choose_trials, ...
                    trial_indices_all(optimization_neuron_mask))) == length(choose_trials)

                neuron_mask = optimization_neuron_mask & analysis_rows;

                geo_present_t = geo_present(neuron_mask);
                flash_one_t = flash_one(neuron_mask);
                path_type_t = path_type(neuron_mask);
                trial_fade_t = trial_fade(neuron_mask);
                trial_list = trial_indices_all(neuron_mask);

                neuron_rows = find(neuron_mask);
                
                if isempty(neuron_rows)
                    continue;
                end
                
                row_min = min(neuron_rows);
                row_max = max(neuron_rows);
                
                smooth_block = FR_WH.smooth_session(row_min:row_max, :);
                
                smooth_all = smooth_block(neuron_rows - row_min + 1, :);
                
                clear smooth_block;

                can_use = 1;

                for cond = 1:24

                    visible_indices = find( ...
                        (path_type_t == cond) & (trial_fade_t == 0));

                    indx_use = trial_list(visible_indices);
                    within_range_indices = (indx_use >= min_value) & (indx_use <= max_value);
                    values_within_range = visible_indices(within_range_indices);

                    if length(values_within_range) < 2
                        can_use = 0;
                    end

                end

                if can_use == 1

                    maze_activity = cell(n_mazes, 1);
                    maze_trial_ids = cell(n_mazes, 1);

                    for maze = 1:n_mazes

                        cond_start = (maze - 1) * conditions_per_maze + 1;
                        cond_end = maze * conditions_per_maze;
                        maze_conditions = cond_start:cond_end;

                        maze_trials = [];
                        maze_ids = [];

                        for cond = maze_conditions

                            visible_indices = find( ...
                                (path_type_t == cond) & (trial_fade_t == 0));

                            indx_use = trial_list(visible_indices);
                            within_range_indices = (indx_use >= min_value) & (indx_use <= max_value);
                            trl_indices = visible_indices(within_range_indices);

                            flash_one_to_use = flash_one_t(trl_indices);
                            geo_present_to_use = geo_present_t(trl_indices);

                            start_indices = floor((flash_one_to_use - geo_present_to_use) * 1000);
                            start_indices = start_indices + 0;
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

                    average_firing_rates = cellfun(@(x) nanmean(x, 'all'), maze_activity);

                    if ~all(average_firing_rates < FR_THRESH)

                        if ~isempty(maze_trial_ids_final)
                            assert(isequal(maze_trial_ids, maze_trial_ids_final), ...
                                'Trial ID/order mismatch across neurons in %s.', session_name);
                        end

                        all_neuron_data = horzcat(all_neuron_data, vertcat(maze_activity{:}));
                        maze_activity_final = maze_activity;
                        maze_trial_ids_final = maze_trial_ids;

                    end

                end

            end

        end

        fprintf('Size of all_neuron_data:\n');
        disp(size(all_neuron_data));

        if isempty(all_neuron_data)
            warning('No usable neurons for %s %s. Skipping session.', MONKEY, session_name);
            continue;
        end

        maze_activity = maze_activity_final;
        maze_trial_ids = maze_trial_ids_final;

        %% PCA and clustering

        normalized_data = zscore(all_neuron_data, 0, 1);

        [coeff, score, latent] = pca(normalized_data);

        pca_reduced_data = score(:, 1:3);

        dist_matrix_pca = pdist(pca_reduced_data, 'euclidean');
        linkage_tree_pca = linkage(dist_matrix_pca, 'ward');

        figure;
        dendrogram(linkage_tree_pca, 0);
        xlabel('Trials');
        ylabel('Distance');
        title(sprintf('Hierarchical Clustering Dendrogram PCA Space: %s %s', ...
            MONKEY, session_name), 'Interpreter', 'none');

        clusters = cluster(linkage_tree_pca, 'maxclust', 2);

        %% Quantify trial distribution for each maze within each cluster

        cluster_distribution = zeros(2, n_mazes);
        cluster_trial_ids = cell(2, n_mazes);
        cluster_trial_positions = cell(2, n_mazes);

        start_index = 0;

        for maze_id = 1:n_mazes

            num_trials_in_maze = size(maze_activity{maze_id}, 1);
            maze_trial_indices = start_index + (1:num_trials_in_maze);
            start_index = start_index + num_trials_in_maze;

            for cluster_id = 1:2

                cluster_trial_indices = find(clusters == cluster_id);
                trial_indices = intersect(cluster_trial_indices, maze_trial_indices);

                cluster_trial_positions{cluster_id, maze_id} = trial_indices;
                within_maze = trial_indices - maze_trial_indices(1) + 1;
                cluster_trial_ids{cluster_id, maze_id} = ...
                    maze_trial_ids{maze_id}(within_maze);
                cluster_distribution(cluster_id, maze_id) = length(trial_indices);

            end

        end

        %% Build maze identity

        maze_identity = [];

        for maze_id = 1:n_mazes
            maze_identity = [maze_identity; maze_id * ones(size(maze_activity{maze_id}, 1), 1)];
        end

        maze_identity = maze_identity(:);

        %% Null distribution

        num_shuffles = 100;
        null_distribution = zeros(2, n_mazes, num_shuffles);

        all_maze_labels = maze_identity;

        for shuffle_idx = 1:num_shuffles

            shuffled_maze_labels = all_maze_labels(randperm(length(all_maze_labels)));
            shuffled_cluster_distribution = zeros(2, n_mazes);

            for cluster_id = 1:2

                cluster_trial_indices = find(clusters == cluster_id);

                for maze_id = 1:n_mazes

                    shuffled_cluster_distribution(cluster_id, maze_id) = ...
                        sum(shuffled_maze_labels(cluster_trial_indices) == maze_id);

                end

            end

            shuffled_percentages = ...
                (shuffled_cluster_distribution ./ sum(shuffled_cluster_distribution, 1)) * 100;

            null_distribution(:, :, shuffle_idx) = shuffled_percentages;

        end

        %% Plot cluster distribution

        cluster_percentages = ...
            (cluster_distribution ./ sum(cluster_distribution, 1)) * 100;

        % Canonical maze 1 is the hierarchical reference and canonical
        % maze 6 is the sequential reference.
        [~, hierarchical_row] = max(cluster_distribution(:, 1));
        [~, sequential_row] = max(cluster_distribution(:, 6));

        strategy_rows_all = cell(2, 1);
        strategy_rows_all{hierarchical_row} = 'hierarchical';
        strategy_rows_all{sequential_row} = 'sequential';

        if hierarchical_row == sequential_row
            error(['The same cluster was selected as both hierarchical ' ...
                'and sequential for %s %s.'], MONKEY, session_name);
        end

        colors = lines(2);

        figure;
        hold on;

        for cluster_id = 1:2
            for maze_id = 1:n_mazes

                dot_size = cluster_percentages(cluster_id, maze_id) * 10;

                scatter(cluster_id, maze_id, dot_size + 1, ...
                    'MarkerFaceColor', colors(cluster_id, :), ...
                    'MarkerEdgeColor', 'k', ...
                    'LineWidth', 0.5);

            end
        end

        xlabel('Cluster');
        ylabel('Maze');
        xticks(1:2);
        yticks(1:n_mazes);
        title(sprintf('Maze Distribution Across Clusters: %s %s', ...
            MONKEY, session_name), 'Interpreter', 'none');
        legend({'Cluster 1', 'Cluster 2'}, 'Location', 'best');

        %% SNR metric: CV ROC-AUC
        
        mask_a = maze_identity == snr_maze_pair(1);
        mask_b = maze_identity == snr_maze_pair(2);
        mask_ab = mask_a | mask_b;

        na = nnz(mask_a);
        nb = nnz(mask_b);
        warning('off', 'all');

        if na < 2 || nb < 2

            warning('Not enough trials in maze %d (%d) or maze %d (%d). AUC set to NaN.', ...
                snr_maze_pair(1), na, snr_maze_pair(2), nb);

            snr_auc = NaN;

        else

            X = pca_reduced_data(mask_ab, :);

            y = zeros(sum(mask_ab), 1);
            y(mask_b(mask_ab)) = 1;

            rng(0);

            K = min([5, na, nb]);

            if K < 2

                snr_auc = NaN;

            else

                cvp = cvpartition(y, 'KFold', K, 'Stratify', true);
                scores = nan(size(y));

                for k = 1:cvp.NumTestSets

                    tr = training(cvp, k);
                    te = test(cvp, k);

                    Mdl = fitcsvm(X(tr, :), y(tr), ...
                        'KernelFunction', 'linear', ...
                        'Standardize', true, ...
                        'ClassNames', [0 1]);

                    try

                        Mdl = fitPosterior(Mdl, X(tr, :), y(tr));
                        [~, s] = predict(Mdl, X(te, :));
                        scores(te) = s(:, 2);

                    catch

                        [~, s] = predict(Mdl, X(te, :));
                        scores(te) = -s(:, 1);

                    end

                end

                [~, ~, ~, snr_auc] = perfcurve(y, scores, 1);

            end

        end

        %% Store results in workspace only

        metrics = struct();
        metrics.monkey = MONKEY;
        metrics.session = session_name;
        metrics.snr_auc = snr_auc;
        metrics.snr_maze_pair = snr_maze_pair;
        metrics.maze_order = maze_order;

        result_idx = result_idx + 1;

        all_results(result_idx).monkey = MONKEY;
        all_results(result_idx).session = session_name;
        all_results(result_idx).maze_order = maze_order;
        all_results(result_idx).snr_maze_pair = snr_maze_pair;
        all_results(result_idx).all_neuron_data = all_neuron_data;
        all_results(result_idx).normalized_data = normalized_data;
        all_results(result_idx).coeff = coeff;
        all_results(result_idx).score = score;
        all_results(result_idx).latent = latent;
        all_results(result_idx).pca_reduced_data = pca_reduced_data;
        all_results(result_idx).linkage_tree_pca = linkage_tree_pca;
        all_results(result_idx).clusters = clusters;
        all_results(result_idx).cluster_distribution = cluster_distribution;
        all_results(result_idx).cluster_trial_ids = cluster_trial_ids;
        all_results(result_idx).cluster_trial_positions = cluster_trial_positions;
        all_results(result_idx).all_trial_ids = vertcat(maze_trial_ids{:});
        all_results(result_idx).null_distribution = null_distribution;
        all_results(result_idx).cluster_percentages = cluster_percentages;
        all_results(result_idx).maze_identity = maze_identity;
        all_results(result_idx).metrics = metrics;

        output_directory = fullfile(output_root, MONKEY);
        if ~exist(output_directory, 'dir')
            mkdir(output_directory);
        end

        output_file = fullfile(output_directory, ...
            ['_Cluster_Distributions_PCA_All_' session_name '.mat']);

        all_trial_ids = vertcat(maze_trial_ids{:});
        save(output_file, 'cluster_distribution', 'cluster_trial_ids', ...
            'cluster_trial_positions', 'all_trial_ids', ...
            'strategy_rows_all', '-v7.3');
        
        fprintf('Finished %s %s | SNR AUC maze %d vs maze %d = %.3f\n', ...
            MONKEY, session_name, snr_maze_pair(1), snr_maze_pair(2), snr_auc);
        fprintf('Saved cluster strategies: %s\n', output_file);

    end

end

fprintf('\nDone. Results are stored in workspace variable: all_results\n');
